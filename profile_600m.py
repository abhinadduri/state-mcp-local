#!/usr/bin/env python
"""Profile 600M SE model training to get kernel-level breakdown.

Runs 30 warmup steps (for torch.compile) then profiles 50 steps.
Outputs a table of top CUDA kernels by time and a Chrome trace.
"""

import os
import sys
import time

import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))

from src.state.emb.nn.model import StateEmbeddingModel
from src.state.emb.nn.tokenizer import SentenceTokenizer, LatentTokenizer
from src.state.emb.data import H5adSentenceDataset
from src.state.emb.utils import get_embedding_cfg, get_dataset_cfg
from src.state.emb.train.trainer import get_embeddings, build_optimizer_and_scheduler


def load_config():
    from src.state.__main__ import load_hydra_config
    return load_hydra_config("emb", overrides=["scale=600m"])


def main():
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    cfg.experiment.compiled = True
    batch_size = cfg.model.batch_size  # 128 for 600m

    emb_cfg = get_embedding_cfg(cfg)
    ds_cfg = get_dataset_cfg(cfg)

    # Build tokenizer
    tokenizer_type = getattr(cfg.model, "tokenizer", "sentence")
    if tokenizer_type == "latent":
        n_latent = getattr(cfg.model, "n_latent", 256)
        tokenizer = LatentTokenizer(
            n_genes=emb_cfg.num, n_latent=n_latent,
            token_dim=emb_cfg.size, d_model=cfg.model.emsize,
            nhead=cfg.model.nhead, d_hid=cfg.model.d_hid,
            nlayers=cfg.model.nlayers, output_dim=cfg.model.output_dim,
            dropout=cfg.model.dropout, compiled=False, cfg=cfg,
        )
        print(f"Using LatentTokenizer: n_genes={emb_cfg.num}, n_latent={n_latent}")
    else:
        tokenizer = SentenceTokenizer(
            token_dim=emb_cfg.size, d_model=cfg.model.emsize,
            nhead=cfg.model.nhead, d_hid=cfg.model.d_hid,
            nlayers=cfg.model.nlayers, output_dim=cfg.model.output_dim,
            dropout=cfg.model.dropout, compiled=False, cfg=cfg,
        )

    # Collator
    train_collator = tokenizer.make_collator(cfg, is_train=True)

    # Dataset
    train_dataset = H5adSentenceDataset(cfg)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=train_collator, num_workers=16,
        persistent_workers=True, pin_memory=True, prefetch_factor=4, drop_last=True,
    )

    # Model
    model = StateEmbeddingModel(
        token_dim=emb_cfg.size, d_model=cfg.model.emsize,
        nhead=cfg.model.nhead, d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers, output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout, warmup_steps=0, compiled=False,
        max_lr=cfg.optimizer.max_lr, emb_size=emb_cfg.size,
        collater=train_collator, cfg=cfg, tokenizer=tokenizer,
    )
    model = model.cuda()

    # Embeddings
    all_pe = get_embeddings(cfg).to(torch.bfloat16)
    all_pe.requires_grad = False
    model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    # Warm ESM2 cache (LatentTokenizer only)
    if hasattr(model.tokenizer, "_get_esm2_proj_table"):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.tokenizer._get_esm2_proj_table(model.tokenizer.pe_embedding.weight.device)

    # Compile
    print("Compiling tokenizer and decoder...")
    model.tokenizer = torch.compile(model.tokenizer)
    model._decode = torch.compile(model._decode)

    # Optimizer
    grad_accum = cfg.optimizer.gradient_accumulation_steps
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps=200)

    model.train()
    data_iter = iter(train_loader)

    def get_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    # Warmup: 30 optimizer steps
    warmup_steps = 30
    print(f"Warming up for {warmup_steps} optimizer steps ({warmup_steps * grad_accum} microsteps)...")
    for step in range(warmup_steps):
        for _ in range(grad_accum):
            batch = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)
            (loss / grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    print(f"Warmup done. Loss: {loss.item():.4f}")
    torch.cuda.reset_peak_memory_stats()

    # Profile: 50 optimizer steps
    profile_steps = 50
    print(f"\nProfiling {profile_steps} optimizer steps ({profile_steps * grad_accum} microsteps)...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        t0 = time.time()
        for step in range(profile_steps):
            for _ in range(grad_accum):
                batch = get_batch()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch)
                (loss / grad_accum).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        wall_time = time.time() - t0

    cells_per_sec = batch_size * grad_accum * profile_steps / wall_time
    avg_step = wall_time / profile_steps
    print(f"\nWall time: {wall_time:.1f}s for {profile_steps} steps")
    print(f"Avg step time: {avg_step*1000:.1f}ms")
    print(f"Cells/sec: {cells_per_sec:.0f}")

    # Print top CUDA kernels by total time
    print("\n" + "=" * 80)
    print("TOP 40 CUDA KERNELS BY TOTAL TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

    # Print top CPU operations
    print("\n" + "=" * 80)
    print("TOP 30 OPERATIONS BY CPU TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Export Chrome trace
    trace_dir = "/tmp/se_600m_profile"
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to: {trace_path}")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
