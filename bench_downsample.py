"""Benchmark LatentTokenizer with num_downsample augmentations.

Tests at 100M scale (d=1024, nlayers=8) and 600M scale (d=2048, nlayers=16),
with k_top=4096 and num_downsample=1 vs 4.
"""

import gc
import time

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.state.emb.nn.tokenizer import LatentTokenizer
from src.state.emb.nn.model import StateEmbeddingModel
from src.state.emb.data import H5adSentenceDataset
from src.state.emb.train.trainer import get_embeddings
from src.state.emb.utils import get_embedding_cfg


def load_cfg():
    from hydra import compose, initialize_config_dir
    import os

    config_dir = os.path.join(os.path.dirname(__file__), "src", "state", "configs")
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = compose(config_name="state-defaults")
    return cfg


def build_model_and_loader(cfg, emsize, nlayers, k_top, batch_size, num_downsample):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.emsize = emsize
    cfg.model.d_hid = emsize * 2
    cfg.model.output_dim = emsize
    cfg.model.batch_size = batch_size
    cfg.model.tokenizer = "latent"
    cfg.model.n_latent = 256
    cfg.model.k_top = k_top
    cfg.model.nhead = 16
    cfg.model.nlayers = nlayers
    cfg.model.num_downsample = num_downsample

    tokenizer = LatentTokenizer(
        n_genes=get_embedding_cfg(cfg).num,
        n_latent=256,
        token_dim=get_embedding_cfg(cfg).size,
        d_model=emsize,
        nhead=16,
        d_hid=emsize * 2,
        nlayers=nlayers,
        output_dim=emsize,
        dropout=0.0,
        compiled=False,
        cfg=cfg,
    )

    collator = tokenizer.make_collator(cfg, is_train=True)

    dataset = H5adSentenceDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
    )

    model = StateEmbeddingModel(
        token_dim=get_embedding_cfg(cfg).size,
        d_model=emsize,
        nhead=16,
        d_hid=emsize * 2,
        nlayers=nlayers,
        output_dim=emsize,
        dropout=0.0,
        warmup_steps=1000,
        compiled=False,
        max_lr=1e-5,
        emb_size=get_embedding_cfg(cfg).size,
        collater=collator,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    model = model.cuda()

    all_pe = get_embeddings(cfg)
    all_pe.requires_grad = False
    model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    model = model.train()
    return model, loader, cfg


def benchmark_config(cfg_base, emsize, nlayers, k_top, batch_size, num_downsample,
                     warmup_steps=10, measure_steps=30):
    torch.cuda.empty_cache()
    gc.collect()

    model, loader, cfg = build_model_and_loader(
        cfg_base, emsize, nlayers, k_top, batch_size, num_downsample
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    # Effective batch = batch_size * num_downsample (collator duplicates)
    eff_batch = batch_size * num_downsample

    loader_iter = iter(loader)
    scaler = torch.amp.GradScaler()

    print(f"  Params: {n_trainable:.1f}M trainable / {n_total:.1f}M total | eff_batch={eff_batch}")
    print(f"  Warming up ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        batch = next(loader_iter)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, i)
        scaler.scale(loss).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print(f"  Measuring ({measure_steps} steps)...")
    k_actual_list = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(measure_steps):
        batch = next(loader_iter)
        if hasattr(batch, 'gene_indices'):
            k_actual_list.append(batch.gene_indices.shape[1])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, warmup_steps + i)
        scaler.scale(loss).backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # cells/sec counts unique cells entering the dataloader (before augmentation)
    total_cells = measure_steps * batch_size
    cells_per_sec = total_cells / elapsed
    ms_per_step = (elapsed / measure_steps) * 1000
    avg_k = sum(k_actual_list) / len(k_actual_list) if k_actual_list else 0

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    del model, loader, loader_iter
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "emsize": emsize,
        "nlayers": nlayers,
        "k_top": k_top,
        "batch_size": batch_size,
        "num_ds": num_downsample,
        "eff_batch": eff_batch,
        "trainable_M": n_trainable,
        "avg_k": avg_k,
        "ms_per_step": ms_per_step,
        "cells_per_sec": cells_per_sec,
        "mem_GB": mem_gb,
    }


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    cfg = load_cfg()

    # (emsize, nlayers, k_top, batch_size, num_downsample)
    configs = [
        # 100M scale (d=1024, nlayers=8)
        (1024, 8, 4096, 128, 1),   # baseline
        (1024, 8, 4096, 128, 4),   # with augmentation
        (1024, 8, 4096, 32, 4),    # B=32 so eff_batch=128
        # 600M scale (d=2048, nlayers=16)
        (2048, 16, 4096, 64, 1),   # baseline (start conservative on B)
        (2048, 16, 4096, 32, 1),   # smaller B if 64 OOMs
        (2048, 16, 4096, 16, 4),   # with augmentation, eff_batch=64
        (2048, 16, 4096, 32, 4),   # with augmentation, eff_batch=128
    ]

    results = []
    for emsize, nlayers, k_top, bs, nds in configs:
        label = f"d={emsize} L={nlayers} k={k_top} B={bs} ds={nds}"
        print(f"\n{'='*60}")
        print(f"Benchmarking: {label}")
        print(f"{'='*60}")
        try:
            r = benchmark_config(cfg, emsize, nlayers, k_top, bs, nds)
            results.append(r)
            print(f"  → {r['cells_per_sec']:.0f} cells/sec | {r['ms_per_step']:.0f}ms/step | eff_B={r['eff_batch']} | {r['mem_GB']:.1f} GB")
        except Exception as e:
            print(f"  → FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "emsize": emsize, "nlayers": nlayers, "k_top": k_top,
                "batch_size": bs, "num_ds": nds, "eff_batch": bs * nds,
                "error": str(e),
            })

    print(f"\n\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<40} {'cells/s':>8} {'ms/step':>8} {'eff_B':>6} {'Mem GB':>7} {'Params':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*8}")
    for r in results:
        if "error" in r:
            label = f"d={r['emsize']} L={r['nlayers']} B={r['batch_size']} ds={r['num_ds']}"
            print(f"{label:<40} {'FAILED':>8}")
        else:
            label = f"d={r['emsize']} L={r['nlayers']} B={r['batch_size']} ds={r['num_ds']}"
            print(f"{label:<40} {r['cells_per_sec']:>8.0f} {r['ms_per_step']:>8.0f} {r['eff_batch']:>6} {r['mem_GB']:>7.1f} {r['trainable_M']:>7.1f}M")


if __name__ == "__main__":
    main()
