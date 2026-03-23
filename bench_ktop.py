"""Benchmark LatentTokenizer throughput with and without k_top truncation.

Tests 4 configurations:
  - d_model=512,  k_top=None (all ~19K genes)
  - d_model=512,  k_top=4096
  - d_model=1024, k_top=None
  - d_model=1024, k_top=4096

Measures forward+backward throughput (cells/sec) on real data.
"""

import gc
import time

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.state.emb.nn.tokenizer import LatentTokenizer, LatentBatch
from src.state.emb.nn.model import StateEmbeddingModel
from src.state.emb.data import H5adSentenceDataset
from src.state.emb.train.trainer import get_embeddings
from src.state.emb.utils import get_embedding_cfg, get_dataset_cfg


def load_cfg():
    from hydra import compose, initialize_config_dir
    import os

    config_dir = os.path.join(os.path.dirname(__file__), "src", "state", "configs")
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = compose(config_name="state-defaults")
    return cfg


def build_model_and_loader(cfg, emsize, k_top, batch_size=128):
    """Build LatentTokenizer model + dataloader for a given config."""
    # Override config
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.emsize = emsize
    cfg.model.d_hid = emsize * 2
    cfg.model.output_dim = emsize
    cfg.model.batch_size = batch_size
    cfg.model.tokenizer = "latent"
    cfg.model.n_latent = 256
    cfg.model.k_top = k_top
    cfg.model.nhead = 16

    n_latent = 256
    tokenizer = LatentTokenizer(
        n_genes=get_embedding_cfg(cfg).num,
        n_latent=n_latent,
        token_dim=get_embedding_cfg(cfg).size,
        d_model=emsize,
        nhead=cfg.model.nhead,
        d_hid=emsize * 2,
        nlayers=cfg.model.nlayers,
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
        nhead=cfg.model.nhead,
        d_hid=emsize * 2,
        nlayers=cfg.model.nlayers,
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

    # Load frozen protein embeddings
    all_pe = get_embeddings(cfg)
    all_pe.requires_grad = False
    model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    model = model.train()
    return model, loader, cfg


def benchmark_config(cfg_base, emsize, k_top, batch_size=128, warmup_steps=10, measure_steps=30):
    """Run forward+backward and measure throughput."""
    torch.cuda.empty_cache()
    gc.collect()

    model, loader, cfg = build_model_and_loader(cfg_base, emsize, k_top, batch_size)

    # Count params
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    loader_iter = iter(loader)

    scaler = torch.amp.GradScaler()

    # Warmup
    print(f"  Warming up ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        batch = next(loader_iter)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, i)
        scaler.scale(loss).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Measure
    print(f"  Measuring ({measure_steps} steps)...")
    k_actual_list = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(measure_steps):
        batch = next(loader_iter)
        # Track actual k_max per step
        if hasattr(batch, 'gene_indices'):
            k_actual_list.append(batch.gene_indices.shape[1])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, warmup_steps + i)
        scaler.scale(loss).backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_cells = measure_steps * batch_size
    cells_per_sec = total_cells / elapsed
    ms_per_step = (elapsed / measure_steps) * 1000
    avg_k = sum(k_actual_list) / len(k_actual_list) if k_actual_list else 0

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    # Cleanup
    del model, loader, loader_iter
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "emsize": emsize,
        "k_top": k_top or "all",
        "avg_k": avg_k,
        "params_M": n_params,
        "batch_size": batch_size,
        "ms_per_step": ms_per_step,
        "cells_per_sec": cells_per_sec,
        "mem_GB": mem_gb,
    }


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    cfg = load_cfg()

    configs = [
        (512, None, 128),    # d=512, all genes, B=128
        (512, 4096, 128),    # d=512, top-4096, B=128
        (1024, None, 64),    # d=1024, all genes, B=64 (OOMs at 128)
        (1024, 4096, 128),   # d=1024, top-4096, B=128
    ]

    results = []
    for emsize, k_top, bs in configs:
        label = f"d={emsize}, k_top={k_top or 'all'}, B={bs}"
        print(f"\n{'='*60}")
        print(f"Benchmarking: {label}")
        print(f"{'='*60}")
        try:
            r = benchmark_config(cfg, emsize, k_top, batch_size=bs)
            results.append(r)
            print(f"  → {r['cells_per_sec']:.0f} cells/sec | {r['ms_per_step']:.0f}ms/step | avg_k={r['avg_k']:.0f} | {r['mem_GB']:.1f} GB | {r['params_M']:.1f}M params")
        except Exception as e:
            print(f"  → FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"emsize": emsize, "k_top": k_top or "all", "error": str(e)})

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<30} {'cells/sec':>10} {'ms/step':>10} {'avg_k':>8} {'Mem GB':>8} {'Params':>8}")
    print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        if "error" in r:
            print(f"d={r['emsize']}, k={r['k_top']:<18} {'FAILED':>10}")
        else:
            label = f"d={r['emsize']}, k={r['k_top']}"
            print(f"{label:<30} {r['cells_per_sec']:>10.0f} {r['ms_per_step']:>10.0f} {r['avg_k']:>8.0f} {r['mem_GB']:>8.1f} {r['params_M']:>7.1f}M")


if __name__ == "__main__":
    main()
