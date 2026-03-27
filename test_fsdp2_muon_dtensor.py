"""Test: Muon optimizer with DTensor-native operations after FSDP2."""
import os
import time
import sys
sys.path.insert(0, "/home/aadduri/state-mcp-local")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from src.state.tx.optim import MuonWithAuxAdamW


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Simple model with matrix params (like a small transformer)
    model = nn.Sequential(
        nn.Linear(4096, 4096, bias=False),
        nn.ReLU(),
        nn.Linear(4096, 4096, bias=False),
        nn.ReLU(),
        nn.Linear(4096, 4096, bias=False),
    ).to(torch.bfloat16).cuda()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params/1e6:.1f}M params", flush=True)

    # Apply FSDP2
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)
    fully_shard(model)

    if rank == 0:
        print(f"FSDP2 applied", flush=True)

    # Build Muon optimizer (all params are matrix-like)
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    if rank == 0:
        print(f"Muon: {len(muon_params)} params, AdamW: {len(adamw_params)} params", flush=True)
        for i, p in enumerate(muon_params):
            from torch.distributed.tensor import DTensor
            is_dt = isinstance(p, DTensor)
            print(f"  param {i}: shape={tuple(p.shape)} is_dtensor={is_dt}", flush=True)

    optimizer = MuonWithAuxAdamW(
        muon_params, adamw_params,
        lr=0.001, weight_decay=0.01,
    )

    # Training loop
    x = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
    model.train()

    for step in range(5):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        if rank == 0:
            print(f"Step {step}: fwd+bwd={t1-t0:.3f}s  optim={t2-t1:.3f}s  loss={loss.item():.4f}", flush=True)

    if rank == 0:
        print(f"\nPeak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        print("Done!", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
