"""Test: functional collectives after FSDP2 backward (vs manual dist.all_gather_into_tensor)."""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._functional_collectives import all_gather_tensor


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build a model similar in structure to the real one (many layers)
    layers = []
    for _ in range(32):
        layers.extend([nn.Linear(1024, 1024, bias=False), nn.ReLU()])
    model = nn.Sequential(*layers).to(torch.bfloat16).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model: {n_params/1e6:.1f}M params ({len(list(model.parameters()))} params)", flush=True)

    # Apply FSDP2 per-layer
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)
    fully_shard(model)

    # Forward + backward
    x = torch.randn(32, 1024, dtype=torch.bfloat16, device="cuda")
    model.train()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()

    params = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if rank == 0:
        print(f"After backward: {len(params)} params with grads", flush=True)

    # Test 1: functional collective all_gather_tensor
    if rank == 0:
        print("\n--- Test 1: functional all_gather_tensor ---", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for name, p in params:
        local_grad = p.grad._local_tensor.detach().clone()
        # Functional collective: integrates with FSDP2's stream management
        full_grad = all_gather_tensor(local_grad, gather_dim=0, group=dist.group.WORLD)
    torch.cuda.synchronize()
    t1 = time.time()
    if rank == 0:
        print(f"  {len(params)} params: {t1-t0:.3f}s ({(t1-t0)*1000/len(params):.2f}ms/param)", flush=True)

    # Test 2: manual dist.all_gather_into_tensor
    if rank == 0:
        print("\n--- Test 2: manual dist.all_gather_into_tensor ---", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for name, p in params:
        local_grad = p.grad._local_tensor.detach().clone()
        gathered = torch.empty(
            local_grad.shape[0] * world_size, *local_grad.shape[1:],
            dtype=local_grad.dtype, device=local_grad.device,
        )
        dist.all_gather_into_tensor(gathered, local_grad.contiguous())
    torch.cuda.synchronize()
    t1 = time.time()
    if rank == 0:
        print(f"  {len(params)} params: {t1-t0:.3f}s ({(t1-t0)*1000/len(params):.2f}ms/param)", flush=True)

    # Test 3: Full Muon-like loop with functional collectives
    if rank == 0:
        print("\n--- Test 3: Muon-like loop (functional collectives + Newton-Schulz) ---", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for name, p in params:
        local_grad = p.grad._local_tensor.detach().clone()
        # All-gather
        full_grad = all_gather_tensor(local_grad, gather_dim=0, group=dist.group.WORLD)
        # Newton-Schulz (5 steps on regular tensor)
        x = full_grad
        if x.shape[0] > x.shape[1]:
            x = x.transpose(0, 1)
        x = x / (torch.linalg.norm(x.float()) + 1e-7)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(5):
            gram = x @ x.transpose(0, 1)
            x = a * x + (b * gram + c * gram @ gram) @ x
    torch.cuda.synchronize()
    t1 = time.time()
    if rank == 0:
        print(f"  {len(params)} params: {t1-t0:.3f}s ({(t1-t0)*1000/len(params):.2f}ms/param)", flush=True)

    if rank == 0:
        print("\nAll tests passed!", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
