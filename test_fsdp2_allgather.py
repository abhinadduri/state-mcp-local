"""Minimal test: can we do dist.all_gather_into_tensor after FSDP2 backward?"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Simple model with matrix params
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

    # Apply FSDP2 per-layer then root
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)
    fully_shard(model)

    if rank == 0:
        print(f"FSDP2 applied, mem={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Forward + backward
    x = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
    model.train()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    if rank == 0:
        print("Forward+backward done", flush=True)

    # Check gradient state
    for i, (name, p) in enumerate(model.named_parameters()):
        if rank == 0 and i < 5:
            print(f"  {name}: shape={tuple(p.shape)} grad={'yes' if p.grad is not None else 'no'} "
                  f"is_dtensor={hasattr(p.data, '_local_tensor')}", flush=True)

    # Test 1: all_gather with DEFAULT group
    if rank == 0:
        print("\n--- Test 1: all_gather on default group ---", flush=True)
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.grad is None:
            continue
        local_grad = p.grad._local_tensor.detach().clone()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        gathered = torch.empty(
            local_grad.shape[0] * world_size, *local_grad.shape[1:],
            dtype=local_grad.dtype, device=local_grad.device,
        )
        dist.all_gather_into_tensor(gathered, local_grad.contiguous())
        torch.cuda.synchronize()
        t1 = time.time()
        if rank == 0:
            print(f"  {name}: all_gather OK ({t1-t0:.4f}s) shape {tuple(local_grad.shape)} -> {tuple(gathered.shape)}", flush=True)

    # Test 2: all_gather with NEW group
    if rank == 0:
        print("\n--- Test 2: all_gather on new group ---", flush=True)
    new_group = dist.new_group()
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.grad is None:
            continue
        local_grad = p.grad._local_tensor.detach().clone()
        torch.cuda.synchronize()
        t0 = time.time()
        gathered = torch.empty(
            local_grad.shape[0] * world_size, *local_grad.shape[1:],
            dtype=local_grad.dtype, device=local_grad.device,
        )
        dist.all_gather_into_tensor(gathered, local_grad.contiguous(), group=new_group)
        torch.cuda.synchronize()
        t1 = time.time()
        if rank == 0:
            print(f"  {name}: all_gather OK ({t1-t0:.4f}s)", flush=True)

    # Test 3: multiple forward+backward then all_gather (like a training loop)
    if rank == 0:
        print("\n--- Test 3: loop of fwd+bwd+allgather ---", flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            local_grad = p.grad._local_tensor.detach().clone()
            gathered = torch.empty(
                local_grad.shape[0] * world_size, *local_grad.shape[1:],
                dtype=local_grad.dtype, device=local_grad.device,
            )
            dist.all_gather_into_tensor(gathered, local_grad.contiguous())
        torch.cuda.synchronize()
        t1 = time.time()
        # Now do a simple param update (on local tensor)
        for p in model.parameters():
            if p.grad is not None:
                p.data._local_tensor.add_(p.grad._local_tensor, alpha=-0.01)
        if rank == 0:
            print(f"  Step {step}: allgather all params {t1-t0:.4f}s loss={loss.item():.4f}", flush=True)

    if rank == 0:
        print("\nAll tests passed!", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
