"""Test: DTensor.full_tensor() for Muon after FSDP2 backward."""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import DTensor


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # 32-layer model like the real one
    layers = []
    for _ in range(32):
        layers.extend([nn.Linear(1024, 1024, bias=False), nn.ReLU()])
    model = nn.Sequential(*layers).to(torch.bfloat16).cuda()

    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)
    fully_shard(model)

    # Forward + backward
    x = torch.randn(32, 1024, dtype=torch.bfloat16, device="cuda")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()

        torch.cuda.synchronize()
        t0 = time.time()

        # Muon-like step using full_tensor()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            # Get the full gradient via DTensor.full_tensor()
            full_grad = p.grad.full_tensor()  # all-gathers via DTensor internals

            # Newton-Schulz on regular tensor
            x_ns = full_grad
            if x_ns.shape[0] > x_ns.shape[1]:
                x_ns = x_ns.transpose(0, 1)
            x_ns = x_ns / (torch.linalg.norm(x_ns.float()) + 1e-7)
            a, b, c = 3.4445, -4.7750, 2.0315
            for _ in range(5):
                gram = x_ns @ x_ns.transpose(0, 1)
                x_ns = a * x_ns + (b * gram + c * gram @ gram) @ x_ns

            # Extract local shard and apply
            shard_rows = p.data._local_tensor.shape[0]
            if full_grad.shape[0] > full_grad.shape[1]:
                # Was transposed for NS
                local_orth = x_ns.transpose(0, 1)[rank * shard_rows:(rank+1) * shard_rows]
            else:
                local_orth = x_ns[rank * shard_rows:(rank+1) * shard_rows]
            p.data._local_tensor.add_(local_orth.to(p.data._local_tensor.dtype), alpha=-0.01)

        torch.cuda.synchronize()
        t1 = time.time()
        if rank == 0:
            print(f"Step {step}: muon step {t1-t0:.3f}s loss={loss.item():.4f}", flush=True)

    if rank == 0:
        print("Done!", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
