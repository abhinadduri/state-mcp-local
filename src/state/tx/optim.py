from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None


def _is_dtensor(t: torch.Tensor) -> bool:
    """Check if a tensor is an FSDP2 DTensor (sharded parameter)."""
    return DTensor is not None and isinstance(t, DTensor)


def _reshape_matrix(param: torch.Tensor) -> torch.Tensor:
    if param.ndim == 2:
        return param
    if param.ndim < 2:
        raise ValueError(f"Muon requires matrix-like parameters; got shape {tuple(param.shape)}")
    return param.reshape(param.shape[0], -1)


def _orthogonalize_update(matrix: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """Newton-Schulz orthogonalization on a 2D matrix (regular tensor)."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D update matrix, got shape {tuple(matrix.shape)}")

    transposed = False
    # Keep in input dtype (bf16) for 2x faster tensor-core matmuls vs fp32/TF32.
    # Only the norm computation upcasts to fp32 to avoid overflow.
    x = matrix
    if x.shape[0] > x.shape[1]:
        x = x.transpose(0, 1)
        transposed = True

    x = x / (torch.linalg.norm(x.float()) + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(1, steps)):
        gram = x @ x.transpose(0, 1)
        x = a * x + (b * gram + c * gram @ gram) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.to(dtype=matrix.dtype)


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Apply Muon to matrix-like parameters and AdamW to the remainder.

    Works with both regular tensors (single-GPU, DDP) and FSDP2 DTensors.
    For FSDP2: momentum/Nesterov are computed on DTensors (element-wise, local).
    Newton-Schulz uses DTensor.full_tensor() for FSDP2-native all-gather, then
    runs on regular tensors to avoid per-operation distributed dispatch overhead.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.Tensor],
        adamw_params: Iterable[torch.Tensor],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        muon_eps: float = 1e-7,
        adamw_lr: float | None = None,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        param_groups: list[dict[str, object]] = []
        if muon_params:
            param_groups.append(
                {
                    "params": muon_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "kind": "muon",
                    "momentum": momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                    "eps": muon_eps,
                }
            )
        if adamw_params:
            param_groups.append(
                {
                    "params": adamw_params,
                    "lr": adamw_lr if adamw_lr is not None else lr,
                    "weight_decay": weight_decay,
                    "kind": "adamw",
                    "betas": adamw_betas,
                    "eps": adamw_eps,
                }
            )
        if not param_groups:
            raise ValueError("MuonWithAuxAdamW requires at least one trainable parameter.")

        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # Timing instrumentation for FSDP2 debugging (first 5 steps only)
        _diag = getattr(self, '_fsdp_diag_count', 0)
        _do_diag = _diag < 5 and any(_is_dtensor(p) for g in self.param_groups for p in g["params"][:1])
        if _do_diag:
            import time as _time
            torch.cuda.synchronize()
        for group in self.param_groups:
            kind = str(group.get("kind", "adamw")).lower()
            if _do_diag:
                _gt0 = _time.time()
            if kind == "muon":
                self._step_muon_group(group)
            elif kind == "adamw":
                self._step_adamw_group(group)
            else:
                raise ValueError(f"Unsupported optimizer group kind {kind!r}.")
            if _do_diag:
                torch.cuda.synchronize()
                _gt1 = _time.time()
                if dist.get_rank() == 0:
                    print(f"  [optim] {kind} group: {_gt1-_gt0:.3f}s ({len(group['params'])} params)", flush=True)
        if _do_diag:
            self._fsdp_diag_count = _diag + 1
        return loss

    def _step_muon_group(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        momentum = float(group.get("momentum", 0.95))
        nesterov = bool(group.get("nesterov", True))
        ns_steps = int(group.get("ns_steps", 5))
        eps = float(group.get("eps", 1e-7))

        # Separate FSDP2 (DTensor) params from regular params
        fsdp_params = []
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients.")

            if _is_dtensor(param.data):
                fsdp_params.append(param)
            else:
                # Standard single-GPU / DDP path
                param_data = param.detach()
                if weight_decay != 0.0:
                    param_data.mul_(1.0 - lr * weight_decay)

                grad_view = _reshape_matrix(param.grad.detach())
                param_view = _reshape_matrix(param_data)
                state = self.state[param]
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(grad_view)
                    state["momentum_buffer"] = momentum_buffer

                momentum_buffer.mul_(momentum).add_(grad_view)
                update = grad_view.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer
                orthogonal_update = _orthogonalize_update(update, steps=ns_steps, eps=eps)
                scale = math.sqrt(max(1.0, orthogonal_update.shape[0] / max(1, orthogonal_update.shape[1])))
                param_view.add_(orthogonal_update, alpha=-lr * scale)

        # FSDP2 path: per-param with DTensor.full_tensor() all-gather
        if fsdp_params:
            for param in fsdp_params:
                self._step_muon_param_fsdp(
                    param, lr, weight_decay, momentum, nesterov, ns_steps, eps,
                )

    def _step_muon_param_fsdp(
        self, param, lr, weight_decay, momentum_val, nesterov, ns_steps, eps,
    ) -> None:
        """FSDP2-aware Muon step using DTensor.full_tensor().

        Momentum and Nesterov are computed on DTensors (element-wise, local).
        full_tensor() does an FSDP2-native all-gather to get the complete update
        matrix.  Newton-Schulz then runs on a regular tensor, avoiding per-
        iteration distributed matmul dispatch overhead.
        """
        # Weight decay on DTensor (element-wise, local)
        if weight_decay != 0.0:
            param.detach().mul_(1.0 - lr * weight_decay)

        grad_view = _reshape_matrix(param.grad.detach())

        # Momentum buffer (DTensor, same placement as grad — element-wise, local)
        state = self.state[param]
        mb = state.get("momentum_buffer")
        if mb is None:
            mb = torch.zeros_like(grad_view)
            state["momentum_buffer"] = mb
        mb.mul_(momentum_val).add_(grad_view)

        # Nesterov combination (DTensor element-wise, local)
        update = grad_view.add(mb, alpha=momentum_val) if nesterov else mb

        # FSDP2-native all-gather: reconstructs full matrix from shards.
        full_update = update.full_tensor()

        # Newton-Schulz on regular tensor (no distributed overhead)
        orth = _orthogonalize_update(full_update, steps=ns_steps, eps=eps)
        scale = math.sqrt(max(1.0, orth.shape[0] / max(1, orth.shape[1])))

        # Extract this rank's shard and apply to local param data
        rank = dist.get_rank()
        shard_rows = param.data._local_tensor.shape[0]
        local_orth = orth[rank * shard_rows : (rank + 1) * shard_rows]
        _reshape_matrix(param.data._local_tensor).add_(local_orth, alpha=-lr * scale)

    def _step_adamw_group(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients.")

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = state["step"]

            grad_data = grad.detach()
            if weight_decay != 0.0:
                param.detach().mul_(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad_data, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).add_(grad_data * grad_data, alpha=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
            denom.add_(eps)
            step_size = lr / bias_correction1

            param.detach().add_(exp_avg / denom, alpha=-step_size)
