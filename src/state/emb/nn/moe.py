"""Mixture of Experts (MoE) components for the State Embedding model.

Replaces dense FFN layers with sparsely-activated expert FFNs.
Supports two modes:
- FSDP2 mode: padded bmm with stacked expert weights (default)
- Expert Parallel (EP) mode: all-to-all token dispatch, each GPU owns 1 expert

Key components:
- TopKRouter: Token-to-expert routing with load balancing and z-loss
- MoEFFN: Dropless MoE FFN with configurable backend
- MoETransformerEncoderLayer: Drop-in replacement for FlashTransformerEncoderLayer
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

log = logging.getLogger(__name__)


@torch.compiler.disable
def _ep_all_to_all(recv: torch.Tensor, send: torch.Tensor, group) -> torch.Tensor:
    """Thin wrapper so all_to_all_single is a minimal graph break for torch.compile."""
    dist.all_to_all_single(recv, send, group=group)
    return recv


class TopKRouter(nn.Module):
    """Token-to-expert router with auxiliary losses for balanced expert utilization."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # Small init prevents early routing imbalance (ST-MoE recommendation)
        nn.init.trunc_normal_(self.gate.weight, std=0.001)

    def forward(self, x: torch.Tensor):
        # Force float32 for numerical stability (ST-MoE: bfloat16 softmax causes instability)
        router_logits = self.gate(x).float()
        router_logits = router_logits.clamp(-20.0, 20.0)  # safety net for softmax
        scores = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        # Cast weights back to input dtype for downstream compute
        top_k_weights = top_k_weights.to(x.dtype)
        return top_k_weights, top_k_indices, router_logits


def load_balancing_loss(router_logits: torch.Tensor, top_k_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch Transformer load balancing loss: L = E * sum(f_i * P_i)."""
    scores = F.softmax(router_logits, dim=-1)
    num_tokens = router_logits.shape[0]
    one_hot = F.one_hot(top_k_indices, num_experts).float()
    tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)
    f = tokens_per_expert / (num_tokens * top_k_indices.shape[1])
    P = scores.mean(dim=0)
    return num_experts * (f * P).sum()


def _compute_balance_stats(router_logits: torch.Tensor, top_k_indices: torch.Tensor, num_experts: int):
    """Compute per-expert token fraction (f) and mean routing probability (P).

    Returns raw counts so they can be accumulated across micro-batches for
    global-batch load balancing (Qwen, ACL 2025).
    """
    scores = F.softmax(router_logits, dim=-1)
    num_tokens = router_logits.shape[0]
    one_hot = F.one_hot(top_k_indices, num_experts).float()
    tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)  # [E]
    score_sum = scores.sum(dim=0)  # [E]
    return tokens_per_expert, score_sum, num_tokens


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss (ST-MoE): penalizes large logits for stability."""
    log_z = torch.logsumexp(router_logits, dim=-1)
    return (log_z ** 2).mean()


class MoEFFN(nn.Module):
    """Dropless Mixture of Experts FFN.

    Supports two backends:
    - "bmm": padded batched matmul with stacked [E, d, h] weights (default)
    - "ep": expert parallelism — each GPU owns 1 expert, tokens dispatched via all-to-all

    EP mode is enabled by calling `enable_expert_parallel(process_group)` after init.
    """

    def __init__(
        self,
        d_model: int,
        d_hid: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        num_shared_experts: int = 0,
        n_layers: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hid = d_hid
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k)

        # Routed expert weights: [E, d_in, d_out] for bmm or sharded for EP
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, d_hid))
        self.b1 = nn.Parameter(torch.zeros(num_experts, 1, d_hid))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_hid, d_model))
        self.b2 = nn.Parameter(torch.zeros(num_experts, 1, d_model))
        import math
        residual_scale = 1.0 / math.sqrt(2 * n_layers)
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.w1[i])
            nn.init.kaiming_uniform_(self.w2[i])
            # Depth-scaled init on output projection (GPT-2/GPT-3 scheme)
            self.w2.data[i] *= residual_scale

        # Shared experts: always active, not routed (DeepSeek-style)
        self.num_shared_experts = num_shared_experts
        if num_shared_experts > 0:
            self.shared_w1 = nn.Linear(d_model, d_hid * num_shared_experts, bias=True)
            self.shared_w2 = nn.Linear(d_hid * num_shared_experts, d_model, bias=True)
        else:
            self.shared_w1 = None

        self.dropout_p = dropout
        self._aux_loss = None
        self._router_z_loss = None

        # Global-batch load balancing: accumulate stats across micro-batches.
        # Use register_buffer (persistent=False) so tensors move with .cuda()
        # but don't appear in state_dict.  Crucially, starting as real tensors
        # (not None) avoids a type-change that triggers torch.compile
        # recompilation on the first forward call.
        self.register_buffer("_accum_tokens_per_expert", torch.zeros(num_experts), persistent=False)
        self.register_buffer("_accum_score_sum", torch.zeros(num_experts), persistent=False)
        self._accum_num_tokens = 0

        # EP state (set by enable_expert_parallel)
        self._ep_group = None
        self._ep_rank = None
        self._ep_size = None

    def enable_expert_parallel(self, process_group):
        """Enable expert parallelism. Each rank owns num_experts/world_size experts.

        Supports num_experts > world_size (e.g., 32 experts on 8 GPUs = 4 per GPU).
        Call after model init but before FSDP2 wrapping.
        """
        self._ep_group = process_group
        self._ep_rank = dist.get_rank(process_group)
        self._ep_size = dist.get_world_size(process_group)

        assert self.num_experts % self._ep_size == 0, \
            f"num_experts ({self.num_experts}) must be divisible by EP size ({self._ep_size})"

        self._experts_per_rank = self.num_experts // self._ep_size

        # Slice expert weights to keep only this rank's experts
        start = self._ep_rank * self._experts_per_rank
        end = start + self._experts_per_rank
        with torch.no_grad():
            self.w1 = nn.Parameter(self.w1[start:end].clone())
            self.b1 = nn.Parameter(self.b1[start:end].clone())
            self.w2 = nn.Parameter(self.w2[start:end].clone())
            self.b2 = nn.Parameter(self.b2[start:end].clone())

        log.info(f"EP enabled: rank {self._ep_rank} owns experts {start}-{end-1} "
                 f"({self._experts_per_rank} per GPU)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        top_k_weights, top_k_indices, router_logits = self.router(x_flat)

        # Accumulate balance stats across micro-batches for global-batch loss.
        # Detach previous micro-batch stats so only the current micro-batch's
        # router gets gradients (previous graphs are already freed by backward).
        tpe, ss, nt = _compute_balance_stats(router_logits, top_k_indices, self.num_experts)
        self._accum_tokens_per_expert = self._accum_tokens_per_expert.detach() + tpe
        self._accum_score_sum = self._accum_score_sum.detach() + ss
        self._accum_num_tokens = self._accum_num_tokens + nt

        # Compute loss from accumulated global-batch stats
        f = self._accum_tokens_per_expert / (self._accum_num_tokens * top_k_indices.shape[1])
        P = self._accum_score_sum / self._accum_num_tokens
        self._aux_loss = self.num_experts * (f * P).sum()
        self._router_z_loss = router_z_loss(router_logits)

        if self._ep_group is not None:
            output = self._forward_ep(x_flat, top_k_weights, top_k_indices)
        else:
            output = self._forward_bmm(x_flat, top_k_weights, top_k_indices)

        # Shared expert: always-active FFN added to routed output
        if self.shared_w1 is not None:
            shared_out = self.shared_w2(F.gelu(self.shared_w1(x_flat)))
            output = output + shared_out

        return output.reshape(B, T, D)

    def _ep_pack(self, x_flat, top_k_weights, top_k_indices, N, E, K, C):
        """Phase 1: Route tokens and pack into padded [E, C, D] buffer.

        Compilable — no NCCL ops. Fuses ~20 small ops into 1-2 kernels.
        """
        D = x_flat.shape[1]
        M = N * K

        flat_experts = top_k_indices.reshape(-1)
        flat_weights = top_k_weights.reshape(-1)
        flat_token_idx = torch.arange(N, device=x_flat.device).unsqueeze(1).expand(-1, K).reshape(-1)

        sort_idx = flat_experts.argsort()
        sorted_token_idx = flat_token_idx[sort_idx]
        sorted_experts = flat_experts[sort_idx]
        sorted_weights = flat_weights[sort_idx]

        counts = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        counts.scatter_add_(0, sorted_experts, torch.ones(M, dtype=torch.long, device=x_flat.device))

        offsets = torch.zeros(E + 1, dtype=torch.long, device=x_flat.device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        expert_ids = torch.repeat_interleave(torch.arange(E, device=x_flat.device), counts)
        positions = torch.arange(M, device=x_flat.device) - offsets[expert_ids]

        weighted = (x_flat[sorted_token_idx] * sorted_weights.unsqueeze(-1)).to(torch.bfloat16)
        padded_send = weighted.new_zeros(E, C, D)
        padded_send[expert_ids, positions] = weighted

        return padded_send, sort_idx, expert_ids, positions

    def _ep_unpack(self, recv_back_flat, sort_idx, expert_ids, positions, E, C, N, K, dtype):
        """Phase 5: Unpack results from padded buffer.

        Compilable — no NCCL ops. Fuses ~8 small ops into 1-2 kernels.
        """
        D = recv_back_flat.shape[1]
        recv_back = recv_back_flat.reshape(E, C, D)
        results = recv_back[expert_ids, positions].to(dtype)

        unsort_idx = sort_idx.argsort()
        output = results[unsort_idx].reshape(N, K, D).sum(dim=1)
        return output

    def _forward_ep(self, x_flat, top_k_weights, top_k_indices):
        """Expert parallel forward with optional Triton-fused pack/unpack."""
        N, D = x_flat.shape
        E = self.num_experts
        K = self.top_k
        ep_group = self._ep_group
        ep_size = self._ep_size
        epr = self._experts_per_rank
        M = N * K

        # Dropless: capacity = max tokens assigned to any expert.
        # Synced across EP ranks so all-to-all buffers match.
        _flat_experts = top_k_indices.reshape(-1)
        _counts = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        _counts.scatter_add_(0, _flat_experts, torch.ones(M, dtype=torch.long, device=x_flat.device))
        _BUCKET = 64
        local_max = _counts.max()
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=ep_group)
        C = max(local_max.item(), 1)
        C = ((C + _BUCKET - 1) // _BUCKET) * _BUCKET

        # Phase 1: Pack — use Triton if available (1 kernel vs ~20 PyTorch ops)
        try:
            from .moe_triton import HAS_TRITON, triton_ep_pack, triton_ep_unpack
        except ImportError:
            HAS_TRITON = False

        if HAS_TRITON:
            padded_send, token_ids, expert_ids, positions = triton_ep_pack(
                x_flat, top_k_weights, top_k_indices, E, C)
        else:
            padded_send, sort_idx, expert_ids, positions = self._ep_pack(
                x_flat, top_k_weights, top_k_indices, N, E, K, C)

        # Phase 2: All-to-all dispatch (NCCL)
        send_flat = padded_send.reshape(-1, D)
        if not send_flat.is_contiguous():
            send_flat = send_flat.contiguous()
        recv_flat = torch.empty_like(send_flat)
        _ep_all_to_all(recv_flat, send_flat, ep_group)

        # Phase 3: Local expert BMM
        recv_merged = recv_flat.reshape(ep_size, epr, C, D).permute(1, 0, 2, 3).reshape(epr, ep_size * C, D)
        h = torch.bmm(recv_merged, self.w1) + self.b1
        h = F.gelu(h)
        if self.dropout_p > 0 and self.training:
            h = F.dropout(h, p=self.dropout_p, training=True)
        out = torch.bmm(h, self.w2) + self.b2

        # Phase 4: All-to-all combine (NCCL)
        send_back = out.reshape(epr, ep_size, C, D).permute(1, 0, 2, 3).contiguous().reshape(-1, D)
        recv_back_flat = torch.empty_like(send_back)
        _ep_all_to_all(recv_back_flat, send_back, ep_group)

        # Phase 5: Unpack
        if HAS_TRITON:
            return triton_ep_unpack(recv_back_flat, token_ids, expert_ids, positions, N, E, C, K)
        else:
            return self._ep_unpack(recv_back_flat, sort_idx, expert_ids, positions, E, C, N, K, x_flat.dtype)

    def _forward_bmm(self, x_flat, top_k_weights, top_k_indices):
        """Padded bmm forward (no EP), dropless.

        Capacity = max tokens per expert, bucketed to 64 for torch.compile
        shape stability.  No tokens are dropped.
        """
        N, D = x_flat.shape
        E = self.num_experts
        K = self.top_k

        flat_token_idx = torch.arange(N, device=x_flat.device).unsqueeze(1).expand(-1, K).reshape(-1)
        flat_experts = top_k_indices.reshape(-1)
        flat_weights = top_k_weights.reshape(-1)
        M = N * K

        sort_idx = flat_experts.argsort()
        sorted_token_idx = flat_token_idx[sort_idx]
        sorted_experts = flat_experts[sort_idx]
        sorted_weights = flat_weights[sort_idx]

        expert_counts = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        expert_counts.scatter_add_(0, sorted_experts, torch.ones(M, dtype=torch.long, device=x_flat.device))

        # Dropless: capacity = max tokens assigned to any single expert,
        # bucketed to a multiple of 64 for stable shapes under torch.compile.
        # No tokens are dropped — every token reaches its assigned expert.
        _BUCKET = 64
        capacity = max(expert_counts.max().item(), 1)
        capacity = ((capacity + _BUCKET - 1) // _BUCKET) * _BUCKET

        offsets = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        torch.cumsum(expert_counts[:-1], dim=0, out=offsets[1:])
        global_pos = torch.arange(M, device=x_flat.device)
        positions = global_pos - offsets[sorted_experts]

        padded_tokens = x_flat.new_zeros(E, capacity, D)
        padded_weights = x_flat.new_zeros(E, capacity)
        padded_out_idx = torch.zeros(E, capacity, dtype=torch.long, device=x_flat.device)

        padded_tokens[sorted_experts, positions] = x_flat[sorted_token_idx]
        padded_weights[sorted_experts, positions] = sorted_weights
        padded_out_idx[sorted_experts, positions] = sorted_token_idx

        h = torch.bmm(padded_tokens, self.w1) + self.b1
        h = F.gelu(h)
        if self.dropout_p > 0 and self.training:
            h = F.dropout(h, p=self.dropout_p, training=True)
        expert_out = torch.bmm(h, self.w2) + self.b2
        expert_out = expert_out * padded_weights.unsqueeze(-1)

        output = torch.zeros_like(x_flat)
        flat_idx = padded_out_idx.reshape(-1).unsqueeze(-1).expand(-1, D)
        output.scatter_add_(0, flat_idx, expert_out.reshape(-1, D))
        return output

    @property
    def aux_losses(self):
        losses = {}
        if self._aux_loss is not None:
            losses["load_balance"] = self._aux_loss
        if self._router_z_loss is not None:
            losses["router_z"] = self._router_z_loss
        return losses

    def reset_balance_stats(self):
        """Reset accumulated balance stats after each optimizer step."""
        self._accum_tokens_per_expert = torch.zeros_like(self._accum_tokens_per_expert)
        self._accum_score_sum = torch.zeros_like(self._accum_score_sum)
        self._accum_num_tokens = 0


class MoETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with MoE FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_shared_experts: int = 0,
        n_layers: int = 1,
    ):
        super().__init__()
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.moe_ffn = MoEFFN(
            d_model=d_model,
            d_hid=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            num_shared_experts=num_shared_experts,
            n_layers=n_layers,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Depth-scaled init on attention output projection
        import math
        with torch.no_grad():
            self.out_proj.weight *= 1.0 / math.sqrt(2 * n_layers)

    def _attention_block(self, src):
        """Attention sub-block — compilable (no NCCL ops)."""
        normed = self.norm1(src)
        qkv = self.qkv_proj(normed)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        head_dim = self.d_model // self.nhead
        B_size, T_size = src.size(0), src.size(1)
        q = q.view(B_size, T_size, self.nhead, head_dim).transpose(1, 2)
        k = k.view(B_size, T_size, self.nhead, head_dim).transpose(1, 2)
        v = v.view(B_size, T_size, self.nhead, head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B_size, T_size, self.d_model)
        attn_output = self.out_proj(attn_output)
        return src + self.dropout_layer(attn_output)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self._attention_block(src)
        normed2 = self.norm2(src)
        ff_output = self.moe_ffn(normed2)
        src = src + self.dropout_layer(ff_output)
        return src

    @property
    def aux_losses(self):
        return self.moe_ffn.aux_losses


def enable_expert_parallel(model: nn.Module, process_group):
    """Enable EP on all MoE layers in the model."""
    for module in model.modules():
        if isinstance(module, MoEFFN):
            module.enable_expert_parallel(process_group)


def collect_moe_aux_losses(model: nn.Module):
    """Collect and average auxiliary losses from all MoE layers in the model."""
    total_lb = torch.tensor(0.0, device="cuda")
    total_rz = torch.tensor(0.0, device="cuda")
    n = 0

    for module in model.modules():
        if isinstance(module, MoEFFN):
            losses = module.aux_losses
            if "load_balance" in losses:
                total_lb = total_lb + losses["load_balance"]
            if "router_z" in losses:
                total_rz = total_rz + losses["router_z"]
            n += 1

    if n > 0:
        total_lb = total_lb / n
        total_rz = total_rz / n

    return {"moe_load_balance": total_lb, "moe_router_z": total_rz, "moe_num_layers": n}


def collect_moe_expert_stats(model: nn.Module):
    """Collect expert utilization statistics from all MoE layers.

    Must be called BEFORE reset_moe_balance_stats() since it reads the
    accumulated token counts and routing probabilities.
    """
    all_f = []
    all_P = []

    for module in model.modules():
        if isinstance(module, MoEFFN) and module._accum_num_tokens > 0:
            K = module.top_k
            f = module._accum_tokens_per_expert / (module._accum_num_tokens * K)
            P = module._accum_score_sum / module._accum_num_tokens
            all_f.append(f.detach().float())
            all_P.append(P.detach().float())

    if not all_f:
        return {}

    # Average across MoE layers
    f_avg = torch.stack(all_f).mean(dim=0)  # [E]
    P_avg = torch.stack(all_P).mean(dim=0)  # [E]

    E = f_avg.shape[0]
    f_min = f_avg.min().item()

    return {
        "moe/token_frac_max": f_avg.max().item(),
        "moe/token_frac_min": f_min,
        "moe/token_frac_std": f_avg.std().item(),
        "moe/routing_prob_max": P_avg.max().item(),
        "moe/routing_prob_min": P_avg.min().item(),
        "moe/routing_prob_std": P_avg.std().item(),
        # max/min ratio: 1.0 = perfect balance, higher = worse
        "moe/imbalance_ratio": (f_avg.max().item() / f_min) if f_min > 0 else float("inf"),
        # fraction of experts getting < 50% of ideal share
        "moe/underutilized_experts": int((f_avg < 0.5 / E).sum().item()),
    }


def reset_moe_balance_stats(model: nn.Module):
    """Reset accumulated balance stats on all MoE layers (call after optimizer step)."""
    for module in model.modules():
        if isinstance(module, MoEFFN):
            module.reset_balance_stats()
