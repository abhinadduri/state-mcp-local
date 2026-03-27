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

log = logging.getLogger(__name__)


class TopKRouter(nn.Module):
    """Token-to-expert router with auxiliary losses for balanced expert utilization."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        router_logits = self.gate(x)
        scores = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
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
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.w1[i])
            nn.init.kaiming_uniform_(self.w2[i])

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

        # Global-batch load balancing: accumulate stats across micro-batches
        self._accum_tokens_per_expert = None  # [E]
        self._accum_score_sum = None  # [E]
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
        import torch.distributed as dist

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

        # Accumulate balance stats across micro-batches for global-batch loss
        tpe, ss, nt = _compute_balance_stats(router_logits, top_k_indices, self.num_experts)
        if self._accum_tokens_per_expert is None:
            self._accum_tokens_per_expert = tpe
            self._accum_score_sum = ss
            self._accum_num_tokens = nt
        else:
            self._accum_tokens_per_expert = self._accum_tokens_per_expert + tpe
            self._accum_score_sum = self._accum_score_sum + ss
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

    def _forward_ep(self, x_flat, top_k_weights, top_k_indices):
        """Expert parallel forward: all-to-all dispatch → local experts (bmm) → all-to-all combine.

        Supports multiple experts per GPU (e.g., 32 experts on 8 GPUs = 4 per GPU).
        Tokens for this rank's experts are grouped and processed via bmm.
        """
        import torch.distributed as dist

        N, D = x_flat.shape
        E = self.num_experts
        K = self.top_k
        ep_group = self._ep_group
        ep_size = self._ep_size
        epr = self._experts_per_rank  # experts per rank
        comm_dtype = torch.bfloat16

        # Flatten top-k selections
        flat_experts = top_k_indices.reshape(-1)  # [N*K]
        flat_weights = top_k_weights.reshape(-1)
        flat_token_idx = torch.arange(N, device=x_flat.device).unsqueeze(1).expand(-1, K).reshape(-1)

        # Sort by expert
        sort_idx = flat_experts.argsort()
        sorted_token_idx = flat_token_idx[sort_idx]
        sorted_experts = flat_experts[sort_idx]
        sorted_weights = flat_weights[sort_idx]

        # Count tokens per expert
        local_counts = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        local_counts.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts, dtype=torch.long))

        # For all-to-all: group counts by rank (sum experts per rank)
        # rank r owns experts [r*epr, (r+1)*epr)
        rank_send_counts = local_counts.reshape(ep_size, epr).sum(dim=1)  # [ep_size]

        # Exchange rank-level counts
        rank_recv_counts = torch.empty_like(rank_send_counts)
        dist.all_to_all_single(rank_recv_counts, rank_send_counts, group=ep_group)

        input_splits = rank_send_counts.tolist()
        output_splits = rank_recv_counts.tolist()

        # Gather sorted tokens and pre-weight
        sorted_tokens = x_flat[sorted_token_idx]
        sorted_weighted = (sorted_tokens * sorted_weights.unsqueeze(-1)).to(comm_dtype)

        # All-to-all dispatch
        recv_total = sum(output_splits)
        recv_tokens = torch.empty(recv_total, D, device=x_flat.device, dtype=comm_dtype)
        dist.all_to_all_single(recv_tokens, sorted_weighted,
                               output_split_sizes=output_splits,
                               input_split_sizes=input_splits,
                               group=ep_group)

        # We also need to know which local expert each received token goes to.
        # Exchange the per-expert counts (not just per-rank) for local routing.
        local_expert_counts = local_counts.clone()
        recv_expert_counts = torch.empty_like(local_expert_counts)
        dist.all_to_all_single(recv_expert_counts, local_expert_counts, group=ep_group)
        # recv_expert_counts[e] = how many tokens were sent to expert e from all ranks
        # Our local experts are indices [rank*epr, (rank+1)*epr)
        my_start = self._ep_rank * epr
        my_expert_counts = recv_expert_counts[my_start:my_start + epr]  # [epr]

        # Local expert computation via bmm on this rank's experts
        w1 = self.w1.to(comm_dtype)  # [epr, D, H]
        b1 = self.b1.to(comm_dtype)  # [epr, 1, H]
        w2 = self.w2.to(comm_dtype)  # [epr, H, D]
        b2 = self.b2.to(comm_dtype)  # [epr, 1, D]

        if epr == 1:
            # Single expert per GPU: simple matmul
            h = F.gelu(recv_tokens @ w1.squeeze(0) + b1.squeeze(0).squeeze(0))
            if self.dropout_p > 0 and self.training:
                h = F.dropout(h, p=self.dropout_p, training=True)
            expert_out = h @ w2.squeeze(0) + b2.squeeze(0).squeeze(0)
        else:
            # Multiple experts per GPU: pad and bmm
            max_tok = my_expert_counts.max() if my_expert_counts.numel() > 0 else torch.tensor(0)
            max_tok_val = max_tok.item() if max_tok.numel() == 1 else int(max_tok)
            if max_tok_val == 0:
                expert_out = recv_tokens.new_zeros(recv_total, D)
            else:
                # Split recv_tokens by local expert
                padded = recv_tokens.new_zeros(epr, max_tok_val, D)
                offsets = torch.zeros(epr + 1, dtype=torch.long, device=x_flat.device)
                torch.cumsum(my_expert_counts, dim=0, out=offsets[1:])
                for e in range(epr):
                    cnt = my_expert_counts[e].item()
                    if cnt > 0:
                        padded[e, :cnt] = recv_tokens[offsets[e]:offsets[e+1]]

                h = torch.bmm(padded, w1) + b1
                h = F.gelu(h)
                if self.dropout_p > 0 and self.training:
                    h = F.dropout(h, p=self.dropout_p, training=True)
                out_padded = torch.bmm(h, w2) + b2  # [epr, max_tok, D]

                # Unpad
                expert_out = recv_tokens.new_zeros(recv_total, D)
                for e in range(epr):
                    cnt = my_expert_counts[e].item()
                    if cnt > 0:
                        expert_out[offsets[e]:offsets[e+1]] = out_padded[e, :cnt]

        # All-to-all combine: send results back
        send_back = torch.empty(N * K, D, device=x_flat.device, dtype=comm_dtype)
        dist.all_to_all_single(send_back, expert_out,
                               output_split_sizes=input_splits,
                               input_split_sizes=output_splits,
                               group=ep_group)

        # Unsort and accumulate over top-K
        output = torch.zeros(N, D, device=x_flat.device, dtype=x_flat.dtype)
        unsort_idx = sort_idx.argsort()
        unsorting = send_back[unsort_idx].to(x_flat.dtype)
        output = unsorting.reshape(N, K, D).sum(dim=1)

        return output

    def _forward_bmm(self, x_flat, top_k_weights, top_k_indices):
        """Padded bmm forward (no EP)."""
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
        max_tokens = expert_counts.max()

        offsets = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        torch.cumsum(expert_counts[:-1], dim=0, out=offsets[1:])
        global_pos = torch.arange(M, device=x_flat.device)
        positions = global_pos - offsets[sorted_experts]

        padded_tokens = x_flat.new_zeros(E, max_tokens, D)
        padded_weights = x_flat.new_zeros(E, max_tokens)
        padded_out_idx = torch.zeros(E, max_tokens, dtype=torch.long, device=x_flat.device)

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
        self._accum_tokens_per_expert = None
        self._accum_score_sum = None
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
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        residual = src
        qkv = self.qkv_proj(src)
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
        src = self.norm1(residual + self.dropout_layer(attn_output))

        residual2 = src
        ff_output = self.moe_ffn(src)
        src = self.norm2(residual2 + self.dropout_layer(ff_output))
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


def reset_moe_balance_stats(model: nn.Module):
    """Reset accumulated balance stats on all MoE layers (call after optimizer step)."""
    for module in model.modules():
        if isinstance(module, MoEFFN):
            module.reset_balance_stats()
