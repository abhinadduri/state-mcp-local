"""Mixture of Experts (MoE) components for the State Embedding model.

Replaces dense FFN layers with sparsely-activated expert FFNs.
Uses per-expert token gathering for memory-efficient batched computation.

Key components:
- TopKRouter: Token-to-expert routing with load balancing and z-loss
- MoEFFN: Sparse expert FFN with top-k routing
- MoETransformerEncoderLayer: Drop-in replacement for FlashTransformerEncoderLayer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Token-to-expert router with auxiliary losses for balanced expert utilization."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """Route tokens to experts.

        Args:
            x: [num_tokens, d_model]

        Returns:
            top_k_weights: [num_tokens, top_k] normalized routing weights
            top_k_indices: [num_tokens, top_k] expert indices
            router_logits: [num_tokens, num_experts] raw logits for aux losses
        """
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


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss (ST-MoE): penalizes large logits for stability."""
    log_z = torch.logsumexp(router_logits, dim=-1)
    return (log_z ** 2).mean()


class ExpertFFN(nn.Module):
    """Single expert FFN: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_hid: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hid)
        self.w2 = nn.Linear(d_hid, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class MoEFFN(nn.Module):
    """Mixture of Experts FFN using per-expert token gathering.

    Memory-efficient: for each expert, gathers only the tokens assigned to it,
    runs the expert FFN, and scatters results back. No per-token weight expansion.
    With 8 experts and top-2, each expert processes ~25% of tokens on average.
    """

    def __init__(
        self,
        d_model: int,
        d_hid: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_hid, dropout) for _ in range(num_experts)
        ])

        self._aux_loss = None
        self._router_z_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-expert gather-compute-scatter.

        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # [N, D]

        top_k_weights, top_k_indices, router_logits = self.router(x_flat)
        self._aux_loss = load_balancing_loss(router_logits, top_k_indices, self.num_experts)
        self._router_z_loss = router_z_loss(router_logits)

        # Flatten top-k selections: each token appears top_k times
        # flat_indices[i] = token index, flat_experts[i] = expert id, flat_weights[i] = weight
        N = x_flat.shape[0]
        flat_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        flat_experts = top_k_indices.reshape(-1)       # [N*top_k]
        flat_weights = top_k_weights.reshape(-1)       # [N*top_k]

        # Sort by expert for contiguous memory access
        sort_idx = flat_experts.argsort()
        sorted_token_idx = flat_token_idx[sort_idx]
        sorted_experts = flat_experts[sort_idx]
        sorted_weights = flat_weights[sort_idx]

        # Find expert boundaries
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        expert_counts.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts, dtype=torch.long))
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x.device)
        torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

        # Gather tokens, run each expert, scatter back
        output = torch.zeros_like(x_flat)  # [N, D]

        for e in range(self.num_experts):
            start = expert_offsets[e].item()
            end = expert_offsets[e + 1].item()
            if start == end:
                continue

            token_indices = sorted_token_idx[start:end]    # indices into x_flat
            weights_e = sorted_weights[start:end]           # routing weights

            tokens_e = x_flat[token_indices]                # [n_e, D]
            out_e = self.experts[e](tokens_e)               # [n_e, D]

            # Weighted scatter-add back
            output.index_add_(0, token_indices, weights_e.unsqueeze(-1) * out_e)

        return output.reshape(B, T, D)

    @property
    def aux_losses(self):
        losses = {}
        if self._aux_loss is not None:
            losses["load_balance"] = self._aux_loss
        if self._router_z_loss is not None:
            losses["router_z"] = self._router_z_loss
        return losses


class MoETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with MoE FFN.

    Drop-in replacement for FlashTransformerEncoderLayer. Attention is unchanged;
    the dense FFN is replaced with a sparse MoE FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        # Self-attention (identical to FlashTransformerEncoderLayer)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # MoE FFN
        self.moe_ffn = MoEFFN(
            d_model=d_model,
            d_hid=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-Attention Block
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

        # MoE FFN Block
        residual2 = src
        ff_output = self.moe_ffn(src)
        src = self.norm2(residual2 + self.dropout_layer(ff_output))
        return src

    @property
    def aux_losses(self):
        return self.moe_ffn.aux_losses


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
