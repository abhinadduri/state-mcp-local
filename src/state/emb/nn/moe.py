"""Mixture of Experts (MoE) components for the State Embedding model.

Replaces dense FFN layers with sparsely-activated expert FFNs.
Uses ScatterMoE's Triton scatter2scatter kernels for zero-padding-waste
dropless expert computation when available, with padded bmm fallback.

Key components:
- TopKRouter: Token-to-expert routing with load balancing and z-loss
- MoEFFN: Dropless MoE FFN (ScatterMoE backend or padded bmm fallback)
- MoETransformerEncoderLayer: Drop-in replacement for FlashTransformerEncoderLayer
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Try to import ScatterMoE for zero-padding-waste expert computation
try:
    from scattermoe.mlp import MLP as ScatterMoEMLP
    _HAS_SCATTERMOE = True
except ImportError:
    _HAS_SCATTERMOE = False


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


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss (ST-MoE): penalizes large logits for stability."""
    log_z = torch.logsumexp(router_logits, dim=-1)
    return (log_z ** 2).mean()


class MoEFFN(nn.Module):
    """Dropless Mixture of Experts FFN.

    When ScatterMoE is available, uses Triton scatter2scatter kernels for
    zero-padding-waste expert computation. Otherwise falls back to padded bmm.

    All tokens are processed — no capacity factor, no token dropping.
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
        self.d_hid = d_hid
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k)

        if _HAS_SCATTERMOE:
            # ScatterMoE: Triton-based, zero padding waste, variable-length expert groups
            self.experts = ScatterMoEMLP(
                input_size=d_model,
                hidden_size=d_hid,
                num_experts=num_experts,
                top_k=top_k,
                bias=False,
                activation=nn.GELU(),
            )
            self._backend = "scattermoe"
            log.info(f"MoEFFN using ScatterMoE backend ({num_experts}E, top-{top_k})")
        else:
            # Fallback: padded bmm with stacked expert weights
            self.w1 = nn.Parameter(torch.empty(num_experts, d_model, d_hid))
            self.b1 = nn.Parameter(torch.zeros(num_experts, 1, d_hid))
            self.w2 = nn.Parameter(torch.empty(num_experts, d_hid, d_model))
            self.b2 = nn.Parameter(torch.zeros(num_experts, 1, d_model))
            for i in range(num_experts):
                nn.init.kaiming_uniform_(self.w1[i])
                nn.init.kaiming_uniform_(self.w2[i])
            self._backend = "bmm"
            log.info(f"MoEFFN using padded bmm backend ({num_experts}E, top-{top_k})")

        self.dropout_p = dropout
        self._aux_loss = None
        self._router_z_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        top_k_weights, top_k_indices, router_logits = self.router(x_flat)
        self._aux_loss = load_balancing_loss(router_logits, top_k_indices, self.num_experts)
        self._router_z_loss = router_z_loss(router_logits)

        if self._backend == "scattermoe":
            # Cast to expert weight dtype (FSDP2 may unshard to fp32 under autocast)
            expert_dtype = next(self.experts.parameters()).dtype
            x_cast = x_flat.to(expert_dtype) if x_flat.dtype != expert_dtype else x_flat
            w_cast = top_k_weights.to(expert_dtype) if top_k_weights.dtype != expert_dtype else top_k_weights
            output = self.experts(x_cast, w_cast, top_k_indices)
            if output.dtype != x_flat.dtype:
                output = output.to(x_flat.dtype)
        else:
            output = self._forward_bmm(x_flat, top_k_weights, top_k_indices)

        return output.reshape(B, T, D)

    def _forward_bmm(self, x_flat, top_k_weights, top_k_indices):
        """Padded bmm fallback when ScatterMoE is not available."""
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


class MoETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with MoE FFN.

    Drop-in replacement for FlashTransformerEncoderLayer. Attention is unchanged;
    the dense FFN is replaced with a dropless MoE FFN.
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
