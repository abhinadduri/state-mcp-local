"""Tabular Transformer Encoder: interleaved intra-cell and inter-cell attention.

Intra-cell attention: standard self-attention across latent positions within each cell.
Inter-cell attention: Stack-style — project latent tokens to small token_dim, flatten
per cell, then cells attend to each other using their full flattened representation.

This implements the key idea from Stack (tabular attention) adapted to the SE
latent tokenizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .flash_transformer import FlashTransformerEncoderLayer


class InterCellAttentionLayer(nn.Module):
    """DEPRECATED: Position-independent inter-cell attention.

    Each latent position attends across cells independently.
    Kept for backward compatibility with existing checkpoints.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, n_layers=1):
        super().__init__()
        self.attn_layer = FlashTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, n_layers=n_layers,
        )

    def forward(self, src, n_cells_per_set):
        B_total, seq_len, d_model = src.shape
        n_sets = B_total // n_cells_per_set
        x = src.reshape(n_sets, n_cells_per_set, seq_len, d_model)
        x = x.permute(0, 2, 1, 3).reshape(n_sets * seq_len, n_cells_per_set, d_model)
        x = self.attn_layer(x)
        x = x.reshape(n_sets, seq_len, n_cells_per_set, d_model)
        x = x.permute(0, 2, 1, 3).reshape(B_total, seq_len, d_model)
        return x


class StackStyleInterCellLayer(nn.Module):
    """Stack-style inter-cell attention: flatten latent tokens per cell, then attend.

    Matches Stack's gene-wise attention pattern:
    1. Project each latent token from d_model → token_dim_inter (compression)
    2. Flatten all latent tokens per cell: [n_latent, token_dim_inter] → [n_latent * token_dim_inter]
    3. Cells attend to each other using their full flattened representation
    4. Unflatten and project back up to d_model

    This preserves the holistic cell view during inter-cell attention,
    unlike the position-independent approach.
    """

    def __init__(self, d_model, n_latent, token_dim_inter=8, n_heads=8, dropout=0.1, n_layers=1):
        super().__init__()
        self.n_latent = n_latent
        self.token_dim_inter = token_dim_inter
        self.inter_dim = n_latent * token_dim_inter  # e.g. 256 * 8 = 2048

        # Compress each latent position: d_model → token_dim_inter
        self.down_proj = nn.Linear(d_model, token_dim_inter)
        # Expand back: token_dim_inter → d_model
        self.up_proj = nn.Linear(token_dim_inter, d_model)

        # Inter-cell attention on flattened cell representations
        # Each cell is one token of dim = n_latent * token_dim_inter
        # Use attention-only (no large FFN) — Stack's MLP operates on token_dim, not inter_dim
        self.cell_attn_layer = FlashTransformerEncoderLayer(
            self.inter_dim, n_heads, self.inter_dim,  # no FFN expansion
            dropout=dropout, n_layers=n_layers,
        )

        # Post-attention MLP on token_dim_inter (cheap, matches Stack)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim_inter, token_dim_inter * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim_inter * 4, token_dim_inter),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(token_dim_inter)

        # Residual norm after projecting back to d_model
        self.norm = nn.LayerNorm(d_model)

        # Depth-scaled init on up_proj so early layers start near identity
        import math
        with torch.no_grad():
            self.up_proj.weight *= 1.0 / math.sqrt(2 * n_layers)

    def forward(self, src, n_cells_per_set):
        """
        Args:
            src: [B_total, seq_len, d_model] where seq_len = 1 + n_latent (+ optional ds token)
                 B_total = n_sets * n_cells_per_set
            n_cells_per_set: int
        Returns:
            [B_total, seq_len, d_model]
        """
        B_total, seq_len, d_model = src.shape
        n_sets = B_total // n_cells_per_set

        # Extract latent tokens (skip CLS at position 0, and any trailing special tokens)
        latent_tokens = src[:, 1:1 + self.n_latent, :]  # [B_total, n_latent, d_model]
        residual = latent_tokens

        # Project down: [B_total, n_latent, d_model] → [B_total, n_latent, token_dim_inter]
        projected = self.down_proj(latent_tokens)

        # Flatten per cell: [n_sets, n_cells, n_latent * token_dim_inter]
        projected = projected.reshape(n_sets, n_cells_per_set, self.inter_dim)

        # Inter-cell attention: cells attend to each other using full representation
        # Attention matrix is [n_sets, n_heads, n_cells, n_cells]
        cell_out = self.cell_attn_layer(projected)  # [n_sets, n_cells, inter_dim]

        # Unflatten: [B_total, n_latent, token_dim_inter]
        cell_out = cell_out.reshape(B_total, self.n_latent, self.token_dim_inter)

        # MLP on token_dim_inter (cheap, matches Stack's per-token MLP)
        mlp_out = self.mlp(cell_out)
        cell_out = self.mlp_norm(cell_out + mlp_out)

        # Project back up: [B_total, n_latent, d_model]
        cell_out = self.up_proj(cell_out)

        # Residual + norm on latent tokens
        updated_latents = self.norm(residual + cell_out)

        # Reconstruct full sequence: CLS + updated_latents + any trailing tokens
        out = src.clone()
        out[:, 1:1 + self.n_latent, :] = updated_latents
        return out


class TabularTransformerEncoder(nn.Module):
    """Interleaved intra-cell and inter-cell attention.

    Each "tabular layer" applies:
    1. Intra-cell attention (FlashTransformerEncoderLayer on latent positions)
    2. Inter-cell attention (StackStyleInterCellLayer or InterCellAttentionLayer)

    Final LayerNorm applied at the end (pre-norm architecture).
    """

    def __init__(self, intra_layers, inter_layers, gradient_checkpointing=False, n_cells_per_set=32):
        """
        Args:
            intra_layers: list of FlashTransformerEncoderLayer or MoETransformerEncoderLayer
            inter_layers: list of InterCellAttentionLayer or StackStyleInterCellLayer
            gradient_checkpointing: if True, checkpoint each layer pair
            n_cells_per_set: default cells per set (can be overridden in forward)
        """
        super().__init__()
        assert len(intra_layers) == len(inter_layers)
        self.intra_layers = nn.ModuleList(intra_layers)
        self.inter_layers = nn.ModuleList(inter_layers)
        self.gradient_checkpointing = gradient_checkpointing
        self.n_cells_per_set = n_cells_per_set

        d_model = intra_layers[0].d_model if intra_layers else 512
        self.final_norm = nn.LayerNorm(d_model)

    @property
    def layers(self):
        """All layers for FSDP2 sharding compatibility."""
        result = []
        for intra, inter in zip(self.intra_layers, self.inter_layers):
            result.append(intra)
            result.append(inter)
        return result

    def forward(self, src, n_cells_per_set, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [B_total, seq_len, d_model] where B_total = n_sets * n_cells_per_set
            n_cells_per_set: int
        Returns:
            [B_total, seq_len, d_model]
        """
        mask = src_key_padding_mask if src_key_padding_mask is not None else src_mask
        output = src

        for intra, inter in zip(self.intra_layers, self.inter_layers):
            if self.gradient_checkpointing and self.training:
                output = checkpoint(intra, output, mask, mask, use_reentrant=False)
                output = checkpoint(inter, output, n_cells_per_set, use_reentrant=False)
            else:
                output = intra(output, src_mask=mask, src_key_padding_mask=mask)
                output = inter(output, n_cells_per_set)

        return self.final_norm(output)
