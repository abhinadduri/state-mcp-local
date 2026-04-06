"""Decoder heads for the State Embedding model.

CrossAttentionNBDecoder: Stack-style gene query cross-attention with NB reconstruction loss.
Gene queries cross-attend to the full latent bank (CLS + 256 latents),
producing per-gene representations that are mapped to NB parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import CrossAttentionBlock


class CrossAttentionNBDecoder(nn.Module):
    """Stack-style decoder: gene queries cross-attend to latent bank, predict NB params.

    Architecture:
      1. Gene queries [B, k_max, d_model] from embedding table (DETACHED)
      2. Cross-attention to latent_bank [B, 1+n_latent, d_model]
      3. Output head: [B, k_max, d_model] -> [B, k_max, 2] (px_scale_logit, raw_dispersion)

    NB mean is computed externally: softmax(px_scale_logits) * library_size
    (matching Stack's _compute_nb_parameters at base.py:118-121).
    """

    def __init__(self, d_model: int, nhead: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Cross-attention layers: gene queries attend to latent bank
        # Reuses CrossAttentionBlock from tokenizer.py (Flash SDPA)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout)
            for _ in range(n_layers)
        ])

        # Output head: predicts 2 values per gene (px_scale_logit, raw_dispersion)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, gene_queries, latent_bank):
        """
        Args:
            gene_queries: [B, k_max, d_model] — DETACHED gene embeddings
            latent_bank:  [B, 1+n_latent, d_model] — CLS + 256 latent tokens from encoder
        Returns:
            px_scale_logits: [B, k_max] — logits for gene frequency (softmax externally)
            nb_dispersion:   [B, k_max] — NB dispersion parameter (> 0)
        """
        x = gene_queries
        # All latent tokens are valid (no padding in latent bank)
        # CrossAttentionBlock expects kv_mask as [B, k_max] bool
        kv_mask = torch.ones(
            latent_bank.shape[0], latent_bank.shape[1],
            dtype=torch.bool, device=latent_bank.device,
        )  # [B, 1+n_latent]

        for layer in self.cross_attn_layers:
            x = layer(x, latent_bank, kv_mask)  # [B, k_max, d_model]

        output = self.output_head(x)  # [B, k_max, 2]
        px_scale_logits = output[..., 0]  # [B, k_max]
        nb_dispersion = F.softplus(output[..., 1])  # [B, k_max], > 0
        return px_scale_logits, nb_dispersion
