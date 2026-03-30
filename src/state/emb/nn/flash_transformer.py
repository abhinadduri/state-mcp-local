# File: vci/flash_transformer.py
"""
This module implements a Transformer encoder layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, n_layers=1):
        """
        Initializes the encoder layer with pre-norm architecture.
        Args:
            d_model (int): model dimension.
            nhead (int): number of attention heads.
            dim_feedforward (int): dimension of the feed-forward network.
            dropout (float): dropout probability.
            n_layers (int): total number of layers (for depth-scaled init).
        """
        super().__init__()
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        # Linear projections for Q, K, V in one matrix
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Depth-scaled init on residual output projections (GPT-2/GPT-3 scheme)
        import math
        residual_scale = 1.0 / math.sqrt(2 * n_layers)
        with torch.no_grad():
            self.out_proj.weight *= residual_scale
            self.linear2.weight *= residual_scale

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-norm: normalize before sublayer, add residual after
        normed = self.norm1(src)
        qkv = self.qkv_proj(normed)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        head_dim = self.d_model // self.nhead
        q = q.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)
        k = k.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)
        v = v.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(src.size(0), src.size(1), self.d_model)
        attn_output = self.out_proj(attn_output)
        src = src + self.dropout_layer(attn_output)

        # Pre-norm FFN
        normed2 = self.norm2(src)
        ff_output = self.linear2(self.dropout_layer(F.gelu(self.linear1(normed2))))
        src = src + self.dropout_layer(ff_output)
        return src


class FlashTransformerEncoder(nn.Module):
    def __init__(self, layers, gradient_checkpointing=False):
        """
        A simple encoder that applies a stack of FlashTransformerEncoderLayer instances.
        Uses pre-norm architecture, so a final LayerNorm is applied after all layers.
        Args:
            layers (list[nn.Module]): list of FlashTransformerEncoderLayer instances.
            gradient_checkpointing (bool): if True, checkpoint each layer to save memory.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.gradient_checkpointing = gradient_checkpointing
        # Final norm needed for pre-norm architecture (last layer output is un-normalized)
        d_model = layers[0].d_model if layers else 512
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Applies each encoder layer in sequence, then final normalization.
        Args:
            src: Tensor of shape (B, T, d_model)
            src_mask: (optional) attention mask.
            src_key_padding_mask: (optional) padding mask.
        Returns:
            Tensor of shape (B, T, d_model)
        """
        mask = src_key_padding_mask if src_key_padding_mask is not None else src_mask
        output = src
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                output = checkpoint(layer, output, mask, mask, use_reentrant=False)
            else:
                output = layer(output, src_mask=mask, src_key_padding_mask=mask)
        return self.final_norm(output)
