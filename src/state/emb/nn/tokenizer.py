"""Tokenizer abstraction for the State Embedding model.

A Tokenizer converts raw gene expression counts into transformer-ready tokens.
The model is tokenizer-agnostic — it only sees TokenizerOutput.

Two implementations:
- SentenceTokenizer: explicit self-attention over sampled gene tokens (current approach)
- LatentTokenizer: project to latent tokens via linear reduction (faster)
"""

import math
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_transformer import FlashTransformerEncoderLayer, FlashTransformerEncoder


class SkipBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features * 2, bias=True)
        self.dense = nn.Linear(in_features * 2, in_features, bias=True)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x


class TokenizerOutput(NamedTuple):
    """Common output from all tokenizers."""

    cell_embedding: torch.Tensor  # [B, output_dim] CLS token output (normalized)
    task_gene_embs: torch.Tensor  # [B, n_task, d_model] gene embeddings for decoder
    task_counts: torch.Tensor  # [B, n_task] target counts
    dataset_emb: Optional[torch.Tensor]  # [B, output_dim] dataset token output
    dataset_nums: Optional[torch.Tensor]  # [B] dataset IDs for classification loss


class Tokenizer(nn.Module):
    """Base class for tokenizers."""

    def make_collator(self, cfg, is_train: bool, **kwargs) -> Callable:
        """Return a collate_fn for DataLoader."""
        raise NotImplementedError

    def forward(self, batch) -> TokenizerOutput:
        """GPU-side: convert collated batch → tokens → transformer → output."""
        raise NotImplementedError


class SentenceTokenizer(Tokenizer):
    """Wraps the existing gene-sentence tokenization approach.

    Samples 2048 gene tokens per cell, looks up frozen protein embeddings,
    projects to d_model, adds count encoding, runs transformer.
    Produces a CLS embedding that summarizes the cell.
    """

    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        compiled: bool = False,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.output_dim = output_dim

        # Learnable special tokens
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))

        # Encoder: projects token_dim (e.g. 5120) → d_model (e.g. 512)
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Transformer
        layers = [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout) for _ in range(nlayers)]
        self.transformer_encoder = FlashTransformerEncoder(layers)
        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        # Decoder: d_model → output_dim
        self.decoder = nn.Sequential(
            SkipBlock(d_model),
            nn.Linear(d_model, output_dim, bias=True),
        )
        if compiled:
            self.decoder = torch.compile(self.decoder)

        # Count encoding (scFoundation-style soft binning)
        if cfg and cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(
                nn.Linear(1, 512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(512, 10),
            )
        else:
            self.bin_encoder = None
            self.count_encoder = None

        # Gene embedding layer for task genes (reuses encoder projection)
        self.gene_embedding_layer = self.encoder

        # Dataset correction token
        if cfg and getattr(cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
        else:
            self.dataset_token = None

        # Frozen protein embedding table (set externally after init)
        self.pe_embedding = None

    def make_collator(self, cfg, is_train: bool, **kwargs):
        from ..data.loader import VCIDatasetSentenceCollator

        return VCIDatasetSentenceCollator(cfg, is_train=is_train, **kwargs)

    def forward(self, batch) -> TokenizerOutput:
        device = next(self.parameters()).device

        batch_sentences = batch.batch_sentences.to(device)
        task_genes = batch.task_genes.to(device)
        task_counts = batch.task_counts
        sentence_counts = batch.sentence_counts
        if sentence_counts is not None:
            sentence_counts = sentence_counts.to(device)
        dataset_nums = batch.dataset_nums
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(device)

        # Lookup frozen protein embeddings
        with torch.no_grad():
            batch_sentences = self.pe_embedding(batch_sentences)
            task_embs = self.pe_embedding(task_genes)

        # Normalize and insert CLS token
        batch_sentences = F.normalize(batch_sentences, dim=2)
        batch_sentences[:, 0, :] = self.cls_token.expand(batch_sentences.size(0), -1)

        # Optional dataset token
        mask = batch.masks.to(torch.bool).to(device)
        if self.dataset_token is not None:
            dataset_token = self.dataset_token.expand(batch_sentences.size(0), -1).unsqueeze(1)
            batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)
            mask = torch.cat((mask, torch.zeros(mask.size(0), 1, device=device).bool()), dim=1)

        # Encoder projection + count encoding + transformer + decoder
        src = self.encoder(batch_sentences) * math.sqrt(self.d_model)

        if sentence_counts is not None and self.count_encoder is not None:
            counts_input = sentence_counts.unsqueeze(-1)
            bin_weights = F.softmax(self.count_encoder(counts_input), dim=-1)
            bin_embeddings = self.bin_encoder(torch.arange(10, device=device))
            count_emb = torch.matmul(bin_weights, bin_embeddings)
            if self.dataset_token is not None:
                dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=device)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)
            src = src + count_emb

        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)

        # Extract CLS embedding
        embedding = gene_output[:, 0, :]
        embedding = F.normalize(embedding, dim=1)

        # Extract dataset embedding
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        # Task gene embeddings for decoder
        task_gene_embs = self.gene_embedding_layer(task_embs)

        return TokenizerOutput(
            cell_embedding=embedding,
            task_gene_embs=task_gene_embs,
            task_counts=task_counts,
            dataset_emb=dataset_emb,
            dataset_nums=dataset_nums,
        )
