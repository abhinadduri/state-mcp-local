"""Tokenizer abstraction for the State Embedding model.

A Tokenizer converts raw gene expression counts into transformer-ready tokens.
The model is tokenizer-agnostic — it only sees TokenizerOutput.

Two implementations:
- SentenceTokenizer: explicit self-attention over sampled gene tokens (current approach)
- LatentTokenizer: project to latent tokens via linear reduction (faster)
"""

import logging
import math
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_transformer import FlashTransformerEncoderLayer, FlashTransformerEncoder

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# LatentTokenizer: projects all genes to latent tokens via linear reduction
# ---------------------------------------------------------------------------


class LatentBatch(NamedTuple):
    """Output of LatentCollator."""

    global_counts: torch.Tensor  # [B, n_genes] log1p counts aligned to global gene space
    measurement_mask: torch.Tensor  # [B, n_genes] bool — True where gene was measured
    task_genes: torch.Tensor  # [B, P+N] int32 global gene indices for decoder
    task_counts: torch.Tensor  # [B, P+N] log1p target counts
    dataset_nums: Optional[torch.Tensor]  # [B] dataset index


class LatentCollator:
    """CPU-side collation for LatentTokenizer.

    Aligns each cell's counts to the global gene space using ds_emb_map,
    builds a measurement mask, and samples P+N task genes for reconstruction.
    """

    EXPONENTIATED_UMIS_LIMIT = 5_000_000
    RAW_COUNT_HEURISTIC_THRESHOLD = 35

    def __init__(self, cfg, ds_emb_mapping, n_genes: int, is_train: bool = True):
        self.cfg = cfg
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.n_genes = n_genes
        self.is_train = is_train
        self.use_dataset_info = getattr(cfg.model, "dataset_correction", False)

        # Pre-compute per-dataset mappings as tensors
        self._ds_mapping = {}
        for ds_name, raw_idxs in ds_emb_mapping.items():
            mapping = torch.tensor(raw_idxs, dtype=torch.long) if not isinstance(raw_idxs, torch.Tensor) else raw_idxs.long()
            self._ds_mapping[ds_name] = mapping

    @staticmethod
    def _is_raw_counts(counts: torch.Tensor) -> bool:
        max_val = torch.max(counts).item()
        if max_val > 35:
            return True
        total_umis = int(torch.expm1(counts).sum().item())
        return total_umis > 5_000_000

    def _process_cell(self, counts_raw: torch.Tensor, dataset: str):
        """Normalize counts and align to global gene space. Returns (global_counts, measurement_mask)."""
        # Ensure log1p
        if counts_raw.numel() > 0 and self._is_raw_counts(counts_raw):
            counts_raw = torch.log1p(F.relu(counts_raw))
        elif torch.any(counts_raw < 0):
            counts_raw = F.relu(counts_raw)

        counts_raw = counts_raw.squeeze(0)  # [G_local]
        mapping = self._ds_mapping[dataset]  # [G_local] → global indices, -1 for unmapped

        global_counts = torch.zeros(self.n_genes)
        measurement_mask = torch.zeros(self.n_genes, dtype=torch.bool)

        valid = mapping >= 0
        global_idx = mapping[valid]
        global_counts[global_idx] = counts_raw[valid]
        measurement_mask[global_idx] = True

        return global_counts, measurement_mask

    def _sample_task_genes(self, global_counts: torch.Tensor, measurement_mask: torch.Tensor):
        """Sample P expressed + N unexpressed genes among measured genes for reconstruction."""
        measured_idx = torch.where(measurement_mask)[0]
        if len(measured_idx) == 0:
            # Degenerate: no measured genes — return zeros
            task_genes = torch.zeros(self.P + self.N, dtype=torch.int32)
            task_counts = torch.zeros(self.P + self.N)
            return task_genes, task_counts

        measured_counts = global_counts[measured_idx]
        expressed = measured_idx[measured_counts > 0]
        unexpressed = measured_idx[measured_counts == 0]

        # Sample P expressed genes
        if len(expressed) >= self.P:
            p_idx = expressed[torch.randperm(len(expressed))[: self.P]]
        elif len(expressed) > 0:
            p_idx = expressed[torch.randint(len(expressed), (self.P,))]
        else:
            # No expressed genes — sample from measured
            p_idx = measured_idx[torch.randint(len(measured_idx), (self.P,))]

        # Sample N unexpressed genes
        if len(unexpressed) >= self.N:
            n_idx = unexpressed[torch.randperm(len(unexpressed))[: self.N]]
        elif len(unexpressed) > 0:
            n_idx = unexpressed[torch.randint(len(unexpressed), (self.N,))]
        else:
            # All measured genes are expressed — sample from measured
            n_idx = measured_idx[torch.randint(len(measured_idx), (self.N,))]

        task_genes = torch.cat([p_idx, n_idx]).to(torch.int32)
        task_counts = global_counts[task_genes.long()]
        return task_genes, task_counts

    def __call__(self, batch):
        batch_size = len(batch)
        global_counts_list = []
        mask_list = []
        task_genes_list = []
        task_counts_list = []
        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)

        for i, (counts, idx, dataset, dataset_num) in enumerate(batch):
            gc, mm = self._process_cell(counts, dataset)
            tg, tc = self._sample_task_genes(gc, mm)
            global_counts_list.append(gc)
            mask_list.append(mm)
            task_genes_list.append(tg)
            task_counts_list.append(tc)
            dataset_nums[i] = dataset_num

        return LatentBatch(
            global_counts=torch.stack(global_counts_list),
            measurement_mask=torch.stack(mask_list),
            task_genes=torch.stack(task_genes_list),
            task_counts=torch.stack(task_counts_list),
            dataset_nums=dataset_nums if self.use_dataset_info else None,
        )


class LatentTokenizer(Tokenizer):
    """Projects all genes to latent tokens via linear reduction.

    For each of the n_genes positions:
    - Measured gene: encoder(pe_embedding[i]) + count_emb(count_i)
    - Unmeasured gene: missing_emb[i] (learned per-position)

    Then projects (B, n_genes, d_model) → (B, n_latent, d_model) via Linear(n_genes, n_latent).
    CLS token + transformer + decoder produces the cell embedding.

    ~250x cheaper attention than SentenceTokenizer (n_latent vs pad_length tokens).
    """

    def __init__(
        self,
        n_genes: int,
        n_latent: int,
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
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.d_model = d_model
        self.output_dim = output_dim

        # Encoder: projects protein embedding dim → d_model (shared with task gene embedding)
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.gene_embedding_layer = self.encoder  # alias for decoder task genes

        # Learned embedding for unmeasured gene positions
        self.missing_emb = nn.Parameter(torch.randn(n_genes, d_model) * 0.02)

        # Gene-to-latent projection: Linear(n_genes, n_latent) applied per d_model channel
        self.gene_reduction = nn.Linear(n_genes, n_latent)

        # Count encoding (same as SentenceTokenizer)
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

        # Learnable CLS token (in d_model space, not token_dim)
        self.cls_token = nn.Parameter(torch.randn(1, d_model))

        # Dataset correction token
        if cfg and getattr(cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, d_model))
        else:
            self.dataset_token = None

        # Transformer (operates on n_latent + 1 tokens, much cheaper than SentenceTokenizer)
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

        # Frozen protein embedding table (set externally)
        self.pe_embedding = None

        # Cache for projected gene identity embeddings (computed once)
        self._gene_identity_cache = None

    def make_collator(self, cfg, is_train: bool, **kwargs):
        from .. import utils

        ds_emb_mapping = torch.load(
            utils.get_embedding_cfg(cfg).ds_emb_mapping.format(utils.get_embedding_cfg(cfg).size),
            weights_only=False,
        )
        return LatentCollator(
            cfg=cfg,
            ds_emb_mapping=ds_emb_mapping,
            n_genes=self.n_genes,
            is_train=is_train,
        )

    def _get_gene_identity_embs(self, device):
        """Compute encoder(pe_embedding[0..n_genes]) once and cache."""
        if self._gene_identity_cache is not None and self._gene_identity_cache.device == device:
            return self._gene_identity_cache

        with torch.no_grad():
            all_indices = torch.arange(self.n_genes, device=device)
            pe_out = self.pe_embedding(all_indices)  # [n_genes, token_dim]

        # Encoder projection (has grad — part of the model)
        gene_identity = self.encoder(pe_out)  # [n_genes, d_model]
        self._gene_identity_cache = gene_identity.detach()
        return self._gene_identity_cache

    def forward(self, batch: LatentBatch) -> TokenizerOutput:
        device = next(self.parameters()).device

        global_counts = batch.global_counts.to(device)  # [B, n_genes]
        measurement_mask = batch.measurement_mask.to(device)  # [B, n_genes] bool
        task_genes = batch.task_genes.to(device)
        task_counts = batch.task_counts
        dataset_nums = batch.dataset_nums
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(device)

        B = global_counts.shape[0]

        # --- Build per-gene tokens ---
        # Gene identity embeddings from frozen protein embeddings (cached)
        gene_identity = self._get_gene_identity_embs(device)  # [n_genes, d_model]

        # Count encoding for all gene positions
        if self.count_encoder is not None:
            counts_input = global_counts.unsqueeze(-1)  # [B, n_genes, 1]
            bin_weights = F.softmax(self.count_encoder(counts_input), dim=-1)  # [B, n_genes, 10]
            bin_embeddings = self.bin_encoder(torch.arange(10, device=device))  # [10, d_model]
            count_emb = torch.matmul(bin_weights, bin_embeddings)  # [B, n_genes, d_model]
        else:
            count_emb = torch.zeros(B, self.n_genes, self.d_model, device=device)

        # Measured genes: gene_identity + count_emb
        # Unmeasured genes: missing_emb
        measured_tokens = gene_identity.unsqueeze(0) + count_emb  # [B, n_genes, d_model]
        missing_tokens = self.missing_emb.unsqueeze(0).expand(B, -1, -1)  # [B, n_genes, d_model]

        # Select measured vs missing per position
        mask_3d = measurement_mask.unsqueeze(-1)  # [B, n_genes, 1]
        gene_tokens = torch.where(mask_3d, measured_tokens, missing_tokens)  # [B, n_genes, d_model]

        # --- Project to latent tokens ---
        # gene_reduction: Linear(n_genes, n_latent) applied per d_model channel
        latent_tokens = self.gene_reduction(
            gene_tokens.transpose(1, 2)  # [B, d_model, n_genes]
        ).transpose(1, 2)  # [B, n_latent, d_model]

        # --- Prepend CLS token (+ optional dataset token) ---
        cls = self.cls_token.expand(B, -1).unsqueeze(1)  # [B, 1, d_model]
        if self.dataset_token is not None:
            ds_tok = self.dataset_token.expand(B, -1).unsqueeze(1)  # [B, 1, d_model]
            src = torch.cat([cls, latent_tokens, ds_tok], dim=1)  # [B, 1+n_latent+1, d_model]
        else:
            src = torch.cat([cls, latent_tokens], dim=1)  # [B, 1+n_latent, d_model]

        # Scale (matching SentenceTokenizer convention)
        src = src * math.sqrt(self.d_model)

        # --- Transformer + decoder ---
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)

        # CLS embedding
        embedding = gene_output[:, 0, :]
        embedding = F.normalize(embedding, dim=1)

        # Dataset embedding
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        # --- Task gene embeddings for decoder ---
        with torch.no_grad():
            task_pe = self.pe_embedding(task_genes.long())  # [B, P+N, token_dim]
        task_gene_embs = self.gene_embedding_layer(task_pe)  # [B, P+N, d_model]

        return TokenizerOutput(
            cell_embedding=embedding,
            task_gene_embs=task_gene_embs,
            task_counts=task_counts,
            dataset_emb=dataset_emb,
            dataset_nums=dataset_nums,
        )
