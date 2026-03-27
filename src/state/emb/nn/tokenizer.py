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
from .moe import MoETransformerEncoderLayer

log = logging.getLogger(__name__)


def _build_transformer_layers(d_model, nhead, d_hid, nlayers, dropout, cfg=None):
    """Build transformer layers, optionally replacing every moe_freq-th with MoE."""
    moe_cfg = cfg.model.get("moe", None) if cfg else None
    use_moe = moe_cfg is not None and getattr(moe_cfg, "enable", False)

    if not use_moe:
        return [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout) for _ in range(nlayers)]

    num_experts = getattr(moe_cfg, "num_experts", 8)
    top_k = getattr(moe_cfg, "top_k", 2)
    moe_freq = getattr(moe_cfg, "moe_freq", 2)
    num_shared = getattr(moe_cfg, "num_shared_experts", 0)

    layers = []
    for i in range(nlayers):
        if (i + 1) % moe_freq == 0:
            layers.append(MoETransformerEncoderLayer(
                d_model, nhead, d_hid, num_experts=num_experts, top_k=top_k,
                dropout=dropout, num_shared_experts=num_shared,
            ))
        else:
            layers.append(FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout))

    n_moe = sum(1 for l in layers if isinstance(l, MoETransformerEncoderLayer))
    shared_str = f", {num_shared} shared" if num_shared > 0 else ""
    log.info(f"Built {nlayers} layers: {n_moe} MoE ({num_experts}E, top-{top_k}{shared_str}), {nlayers - n_moe} dense")
    return layers


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

        # Transformer (supports mixed dense + MoE layers via config)
        grad_ckpt = cfg and getattr(cfg.model, "gradient_checkpointing", False)
        layers = _build_transformer_layers(d_model, nhead, d_hid, nlayers, dropout, cfg)
        self.transformer_encoder = FlashTransformerEncoder(layers, gradient_checkpointing=grad_ckpt)
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
        task_counts = batch.task_counts.to(device)
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
# LatentTokenizer: sparse cross-attention from latent queries to measured genes
# ---------------------------------------------------------------------------


class LatentBatch(NamedTuple):
    """Output of LatentCollator. Sparse representation — only measured genes."""

    gene_indices: torch.Tensor  # [B, k_max] int64 global gene IDs, padded with 0
    gene_counts: torch.Tensor  # [B, k_max] float log1p counts, 0 for padding
    gene_mask: torch.Tensor  # [B, k_max] bool — True for real genes, False for padding
    task_genes: torch.Tensor  # [B, P+N] int32 global gene indices for decoder
    task_counts: torch.Tensor  # [B, P+N] log1p target counts
    dataset_nums: Optional[torch.Tensor]  # [B] dataset index


class LatentCollator:
    """CPU-side collation for LatentTokenizer.

    Produces a sparse representation: only measured genes per cell (with their
    global indices and counts), padded to the max measured count in the batch.
    Samples P+N task genes for reconstruction.
    """

    def __init__(self, cfg, ds_emb_mapping, n_genes: int, is_train: bool = True, k_top: Optional[int] = None):
        self.cfg = cfg
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.n_genes = n_genes
        self.is_train = is_train
        self.k_top = k_top  # if set, keep only top-k expressed genes for cross-attention
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
        """Normalize counts, return sparse (gene_indices, gene_counts) in global space."""
        if counts_raw.numel() > 0 and self._is_raw_counts(counts_raw):
            counts_raw = torch.log1p(F.relu(counts_raw))
        elif torch.any(counts_raw < 0):
            counts_raw = F.relu(counts_raw)

        counts_raw = counts_raw.squeeze(0)  # [G_local]
        mapping = self._ds_mapping[dataset]  # [G_local] → global indices, -1 for unmapped

        valid = mapping >= 0
        gene_indices = mapping[valid]  # [k] global gene IDs
        gene_counts = counts_raw[valid]  # [k] log1p counts
        return gene_indices, gene_counts

    def _sample_task_genes(self, gene_indices: torch.Tensor, gene_counts: torch.Tensor):
        """Sample P expressed + N unexpressed genes from measured genes."""
        if len(gene_indices) == 0:
            return torch.zeros(self.P + self.N, dtype=torch.int32), torch.zeros(self.P + self.N)

        expressed_mask = gene_counts > 0
        expressed = gene_indices[expressed_mask]
        unexpressed = gene_indices[~expressed_mask]

        # Sample P expressed genes
        if len(expressed) >= self.P:
            p_idx = expressed[torch.randperm(len(expressed))[: self.P]]
        elif len(expressed) > 0:
            p_idx = expressed[torch.randint(len(expressed), (self.P,))]
        else:
            p_idx = gene_indices[torch.randint(len(gene_indices), (self.P,))]

        # Sample N unexpressed genes
        if len(unexpressed) >= self.N:
            n_idx = unexpressed[torch.randperm(len(unexpressed))[: self.N]]
        elif len(unexpressed) > 0:
            n_idx = unexpressed[torch.randint(len(unexpressed), (self.N,))]
        else:
            n_idx = gene_indices[torch.randint(len(gene_indices), (self.N,))]

        task_genes = torch.cat([p_idx, n_idx]).to(torch.int32)
        # Look up counts for task genes from the sparse representation
        # Build a quick lookup: global_idx → count
        idx_to_count = torch.zeros(self.n_genes)
        idx_to_count[gene_indices] = gene_counts
        task_counts = idx_to_count[task_genes.long()]
        return task_genes, task_counts

    def _truncate_top_k(self, gene_indices: torch.Tensor, gene_counts: torch.Tensor):
        """Keep only the top k_top genes by expression count."""
        if self.k_top is None or len(gene_indices) <= self.k_top:
            return gene_indices, gene_counts
        _, top_idx = torch.topk(gene_counts, self.k_top)
        return gene_indices[top_idx], gene_counts[top_idx]

    def __call__(self, batch):
        num_aug = getattr(self.cfg.model, "num_downsample", 1)
        if num_aug > 1 and self.is_train:
            batch = [item for item in batch for _ in range(num_aug)]

        batch_size = len(batch)
        cells = []  # list of (gene_indices, gene_counts) per cell
        task_genes_list = []
        task_counts_list = []
        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)

        for i, (counts, idx, dataset, dataset_num) in enumerate(batch):
            gi, gc = self._process_cell(counts, dataset)
            # Sample task genes from ALL measured genes (before truncation)
            tg, tc = self._sample_task_genes(gi, gc)
            # Truncate to top-k for cross-attention input
            gi, gc = self._truncate_top_k(gi, gc)
            cells.append((gi, gc))
            task_genes_list.append(tg)
            task_counts_list.append(tc)
            dataset_nums[i] = dataset_num

        # Pad to bucketed size (nearest multiple of 256) so torch.compile sees
        # at most ~8 distinct shapes, avoiding constant recompilation.
        k_max = max(len(c[0]) for c in cells)
        k_max = ((k_max + 255) // 256) * 256
        gene_indices = torch.zeros(batch_size, k_max, dtype=torch.long)
        gene_counts = torch.zeros(batch_size, k_max)
        gene_mask = torch.zeros(batch_size, k_max, dtype=torch.bool)

        for i, (gi, gc) in enumerate(cells):
            k = len(gi)
            gene_indices[i, :k] = gi
            gene_counts[i, :k] = gc
            gene_mask[i, :k] = True

        return LatentBatch(
            gene_indices=gene_indices,
            gene_counts=gene_counts,
            gene_mask=gene_mask,
            task_genes=torch.stack(task_genes_list),
            task_counts=torch.stack(task_counts_list),
            dataset_nums=dataset_nums if self.use_dataset_info else None,
        )


class CrossAttentionBlock(nn.Module):
    """Single cross-attention round: queries attend to key-value tokens."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, queries, kv_tokens, kv_mask):
        B = queries.shape[0]
        n_q = queries.shape[1]
        k_max = kv_tokens.shape[1]

        q = self.q_proj(queries)
        kv = self.kv_proj(kv_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, n_q, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, k_max, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, k_max, self.nhead, self.head_dim).transpose(1, 2)

        # Always apply mask — avoids cudaStreamSynchronize from kv_mask.all().item()
        # which also acts as a torch.compile graph break.
        padding_mask = kv_mask.unsqueeze(-1)
        k = k * padding_mask.unsqueeze(1)
        v = v * padding_mask.unsqueeze(1)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, n_q, self.d_model)
        attn_out = self.out_proj(attn_out)
        return self.norm(queries + attn_out)


class LatentTokenizer(Tokenizer):
    """Sparse cross-attention from latent queries to measured gene tokens.

    Only measured genes are materialized as tokens — no (B, n_genes, d_model)
    intermediate. Scales to 200K+ genes across species.

    For each measured gene:  gene_token = encoder(pe_embedding[i]) + count_emb(count_i)
    Unmeasured genes are simply absent (not padded with a missing embedding).

    Learned latent queries cross-attend to the sparse gene tokens, producing
    (B, n_latent, d_model) latent tokens. Self-attention + decoder produce
    the CLS cell embedding.
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
        self._gradient_checkpointing = cfg and getattr(cfg.model, "gradient_checkpointing", False)

        # Learned gene embeddings vs frozen ESM2 + projection
        self.use_learned_embeddings = cfg and getattr(cfg.model, "use_learned_embeddings", False)

        if self.use_learned_embeddings:
            # Trainable gene embeddings: directly learn d_model-dim vectors per gene
            self.learned_gene_emb = nn.Embedding(n_genes, d_model)
            nn.init.normal_(self.learned_gene_emb.weight, std=0.02)
            self.encoder = nn.Identity()
            self.gene_embedding_layer = self.learned_gene_emb
        else:
            # Encoder: projects protein embedding dim → d_model (shared with task gene embedding)
            self.encoder = nn.Sequential(
                nn.Linear(token_dim, d_model, bias=True),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            self.gene_embedding_layer = self.encoder
            self.learned_gene_emb = None

        # Count encoding
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

        # Learned latent queries — cross-attend to gene tokens
        self.latent_queries = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)

        # Cross-attention rounds: latent queries iteratively attend to gene tokens
        n_cross_rounds = getattr(cfg.model, "n_cross_attn_rounds", 1) if cfg else 1
        self.cross_attn_rounds = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout)
            for _ in range(n_cross_rounds)
        ])
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, d_model))

        # Dataset correction token
        if cfg and getattr(cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, d_model))
        else:
            self.dataset_token = None

        # Self-attention transformer (operates on n_latent + CLS tokens, supports MoE)
        layers = _build_transformer_layers(d_model, nhead, d_hid, nlayers, dropout, cfg)
        self.transformer_encoder = FlashTransformerEncoder(layers, gradient_checkpointing=self._gradient_checkpointing)
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

        # Cache for projected ESM2 table: (n_genes, d_model), computed once
        self._esm2_proj_cache = None

    def make_collator(self, cfg, is_train: bool, **kwargs):
        from .. import utils

        ds_emb_mapping = torch.load(
            utils.get_embedding_cfg(cfg).ds_emb_mapping.format(utils.get_embedding_cfg(cfg).size),
            weights_only=False,
        )
        k_top = getattr(cfg.model, "k_top", None)
        return LatentCollator(
            cfg=cfg,
            ds_emb_mapping=ds_emb_mapping,
            n_genes=self.n_genes,
            is_train=is_train,
            k_top=k_top,
        )

    def _get_esm2_proj_table(self, device):
        """Get projected gene embedding table (n_genes, d_model).

        For ESM2 mode: compute encoder(pe_embedding) once and cache.
        For learned mode: return the trainable embedding weight directly.
        """
        if self.use_learned_embeddings:
            return self.learned_gene_emb.weight  # [n_genes, d_model], trainable

        if self._esm2_proj_cache is not None and self._esm2_proj_cache.device == device:
            return self._esm2_proj_cache

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            all_indices = torch.arange(self.n_genes, device=device)
            pe_out = self.pe_embedding(all_indices)  # [n_genes, token_dim]
            proj = self.encoder(pe_out)  # [n_genes, d_model], bf16 under autocast
        self._esm2_proj_cache = proj.to(torch.bfloat16)  # ensure bf16
        return self._esm2_proj_cache

    def forward(self, batch: LatentBatch) -> TokenizerOutput:
        device = next(self.parameters()).device

        gene_indices = batch.gene_indices.to(device)  # [B, k_max]
        gene_counts = batch.gene_counts.to(device)  # [B, k_max]
        gene_mask = batch.gene_mask.to(device)  # [B, k_max] bool
        task_genes = batch.task_genes.to(device)
        task_counts = batch.task_counts.to(device)
        dataset_nums = batch.dataset_nums
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(device)

        B, k_max = gene_indices.shape

        # --- Build sparse gene tokens (only measured genes) ---
        esm2_table = self._get_esm2_proj_table(device)  # [n_genes, d_model]

        # Gather ESM2 projected embeddings for measured genes only
        gene_embs = esm2_table[gene_indices]  # [B, k_max, d_model]

        # Count encoding for measured genes only
        if self.count_encoder is not None:
            counts_input = gene_counts.unsqueeze(-1)  # [B, k_max, 1]
            bin_weights = F.softmax(self.count_encoder(counts_input), dim=-1)  # [B, k_max, 10]
            bin_embeddings = self.bin_encoder(torch.arange(10, device=device))  # [10, d_model]
            count_emb = torch.matmul(bin_weights, bin_embeddings)  # [B, k_max, d_model]
        else:
            count_emb = torch.zeros(B, k_max, self.d_model, device=device)

        gene_tokens = gene_embs + count_emb  # [B, k_max, d_model]

        # --- Cross-attention: latent queries iteratively attend to sparse gene tokens ---
        latent_tokens = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_latent, d_model]

        for cross_attn in self.cross_attn_rounds:
            if self._gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                latent_tokens = checkpoint(
                    cross_attn, latent_tokens, gene_tokens, gene_mask,
                    use_reentrant=False,
                )
            else:
                latent_tokens = cross_attn(latent_tokens, gene_tokens, gene_mask)

        # --- Prepend CLS token (+ optional dataset token) ---
        cls = self.cls_token.unsqueeze(0).expand(B, 1, -1)  # [B, 1, d_model]
        if self.dataset_token is not None:
            ds_tok = self.dataset_token.unsqueeze(0).expand(B, 1, -1)
            src = torch.cat([cls, latent_tokens, ds_tok], dim=1)
        else:
            src = torch.cat([cls, latent_tokens], dim=1)

        src = src * math.sqrt(self.d_model)

        # --- Self-attention transformer + decoder ---
        output = self.transformer_encoder(src, src_key_padding_mask=None)

        # Only decode CLS (+ dataset) token — skip 256 latent tokens
        if self.dataset_token is not None:
            output = output[:, [0, -1], :]  # [B, 2, d_model]
        else:
            output = output[:, :1, :]  # [B, 1, d_model]
        gene_output = self.decoder(output)

        # CLS embedding
        embedding = gene_output[:, 0, :]
        embedding = F.normalize(embedding, dim=1)

        # Dataset embedding
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        # --- Task gene embeddings for decoder ---
        if self.use_learned_embeddings:
            task_gene_embs = self.learned_gene_emb(task_genes.long())  # [B, P+N, d_model]
        else:
            # Must run encoder WITH gradients so its parameters get updated.
            with torch.no_grad():
                task_pe = self.pe_embedding(task_genes.long())  # [B, P+N, token_dim]
            task_gene_embs = self.gene_embedding_layer(task_pe)  # [B, P+N, d_model] — grad flows here

        return TokenizerOutput(
            cell_embedding=embedding,
            task_gene_embs=task_gene_embs,
            task_counts=task_counts,
            dataset_emb=dataset_emb,
            dataset_nums=dataset_nums,
        )
