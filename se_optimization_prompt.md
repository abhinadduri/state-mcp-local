# SE Model Optimization Campaign

## Goal
Maximize training throughput (cells/sec) for the State Embedding model on H100 GPUs, targeting the ability to train on all of basecount (~167M cells) in ~1 day on a single GPU. Establish FLOP-efficiency benchmarks comparing tokenization strategies.

## Current Baseline (2025-03-22)
- **Throughput**: 256 cells/sec (batch_size=128, single H100)
- **Step time**: 501ms (forward=189ms, backward=309ms, optimizer=2ms)
- **Loss**: MMD (energy distance)
- **Model**: 37M trainable params (8-layer transformer, d_model=512, nhead=16)
- **Sequence length**: pad_length=2048 (all genes as explicit tokens)
- **Token dim**: 5120 (ESM2 protein embeddings, frozen)
- **GPU memory**: 52 GB peak / 80 GB available
- **Bottleneck**: Compute-bound (99% forward+backward on O(2048²) attention)
- At current throughput, 1 epoch over 167M cells ≈ 7.5 days

## Architecture: Tokenizer Abstraction

The core insight: the current collator mixes *tokenization strategy* (how to convert raw counts into transformer tokens) with *data loading*. These should be decoupled. The model's job — CLS bottleneck + transformer + decoder — is agnostic to how tokens are produced.

### Interface

```python
class Tokenizer(nn.Module):
    """Converts raw gene expression counts into transformer-ready tokens."""

    def tokenize(self, counts, dataset_info) -> torch.Tensor:
        """Returns (batch, seq_len, token_dim) ready for the transformer."""
        ...

    @property
    def token_dim(self) -> int: ...

    @property
    def seq_len(self) -> int: ...
```

The model receives tokens from the tokenizer and runs: `CLS + transformer(tokens) → CLS embedding → decoder → reconstruction`.

### Implementation 1: SentenceTokenizer (current approach, preserves pretrained model)

This wraps the existing logic — no behavior change, just extraction into a clean interface.

**Collator output**: sampled gene indices (P+N or P+N+S), counts, sentence indices
**Tokenizer**:
1. Look up frozen protein embeddings: `pe_embedding(gene_indices)` → (B, 2048, 5120)
2. Project: `encoder(5120 → d_model)` → (B, 2048, 512)
3. Add count embeddings (soft binning)
4. Prepend CLS token

**Output**: (B, 2048, 512) — explicit gene tokens
**Attention cost**: O(2048² × 512) per layer = 2.1B ops

### Implementation 2: LatentTokenizer (new, latent bottleneck)

**Collator output**: global-aligned count vector (19,790-dim) + measurement mask
**Tokenizer**:
1. Build per-gene tokens: for each of the 19,790 gene positions:
   - **Measured gene**: `gene_emb[i] + count_emb(count_i)` (protein embedding + count encoding, same as current)
   - **Unmeasured gene**: `missing_emb[i]` (learned per-position embedding, replaces the entire token)
2. Project to latent tokens: `Linear(19790 × token_dim → n_latent × d_model)` or cross-attention
3. Prepend CLS token

**Output**: (B, n_latent, d_model) — e.g., (B, 128, 512)
**Attention cost**: O(128² × 512) per layer = 8.4M ops (**250x cheaper than SentenceTokenizer**)

Key design decisions:
- **n_latent**: number of latent tokens (try 64, 128, 256)
- **d_model (emsize)**: reuse existing 512 for both tokenizers
- **Missing embedding**: learned per-gene-position `nn.Parameter(19790, token_dim)`, used in place of both gene_emb and count_emb when gene is unmeasured
- **Reconstruction target**: same as current — predict expression for queried genes via binary_decoder
- **Gene-to-latent projection**: start with a simple linear; can try cross-attention later if needed

### Collator changes for LatentTokenizer

The collator produces a global-aligned count vector using the existing `ds_emb_map`:
```python
# For each cell:
global_counts = torch.zeros(19790)
measurement_mask = torch.zeros(19790, dtype=torch.bool)
valid = ds_emb_map[dataset] != -1
global_counts[ds_emb_map[dataset][valid]] = log1p_counts[valid]
measurement_mask[ds_emb_map[dataset][valid]] = True
```
No data reprocessing needed — `ds_emb_map` from preprocessing already provides the mapping.

### FLOP-Efficiency Comparison

The campaign's central experiment: **at equal FLOP budgets, which tokenizer achieves lower validation loss?**

| Metric | SentenceTokenizer | LatentTokenizer (n=128) |
|--------|-------------------|------------------------|
| Attention FLOPs/layer | 2.1B | 8.4M |
| Ratio | 250x | 1x |
| Estimated throughput | ~256 cells/sec | ~5,000+ cells/sec (projected) |
| 1 epoch (167M cells) | ~7.5 days | ~9 hours (projected) |

If LatentTokenizer reaches comparable val loss at 10x fewer FLOPs, it's the strictly better vehicle for scaling.

## SE vs Stack Reference

| | Stack (StateICL) | SE (SentenceTokenizer) | SE (LatentTokenizer, projected) |
|---|---|---|---|
| Attention seq len | 100 latent tokens | 2048 gene tokens | 128 latent tokens |
| Token dim | 16 | 512 | 512 (d_model) |
| Attention cost/layer | O(100² × 16) = 160K | O(2048² × 512) = 2.1B | O(128² × 512) = 8.4M |
| Missing gene handling | zeros | N/A (sampled genes only) | learned per-position embedding |
| Protein embeddings | none | ESM2 5120-dim | ESM2 (for measured genes) + learned missing |

## Next Optimization Opportunities

### 1. Simplify binary decoder (HIGH IMPACT)
- `model.py:82-86` — two SkipBlocks applied to all 1024 task genes per cell
- Input dim = output_dim + d_model + z_dim, scales quadratically with d_model
- At 600M (d=2048): each SkipBlock is Linear(2571→5142)+Linear(5142→2571) × 1024 tokens × B — likely 30-50% of forward FLOPs
- **Options**: (a) bilinear decoder `(cell_emb * W * gene_emb).sum()`, (b) project to small bottleneck dim (256) before concat, (c) single Linear→ReLU→Linear(1)

### 2. Use ESM2 proj cache for task gene embeddings (HIGH IMPACT, trivial fix)
- `tokenizer.py:564-566` — re-runs `Linear(5120→d_model)` on 1024 task tokens per cell
- `_esm2_proj_cache` already has the projected result for all genes
- **Fix**: `task_gene_embs = esm2_table[task_genes.long()]` instead of `gene_embedding_layer(pe_embedding(task_genes))`
- Eliminates a full encoder forward on 1024×5120 tokens per step

### 3. CLS-only decoding in tokenizer (MEDIUM-HIGH, trivial fix)
- `tokenizer.py:552` — SkipBlock+Linear runs on all 257 tokens (256 latent + CLS), only CLS output used
- 99% of tokenizer decoder compute is wasted
- **Fix**: slice `output[:, 0:1, :]` before decoder, or `output[:, [0, -1], :]` if dataset_token

### 4. torch.compile (MEDIUM)
- `@torch.compile(disable=True)` on training_step/validation_step
- `compiled=False` in tokenizer and model construction
- Expected 20-40% from kernel fusion on H100 with bf16
- Target: transformer_encoder, binary_decoder, cross-attention block

### 5. Remove padding mask ops when k_top is set (MEDIUM, trivial fix)
- `tokenizer.py:524-528` — `k * mask` and `v * mask` elementwise multiplies on (B, nhead, k_max, head_dim)
- With k_top=4096, most/all cells have exactly 4096 genes (no padding), making the mask ops pure overhead
- **Fix**: skip masking when k_top is set and all cells have >= k_top genes

### 6. Muon optimizer (MEDIUM)
- Replace AdamW; already used in TX model
- Cheaper per-step updates, potentially better convergence

### Latest Benchmarks (H100, bf16-mixed, latent tokenizer, k_top=4096)

| Scale | d_model | nlayers | B | ds | cells/sec | Mem GB | Params |
|-------|---------|---------|---|----|-----------|--------|--------|
| 30M   | 512     | 8       | 128 | 1 | **787** | 16.5 | 38M |
| 100M  | 1024    | 8       | 128 | 1 | **388** | 27.7 | 132M |
| 100M  | 1024    | 8       | 32  | 4 | **97**  | ~28  | 132M |
| 600M  | 2048    | 16      | 64  | 1 | **107** | 36.4 | 755M |
| 600M  | 2048    | 16      | 32  | 4 | **27**  | 67.6 | 755M |

- num_downsample=4 costs exactly 4x in unique cell throughput (expected)
- 600M model fits on single H100 thanks to k_top=4096

## Completed Optimizations
- [x] Loss init once in __init__ (not every step)
- [x] LR scheduler double-stepping fix
- [x] CollatedBatch NamedTuple (clarity)
- [x] Cache ds_emb_idxs tensors in collator
- [x] expand() instead of repeat() in resize_batch + shared_step
- [x] Reduce logging frequency
- [x] Skip tabular-only work for non-tabular losses
- [x] Exclude frozen pe_embedding from optimizer (saves ~800MB Adam state)
- [x] torch.no_grad() on frozen embedding lookups
- [x] Remove CUDA_LAUNCH_BLOCKING=1
- [x] Enable TF32 via float32_matmul_precision("high")
- [x] Fix SDPA dropout (was always on, even during eval)
- [x] Enable mem_efficient_sdp backend
- [x] Add pin_memory + prefetch_factor to val DataLoader
- [x] Increase default batch_size from 96 to 128
- [x] Require experiment.name for training
