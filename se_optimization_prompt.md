# SE Model Optimization Campaign

## Goal
Maximize training throughput (cells/sec) for the State Embedding model on H100 GPUs, targeting the ability to train on all of basecount (~167M cells) in ~1 day on a single GPU. Build toward a multi-species model (200K+ genes). Work maximally in parallel, spawning teammates as needed to do literature search, implement features, etc. The code to change is in /home/aadduri/state-mcp-local. Use local resources well and also use as much of the slurm backend as is helpful towards the campaign goal.

## Current Architecture: LatentTokenizer (sparse Perceiver cross-attention)

The model uses a Tokenizer abstraction (`src/state/emb/nn/tokenizer.py`) that decouples tokenization from the model. The **LatentTokenizer** is the primary approach.

### Data flow
```
LatentCollator (CPU):
  Raw cell counts → sparse (gene_indices[k], gene_counts[k]) in global space
  Top-K truncation: keep top k_top=4096 genes by expression
  Sample P+N task genes for reconstruction (from ALL measured genes, before truncation)

LatentTokenizer.forward (GPU):
  1. Gather ESM2 projected embeddings for measured genes:  esm2_table[indices] → (B, k, d_model)
  2. Count encoding for measured genes:  MLP(counts) → (B, k, d_model)
  3. Gene tokens = esm2_emb + count_emb:  (B, k, d_model)
  4. Cross-attention: latent_queries(256) attend to gene_tokens(k) → (B, 256, d_model)
  5. CLS + self-attention transformer + decoder → cell embedding

Binary decoder (model.py):
  6. Task gene embeddings: pe_embedding(task_genes) → gene_embedding_layer → (B, P+N, d_model)
  7. Concat [task_gene_embs, cell_emb_expanded, read_depth] → (B, P+N, d_model+output_dim+1)
  8. Two SkipBlocks + Linear(1) → (B, P+N, 1) predictions
  9. MMD loss (energy distance)
```

### Key design properties
- **Sparse**: only measured genes materialized — no (B, n_genes, d_model) dense intermediate
- **Scales to 200K+ genes**: cost depends on measured genes per cell (k), not vocabulary size
- **ESM2 cross-species bridge**: orthologs with similar protein sequences get similar K/V vectors in cross-attention
- **Missing genes are absent, not padded**: correct inductive bias for scRNA-seq
- **Top-K truncation**: only top 4096 expressed genes enter cross-attention; task genes still sampled from all measured genes
- **num_downsample**: duplicates each cell N times in the batch for better MMD distribution estimation (costs Nx in unique cell throughput)

### Config defaults
```yaml
model.tokenizer: sentence    # or "latent"
model.n_latent: 256          # latent query count
model.k_top: 4096            # top-K genes for cross-attention (null = all)
model.num_downsample: 1      # MMD augmentation factor
model.emsize: 512            # d_model
model.nlayers: 8             # transformer layers
model.nhead: 16
model.batch_size: 128
loss.name: mmd
loss.kernel: energy
```

### Usage
```bash
# LatentTokenizer (recommended)
python -m src.state emb fit experiment.name=latent_256 model.tokenizer=latent

# 100M scale
python -m src.state emb fit experiment.name=latent_100M model.tokenizer=latent model.emsize=1024 model.d_hid=2048 model.output_dim=1024

# 600M scale
python -m src.state emb fit experiment.name=latent_600M model.tokenizer=latent model.emsize=2048 model.d_hid=4096 model.output_dim=2048 model.nlayers=16

# With top-K override
python -m src.state emb fit experiment.name=test model.tokenizer=latent model.k_top=2048
```

## Benchmarks (single H100, bf16-mixed, 2025-03-23)

### Throughput by scale (LatentTokenizer, k_top=4096)
| Scale | d_model | nlayers | B | ds | cells/sec | Mem GB | Params |
|-------|---------|---------|---|----|-----------|--------|--------|
| 30M   | 512     | 8       | 128 | 1 | **787** | 16.5 | 38M |
| 100M  | 1024    | 8       | 128 | 1 | **388** | 27.7 | 132M |
| 100M  | 1024    | 8       | 32  | 4 | **97**  | ~28  | 132M |
| 600M  | 2048    | 16      | 64  | 1 | **107** | 36.4 | 755M |
| 600M  | 2048    | 16      | 32  | 4 | **27**  | 67.6 | 755M |

- num_downsample=4 costs exactly 4x in unique cell throughput (expected)
- 600M model fits on single H100 thanks to k_top=4096

### Effect of k_top (LatentTokenizer, bf16-mixed, ds=1)
| d_model | k_top | B | cells/sec | Mem GB | Speedup |
|---------|-------|---|-----------|--------|---------|
| 512     | all (~19K) | 128 | 608 | 41.2 | baseline |
| 512     | 4096 | 128 | 787 | 16.5 | 1.3x, 60% less memory |
| 1024    | all (~19K) | 64 | 261 | 37.0 | baseline (OOMs at B=128) |
| 1024    | 4096 | 128 | 388 | 27.7 | 1.5x + enables B=128 |

### Sentence vs Latent comparison (k_top=all, ds=1)
| Tokenizer | d_model | B | cells/sec | Mem GB |
|-----------|---------|---|-----------|--------|
| sentence  | 512     | 128 | 255 | 53.6 |
| latent    | 512     | 128 | 608 | 41.2 |
| sentence  | 1024    | 64  | 129 | 48.4 |
| latent    | 1024    | 128 | 388 | 27.7 |

Latent is **2.4x** faster at d=512, **3x** at d=1024 (including batch size gains).

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

## Completed Optimizations
- [x] Tokenizer abstraction (SentenceTokenizer + LatentTokenizer)
- [x] Sparse Perceiver cross-attention (only measured genes materialized)
- [x] Top-K gene truncation (k_top=4096 default)
- [x] num_downsample support in LatentCollator
- [x] Loss init once in __init__ (not every step)
- [x] LR scheduler double-stepping fix
- [x] CollatedBatch / LatentBatch NamedTuples
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
- [x] task_counts moved to GPU in both tokenizers
