# Pretrained SE Binary Decoder for Perturbation Response Prediction

## Research Question

Can the pretrained binary decoder from the SE-600M (State Embedding) model improve perturbation response prediction in the STATE TX pipeline?

**Answer: No.** The decoder is not the bottleneck. The SE-600M embeddings are essential (~53% pearson_delta boost), but the decoder initialization is irrelevant at convergence. Code changes have been reverted; this document preserves the experimental findings.

## Background

The SE-600M model contains a **binary decoder** trained on 100M+ cells that maps cell embeddings to per-gene expression predictions. The TX model uses a simple MLP decoder (`LatentToGeneDecoder`) initialized randomly. The hypothesis was that leveraging the SE binary decoder's learned gene-cell relationships could improve prediction quality.

### Architecture Details

**SE Binary Decoder**: `SkipBlock(D) -> SkipBlock(D) -> Linear(D, 1)` where D=4107
- Input: `[gene_embed(2048), cell_embed(2048), rda(1), ds_emb(10)]` per gene
- Output: logit per gene (applied per-gene, so processes G=6546 genes independently)
- Total: ~145M parameters

**TX LatentToGeneDecoder**: `Linear(2058, 1024) -> LN -> GELU -> ... -> Linear(512, 6546) -> ReLU`
- Input: `[cell_embed(2048), ds_emb(10)]` = 2058 dims
- Output: all 6546 gene predictions at once
- Total: ~6.5M parameters

**Key difference**: SE binary decoder processes each gene independently using gene-specific protein embeddings (5120-dim -> 2048-dim). The MLP decoder has no per-gene inductive bias.

## Phase 1: Representation Gap Analysis

Compared TX model outputs vs original SE embeddings on K562 test data:

| Metric | Value |
|--------|-------|
| TX output norms | ~0.97 (near unit, matching L2-normalized SE embeddings) |
| Cosine similarity (TX vs SE) | 0.59 |
| Binary decoder correlation (per-cell) | r=0.78 |
| L2 normalization effect | Negligible (TX outputs already near-unit) |

**Conclusion**: TX outputs live in approximately the same space as SE embeddings. The representation gap is moderate (cosine sim 0.59) but not the primary obstacle.

## Phase 2: Integration Approaches

### Approach A: Direct Binary Decoder (PretrainedBinaryDecoder)

Use the frozen SE binary decoder directly as the TX model's gene decoder.

### Approach B: Knowledge Distillation

Train the standard MLP decoder to mimic the SE binary decoder (r=0.9498 after 50 epochs on 50K cells), then use those weights as initialization for TX training.

### Approach C: Baseline (Random MLP Decoder)

Standard TX training with randomly initialized `LatentToGeneDecoder` for comparison.

## Phase 3: Experimental Results

### 10K-Step Results

All runs: K562, `embed_key=X_state`, `output_space=all`, `model_preset=state`, `precision=bf16-mixed`, `batch_size=16`.

| Approach | b/s | val/expr_loss | pearson_delta | Trainable Params |
|----------|-----|--------------|--------------|-----------------|
| A: Direct Binary Decoder | 0.05 | 48.0 | — (cancelled) | 41.2M + 157M frozen |
| B: Distilled Init | ~24 | 3.75 | 0.289 | 53.8M |
| C: Baseline (random) | ~24 | 3.07 | 0.241 | 53.8M |

### 50K-Step Results (Definitive)

Extended training to determine if the distilled init advantage persists.

#### val/expression_loss convergence

| Step | Distilled (B) | Baseline (C) |
|------|--------------|-------------|
| 5K   | 6.81         | 4.50        |
| 10K  | 3.78         | 3.06        |
| 20K  | 2.09         | 1.55        |
| 30K  | 1.29         | 1.20        |
| 40K  | **1.20**     | **1.18**    |
| 50K  | 1.20         | 1.19        |

**Baseline leads at every checkpoint.** Both converge to ~1.19-1.20, but baseline gets there ~10K steps earlier.

#### Evaluation Results (50K steps, best.ckpt)

| Approach | pearson_delta | mse_delta | discrimination_cosine |
|----------|--------------|-----------|----------------------|
| B: Distilled Init (50K) | 0.311 | 0.00340 | 0.746 |
| C: Baseline (50K) | **0.315** | **0.00339** | 0.738 |

**No benefit from distilled init at convergence.** The early advantage (0.289 vs 0.241 at 10K) is a convergence artifact that disappears with sufficient training.

## Phase 4: Embedding-Only Scaling Experiment

Tested whether the TX model's embedding prediction is the bottleneck by training at different scales with `output_space=embedding` (no decoder, OT Energy loss only).

| Model | params | val/emb_loss (10K) | Notes |
|-------|--------|-------------------|-------|
| tiny (128d, 2L) | ~1M | **0.241** | Best performance |
| default (384d, 8L) | ~41M | 0.265 | Overfits at 10K |

**Key finding**: Model capacity is NOT the bottleneck. A 1M-param model matches/beats a 41M-param model. There is a fundamental OT Energy loss floor at ~0.24 that more parameters cannot break through.

### Compute vs Data-Loading

- `output_space=all` (with decoder): **compute-bound** at ~14.5 b/s on H100
- `output_space=embedding` (no decoder): **data-loading-bound**, GPU util 4-29%

## Phase 5: Embedding Effectiveness

To test whether SE-600M embeddings help or hurt, trained models **without any embeddings** (`embed_key=null`).

| Run | embed_key | output_space | **pearson_delta** | discrim_cosine | mse_delta |
|-----|-----------|-------------|-----------------|----------------|-----------|
| Baseline (50K) | X_state | all | **0.315** | **0.738** | **0.0034** |
| No-emb all (50K) | null | all | 0.206 | 0.499 | 0.0047 |
| No-emb gene (50K) | null | gene | 0.199 | 0.544 | 0.0042 |

**Embeddings provide a ~53% boost in pearson_delta** (0.315 vs 0.206). They are essential, not a bottleneck.

Without embeddings, `output_space=gene` vs `all` produces nearly identical pearson_delta (0.199 vs 0.206), confirming the decoder adds little when the upstream representation is weak.

## Conclusions

### Why the Pretrained Decoder Doesn't Help

1. **End-to-end training overwrites initialization**: With `detach_decoder=false`, decoder gradients flow back into the TX model. The decoder weights are fully retrained, so any initialization advantage is temporary. Distilled and random init converge to the same performance at 50K steps.

2. **The MLP decoder is sufficient**: A 7M-param randomly-initialized MLP reaches the same final performance as one initialized from a 145M-param binary decoder.

3. **The decoder is not the bottleneck**: The bottleneck is the TX model's ability to predict the correct perturbation response direction in embedding space (OT Energy loss floor ~0.24).

### What Actually Matters

1. **SE-600M embeddings are critical**: ~53% boost in pearson_delta. The embedding space captures cellular state information that raw gene expression alone cannot provide efficiently.

2. **Model capacity doesn't help**: Tiny (1M params) matches default (41M params) in embedding-only mode. The K562 dataset (~134K cells) is insufficient for larger models.

3. **The OT Energy loss has a floor (~0.24)**: This is the real limit on performance. Future work should investigate:
   - Alternative losses (per-cell contrastive, MMD)
   - Better regularization (default model overfits at 10K steps)
   - More training data (larger Perturb-seq datasets)
   - Auxiliary losses that constrain individual cell predictions

### Code Status

All decoder-related code changes (`PretrainedBinaryDecoder`, `init_decoder_from`, distillation scripts) have been **reverted** from the codebase. This document and the training artifacts below are the only remaining outputs.

## Output Artifacts

- `/data/replogle_llm/distilled_decoder.pt` — Distilled decoder weights (28MB)
- `/data/replogle_llm/tx_runs/k562_pretrained_decoder_direct/` — Approach A (cancelled at step 99)
- `/data/replogle_llm/tx_runs/k562_distilled_decoder_init/` — Approach B (10K steps)
- `/data/replogle_llm/tx_runs/k562_baseline_decoder/` — Approach C (10K steps)
- `/data/replogle_llm/tx_runs/k562_distilled_50k/` — Approach B (50K steps, definitive)
- `/data/replogle_llm/tx_runs/k562_baseline_50k/` — Approach C (50K steps, definitive)
- `/data/replogle_llm/tx_runs/k562_emb_only_tiny/` — Embedding-only: tiny model
- `/data/replogle_llm/tx_runs/k562_emb_only_default/` — Embedding-only: default model
- `/data/replogle_llm/tx_runs/k562_no_emb_all_50k/` — No-embedding: output_space=all
- `/data/replogle_llm/tx_runs/k562_no_emb_gene_50k/` — No-embedding: output_space=gene
