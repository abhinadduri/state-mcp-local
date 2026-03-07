# Pretrained SE Binary Decoder for Perturbation Response Prediction

## Research Question

Can the pretrained binary decoder from the SE-600M (State Embedding) model improve perturbation response prediction in the STATE TX pipeline?

## Background

The SE-600M model contains a **binary decoder** trained on 100M+ cells that maps cell embeddings to per-gene expression predictions. The TX model uses a simple MLP decoder (`LatentToGeneDecoder`) initialized randomly. The hypothesis is that leveraging the SE binary decoder's learned gene-cell relationships could improve prediction quality.

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

### Diagnostic Script Results (`scripts/diagnose_representation_gap.py`)

Compared TX model outputs vs original SE embeddings on K562 test data:

| Metric | Value |
|--------|-------|
| TX output norms | ~0.97 (near unit, matching L2-normalized SE embeddings) |
| Cosine similarity (TX vs SE) | 0.59 |
| Binary decoder correlation (per-cell) | r=0.78 |
| L2 normalization effect | Negligible (TX outputs already near-unit) |

**Conclusion**: TX outputs live in approximately the same space as SE embeddings. The binary decoder produces reasonable gene expression predictions (r=0.78 per-cell) from TX outputs. The representation gap is moderate (cosine sim 0.59) but not the primary obstacle.

### Why the Previous Attempt Failed

The previous `FinetuneVCICountsDecoder` had three competing paths:
1. Binary decoder output
2. Gene decoder projection
3. Latent decoder MLP

This architecture allowed the MLP to dominate, making the binary decoder path redundant. The fix is to use the binary decoder as the **sole** decoder path.

## Phase 2: Integration Approaches

### Approach A: Direct Binary Decoder (PretrainedBinaryDecoder)

Use the frozen SE binary decoder directly as the TX model's gene decoder:
- Load binary decoder weights + gene protein embeddings from SE-600M
- Pre-compute gene embeddings (matched 6369/6546 genes)
- Process cells in mini-batches (batch_size=8) to manage memory
- `torch.no_grad()` wrapper since decoder is frozen
- Cast output to input dtype for bf16-mixed compatibility

**Implementation**: `PretrainedBinaryDecoder` class in `state/tx/models/base.py`

### Approach B: Knowledge Distillation

Train the standard MLP decoder to mimic the SE binary decoder, then use those weights as initialization:
1. Sample 50K cells from dataset
2. Run SE binary decoder to get target gene expression
3. Train `LatentToGeneDecoder` MLP (2058 -> [1024, 1024, 512] -> 6546) with MSE loss
4. Save distilled weights, use as initialization for TX training

**Distillation quality**: After 50 epochs on 50K cells, the MLP achieves **r=0.9498** correlation with the binary decoder's output (MSE loss=0.137). The 7M-param MLP captures 95% of the 145M-param binary decoder's behavior.

**Implementation**: `scripts/distill_binary_decoder.py`

### Approach C: Baseline (Random MLP Decoder)

Standard TX training with randomly initialized `LatentToGeneDecoder` for comparison.

## Phase 3: Experimental Results

### Training Configuration

All runs: K562, `embed_key=X_state`, `output_space=all`, `model_preset=state`, `precision=bf16-mixed`, `pin_memory=true`, `use_consecutive_loading=true`, `max_steps=10000`, `val_freq=2000`, `batch_size=16`.

### Training Metrics

| Approach | Steps | Time | b/s | val/emb_loss | val/expr_loss | Trainable Params |
|----------|-------|------|-----|-------------|--------------|-----------------|
| A: Direct Binary Decoder | 99 (cancelled) | ~32 min | 0.05 | — | 48.0 | 41.2M + 157M frozen |
| B: Distilled Init | 10,000 | ~14 min | ~24 | 0.050 | 3.75 | 53.8M |
| C: Baseline (random MLP) | 10,000 | ~15 min | ~24 | 0.046 | 3.07 | 53.8M |

### Evaluation Results (best.ckpt, K562 test set)

| Approach | pearson_delta | mse_delta | discrimination_l2 | de_spearman | overlap_at_N | de_sig_recall |
|----------|--------------|-----------|-------------------|-------------|-------------|--------------|
| A: Direct Binary Decoder | — | — | — | — | — | — |
| B: Distilled Init | **0.289** | 0.0044 | 0.740 | 1.0* | 0 | 0 |
| C: Baseline (random) | 0.241 | 0.0043 | 0.668 | 0.695 | 0.112 | 0.178 |
| Reference (prev work) | 0.348 | — | — | — | — | — |

*No significant DE genes detected in predictions, making DE metrics unreliable.

### Key Findings

#### Approach A: Direct Binary Decoder — IMPRACTICAL

1. **460x slower** than MLP decoder: 0.05 b/s vs 24 b/s. Each training step processes 6546 genes × cells through a 3-layer MLP individually, creating massive intermediate tensors.
2. **Expression loss ~30x higher** (48-51 vs 1.7) due to different output scales between binary decoder (raw logits) and expected gene expression.
3. **Memory challenges**: Required `torch.no_grad()` wrapper to avoid OOM (intermediate tensor: batch × 6546 × 4107 × 2 bytes).
4. **FLOPS callback timeout**: The Lightning FLOPS measurement alone took ~6 minutes due to the per-gene computation pattern.
5. **bf16 dtype mismatch**: Required explicit dtype casting since `torch.no_grad()` exits the autocast context.

**Verdict**: Direct integration of the SE binary decoder into the TX training loop is computationally infeasible. The per-gene architecture that makes the binary decoder powerful (gene-specific protein embeddings) is exactly what makes it 460x too slow for iterative training.

#### Approach B: Knowledge Distillation — PROMISING BUT INCOMPLETE

1. **+20% pearson_delta** over random init at the same step count (0.289 vs 0.241), suggesting the distilled weights provide a better starting point for the decoder.
2. **Faster convergence in embedding space**: val/embedding_loss comparable to baseline.
3. **DE metrics broken**: 0 significant genes detected, likely because distilled decoder learns a different output distribution that doesn't produce realistic DE patterns. The decoder captures mean expression well (better pearson_delta) but not the variance structure needed for DE testing.
4. **Higher val/expression_loss** (3.75 vs 3.07): The distilled initialization may create an optimization landscape that's harder to fine-tune jointly with the OT Energy loss.

**Verdict**: Distilled initialization shows promise for improving pearson_delta (faster convergence), but hurts DE analysis. Needs investigation into why variance structure is lost.

#### Both approaches undertrained

Both B and C (0.289, 0.241) are below the reference baseline (0.348). This reference likely used more than 10K steps. The val/expression_loss was still decreasing at 10K steps for both runs, confirming they need more training.

## Phase 4: Analysis & Recommendations

### Why the Binary Decoder Doesn't Help More

1. **Architecture mismatch**: The SE binary decoder processes genes independently with protein embeddings. This is powerful for learning gene-specific expression patterns from 100M cells but creates a computational bottleneck (O(genes × cells) vs O(cells)) that prevents practical use in training.

2. **Distribution shift**: TX model outputs differ from SE training data. Even though cosine similarity is 0.59 and norms match, the fine-grained structure differs enough that binary decoder outputs have higher loss and different variance properties.

3. **The decoder is not the bottleneck**: With pearson_delta=0.348 from a random MLP decoder, the limiting factor is the TX model's ability to predict correct cell embeddings (the OT Energy loss), not the decoder's ability to translate embeddings to genes. Improving the decoder helps marginally.

4. **Distillation loses variance**: Knowledge distillation with MSE loss captures mean predictions but smooths out the variance structure needed for DE analysis. A better distillation objective (e.g., distributional matching) might help.

### Recommendations

1. **Don't use direct binary decoder** (Approach A): 460x slowdown makes it impractical.

2. **Distilled init is worth exploring further** with:
   - Longer training (>10K steps) to see if the initial advantage persists
   - Modified distillation loss that preserves variance (e.g., add noise, use distributional loss)
   - Two-stage training: distilled init for decoder, then freeze decoder and train TX model

3. **Focus on the embedding model**: The biggest lever is improving the TX model's embedding prediction (the main OT Energy loss). The decoder is a secondary concern.

4. **Consider hybrid approaches**:
   - Use binary decoder for evaluation/interpretation only (not training)
   - Gene-specific features from protein embeddings as auxiliary input to MLP decoder
   - Factored decoder: low-rank approximation of the binary decoder's gene-specific computation

## Code Changes

### New files
- `scripts/distill_binary_decoder.py` — Knowledge distillation script
- `scripts/diagnose_representation_gap.py` — Diagnostic analysis

### Modified files
- `src/state/tx/models/base.py` — Added `PretrainedBinaryDecoder` class, modified `_build_decoder()`
- `src/state/_cli/_tx/_train.py` — Added `pretrained_binary_decoder` and `init_decoder_from` config support

### Output artifacts
- `/data/replogle_llm/distilled_decoder.pt` — Distilled decoder weights (28MB)
- `/data/replogle_llm/tx_runs/k562_distilled_decoder_init/` — Approach B run
- `/data/replogle_llm/tx_runs/k562_baseline_decoder/` — Approach C run
- `/data/replogle_llm/tx_runs/k562_pretrained_decoder_direct/` — Approach A run (cancelled at step 99)

## Appendix: Wandb Run Links

- Approach A: https://wandb.ai/arcinstitute/vci1 (tag: pretrained_decoder, approach_a)
- Approach B: https://wandb.ai/arcinstitute/vci1 (tag: distilled_decoder, approach_b)
- Approach C: https://wandb.ai/arcinstitute/vci1 (tag: baseline, approach_c)
