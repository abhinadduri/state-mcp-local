# SE Model: In-Context Learning via Position-Aligned Latent Exchange

## Core Idea

The SE LatentTokenizer's 256 learned latent queries are **shared across all cells**. This means `latent_token[i]` from cell A and `latent_token[i]` from cell B were produced by the same query attending to different cells' gene profiles. After training, each latent position develops a **positional semantics** — position 47 might specialize in cell cycle, position 128 in immune activation, etc.

This enables a dual-axis attention pattern structurally analogous to Stack's `TabularAttentionLayer`, but operating in compressed latent space:

```
Intra-cell:  (B*C, 257, d_model)  — latent tokens attend within a cell
Inter-cell:  (B*256, C, d_model)  — cells attend per latent position
```

Each of the 256 latent positions becomes an **independent inter-cell communication channel**. When cells exchange information at position `i`, they are comparing their states for a specific biological program — "how does my immune signaling compare to the population?" This is richer than a single CLS exchange and more structured than Stack's approach (which flattens all gene programs into one 1600-dim vector for inter-cell attention).

## Comparison to Stack

Stack alternates two attention axes every layer:

| | Stack (TabularAttention) | SE-ICL (Latent Exchange) |
|---|---|---|
| Intra-cell | 100 gene tokens self-attend `(B*C, 100, 16)` | 257 latent+CLS tokens self-attend `(B*C, 257, d_model)` |
| Inter-cell | cells attend as flattened vectors `(B, C, 1600)` | cells attend per latent position `(B*256, C, d_model)` |
| Channels | 1 mixed channel (1600-dim) | 256 specialized channels (d_model each) |
| Gene representation | learned linear from counts, no protein info | ESM2 protein embeddings, cross-species transfer |
| Gene vocabulary | fixed `n_genes`, zeros for unmeasured | sparse, variable, scales to 200K+ |
| Per-cell capacity | 100 * 16 = 1,600 dims | 256 * d_model = 131K dims (d=512) |

**Attention cost per layer** (C=128 cells, comparable to Stack's sample_size=256):
- Intra-cell: `C * 257^2 = ~8.5M` entries — similar to Stack's `C * 100^2 = ~2.6M`
- Inter-cell: `256 * C^2 = ~4.2M` entries — cheaper than Stack's `C^2 * 100 = ~6.6M` (at C=256)

SE-ICL is comparable or cheaper in attention cost while being richer in per-cell representation.

## Architecture

### Per-cell pipeline (unchanged from current SE)
```
1. LatentCollator: raw counts → sparse (gene_indices[k], gene_counts[k]), top-K truncation
2. Cross-attention: 256 learned queries attend to k gene tokens → (C, 256, d_model)
3. Prepend CLS token → (C, 257, d_model)
```

### Dual-axis transformer (new, replaces FlashTransformerEncoder)
Each layer alternates:
```
For each layer l = 1..L:
  # Intra-cell: latent tokens attend within their cell
  x = reshape(B, C, 257, d_model) → (B*C, 257, d_model)
  x = self_attn_intra(x)  # flash attention, 257 tokens
  x = reshape back to (B, C, 257, d_model)

  # Inter-cell: cells attend per latent position (including CLS at position 0)
  x = x.permute(0, 2, 1, 3)  → (B, 257, C, d_model)
  x = reshape → (B*257, C, d_model)
  x = self_attn_inter(x)  # flash attention, C tokens
  x = reshape back to (B, C, 257, d_model)

  # FFN (shared or separate for intra/inter)
  x = ffn(x)
```

### Decoder (unchanged)
```
4. Extract CLS token → cell embedding
5. Binary decoder: concat(CLS, gene_emb) → predicted expression
6. MMD loss
```

## Data Loading

### New: ContextDataset

Replace the current single-cell `H5adSentenceDataset` with a `ContextDataset` that yields **groups of C cells** from the same biological context (same H5AD file, contiguous or locality-sampled).

```python
class ContextDataset(Dataset):
    """Yields groups of C cells from the same biological context.

    Each __getitem__ returns C cells from one H5AD file.
    Cells are sampled contiguously (nearby in the file) to share
    tissue/donor/condition context, similar to Stack's locality sampling.
    """

    def __init__(self, cfg, context_size: int = 128):
        self.context_size = context_size
        # ... load file list, cell counts per file ...

    def __getitem__(self, idx):
        # Map idx to (file, start_position)
        # Return C consecutive cells as a list of (counts, idx, dataset, dataset_num)
        ...
```

The `LatentCollator` would be updated to accept a list of C-cell groups and produce a batched `LatentBatch` with shape `(B, C, k_max)` instead of `(B, k_max)`. Here B is the number of contexts (meta-batch), C is cells per context.

### Remove num_downsample

The `num_downsample` augmentation in the collator should be removed as a first step. It complicates the collator and is orthogonal to the ICL approach. It can be re-added later if needed for the MMD loss, but contextual training (C cells per sample) already provides natural augmentation through inter-cell information.

## Training Objective

### Phase 1: Contextual denoising

Keep the existing per-cell expression prediction loss, but with **asymmetric corruption** across cells in a context:
- Randomly corrupt 20-50% of cells heavily (mask 80%+ of their gene tokens before cross-attention)
- Leave the remaining cells lightly corrupted (mask 10-20%, as current)
- All cells predict their own expression as usual

Heavily corrupted cells must rely on inter-cell context to reconstruct their expression — their own cross-attention output is too impoverished. This naturally incentivizes ICL without requiring new loss functions.

### Phase 2: Perturbation prediction (Stack-style finetuning)

Given control cells as context + empty query slots, predict post-perturbation expression:
- Add `query_pos_embedding` to distinguish context from query cells (as in Stack)
- Add causal mask on inter-cell attention: control cells cannot attend to query cells
- Training: flow-matching interpolation (random fraction of query cells receive ground-truth)

## Auxiliary: Masked Latent Reconstruction

Can be combined with either phase. Mask ~30% of latent tokens before self-attention, predict the original cross-attention outputs after self-attention using L2 loss on continuous targets.

```
targets = latent_tokens.detach()          # stop-gradient
mask = torch.rand(B, C, 256) < 0.3
latent_tokens[mask] = self.mask_token     # learned parameter

output = dual_axis_transformer([CLS, latent_tokens])

predicted = recon_head(output[:, :, 1:][mask])
recon_loss = F.mse_loss(predicted, targets[mask])
total_loss = expression_loss + alpha * recon_loss
```

**Precedent:** MAE, data2vec, I-JEPA all reconstruct continuous representations successfully. The targets here are cross-attention outputs that encode gene identity (ESM2) + expression level.

## Implementation Plan

### Step 1: Remove num_downsample from LatentCollator
- Remove augmentation duplication logic
- Simplify collator to single-cell batching only

### Step 2: ContextDataset + ContextCollator
- New dataset class that yields C-cell groups from the same file
- New collator that produces `(B, C, k_max)` batched latent representations
- Cross-attention runs per-cell as before: reshape `(B*C, k_max, d_model)`

### Step 3: DualAxisTransformerEncoder
- New transformer module that alternates intra-cell and inter-cell self-attention
- Initialize from pretrained SE weights for intra-cell layers
- Inter-cell layers init with near-zero output projections (model starts as if no inter-cell communication, gradually learns to use it)

### Step 4: Contextual denoising objective
- Asymmetric gene masking across cells in a context
- All cells still predict their own expression via the existing binary decoder + MMD loss

### Step 5: Perturbation prediction finetuning
- Add query_pos_embedding, causal masking
- Flow-matching interpolation
- This becomes the path to replacing Stack's TX pipeline

## Key Advantages Over Stack

1. **ESM2 cross-species bridge**: orthologs get similar latent representations, enabling multi-species ICL
2. **256 specialized communication channels** vs Stack's 1 mixed channel
3. **Sparse gene handling**: scales to 200K+ genes without affecting inter-cell attention cost
4. **Modular**: cross-attention (per-cell gene compression) is decoupled from inter-cell communication
5. **Pretrained initialization**: can warm-start from existing SE checkpoint, only adding inter-cell layers
