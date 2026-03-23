# SE Model: Latent Token Utilization Ideas

The LatentTokenizer produces 256 latent tokens after self-attention, but only CLS is used. These proposals aim to leverage the latent tokens for better training signal and downstream representations.

## Proposal A: Hybrid Cross-Attention Decoder

**Status**: Under consideration | **Impact**: High | **Complexity**: Medium

Replace the binary decoder (two SkipBlocks on concat(CLS, gene_emb)) with a cross-attention decoder where task genes attend to **[CLS + 256 latent tokens]** (257 keys total):

```
Q = task_gene_embs           [B, P+N, d_model]
K, V = [CLS, latent_tokens]  [B, 257, d_model]
output = cross_attn(Q, K, V) [B, P+N, d_model]
prediction = Linear(output)  [B, P+N, 1]
```

**Why this works:**
- Each task gene selectively reads from relevant latent tokens (pathway-specific info) AND the CLS token (global cell state)
- CLS still participates as a key → gets direct gradient from expression loss → remains a good downstream embedding
- Replaces expensive SkipBlocks (30-50% of forward FLOPs at 600M scale) with a single cross-attention + linear head
- Gradients flow through all 256 latent tokens, not just CLS → better self-attention training

**Key tension resolved:** CLS must be a good downstream embedding (for TX model), but we also want efficient expression prediction. By making CLS one of 257 keys, it stays in the gradient path while latent tokens do the heavy lifting.

**Risks:**
- CLS might become less informative if latent tokens dominate attention. Could add a small CLS-specific auxiliary loss (e.g., read-depth prediction) as insurance.
- Flash attention for decoder cross-attention needs P+N to not vary (it doesn't — P and N are config constants).

## Proposal B: Masked Latent Reconstruction (Auxiliary Loss)

**Status**: Under consideration | **Impact**: Medium-High | **Complexity**: Low (~50 lines)

Mask ~30% of latent tokens before self-attention, predict the original cross-attention outputs after self-attention. Uses L2 or cosine loss on continuous targets.

```
# After cross-attention, before self-attention:
targets = latent_tokens.detach()          # stop-gradient on reconstruction targets
mask = torch.rand(B, 256) < 0.3
latent_tokens[mask] = self.mask_token     # learned [MASK] parameter

# Run self-attention on [CLS, partially_masked_latent]
output = transformer([CLS, latent_tokens])

# Reconstruction loss at masked positions
predicted = recon_head(output[:, 1:][mask])  # small MLP or linear
loss = F.mse_loss(predicted, targets[mask])
total_loss = expression_loss + alpha * recon_loss
```

**Precedent:** MAE reconstructs continuous pixel values; data2vec and I-JEPA predict continuous latent representations. Continuous target prediction is well-established.

**Why this helps:**
- Prevents latent token collapse (many tokens going unused because only CLS gets gradient)
- Forces self-attention to build globally coherent representations — each token must be predictable from its neighbors
- Self-supervised signal, no additional labels needed
- Complements expression prediction loss

**Hyperparameters to tune:** mask_ratio (start 0.3), alpha loss weight (start 0.1), recon_head architecture (linear vs small MLP).

**Risks:** Too much masking could hurt expression prediction. Disable masking during validation/inference.

## Proposal C: CLS Contrastive with num_downsample

**Status**: Under consideration | **Impact**: Medium | **Complexity**: Medium

Use the existing `num_downsample` mechanism to create augmented pairs. Two views of the same cell (different gene top-K sampling due to random tie-breaking, different task gene sampling) should produce similar CLS embeddings.

```
# With num_downsample=2, collator produces pairs: cell_i_view1, cell_i_view2
# After forward pass, CLS embeddings for the same cell should be similar
# InfoNCE: sim(cls_i_v1, cls_i_v2) high; sim(cls_i_v1, cls_j_v2) low
```

**Why this helps:**
- Explicit training signal for CLS quality, independent of the expression decoder
- Particularly useful if Proposal A shifts expression prediction to latent tokens
- Pairs come for free from the existing augmentation pipeline

**Limitation:** With k_top=4096 deterministic truncation, the cross-attention input is identical across augmentations of the same cell. Only the task gene sampling differs. This means CLS embeddings would already be nearly identical without contrastive loss. To get meaningful augmentation diversity, would need either (a) stochastic top-K (sample proportional to expression rather than hard top-K), or (b) random gene dropout.

**Priority:** Lower than A and B. Only becomes important if Proposal A degrades CLS quality.

## Implementation Order

1. **Proposal A** first — highest impact, solves both the decoder cost problem AND latent token utilization
2. **Proposal B** second — low-effort add-on to prevent latent collapse and improve representations
3. **Proposal C** only if CLS quality degrades after A

## Relationship to Current Optimization Campaign

These proposals interact with the runtime optimizations in `se_optimization_prompt.md`:

- **Proposal A replaces optimization #1** (simplify binary decoder) — the cross-attention decoder IS the simplified decoder
- **Optimization #2** (ESM2 cache for task genes) still applies — task gene embeddings feed the cross-attention queries
- **Optimization #3** (CLS-only decoding in tokenizer) needs revision — we'd keep latent tokens in the output, but the tokenizer decoder (SkipBlock+Linear) still only runs on CLS
- **Optimization #4** (torch.compile) still applies
- **Optimization #5** (remove padding mask) still applies
