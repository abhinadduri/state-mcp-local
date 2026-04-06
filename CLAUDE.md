# STATE

## Cross-Model Guidance: STATE vs STACK

### When to use STATE (this repo)

- **Supervised perturbation prediction**: STATE TX excels at predicting perturbation effects when you have large-scale Perturb-seq data with control groups. Use `state tx train` for training and `state tx infer` for inference.

- **STATE embeddings for TX training**: When you need embeddings specifically compatible with STATE TX training (i.e., `embed_key="X_state"`), use `state emb transform`. STATE EMB is designed to produce embeddings that the TX model consumes.

- **Perturbation data preprocessing**: Use `state tx preprocess_train` for the full pert-transform pipeline (gene alignment, normalization, log fold change computation, HVG selection).

- **Model evaluation**: Use `state tx evaluate` for structured evaluation with cell-eval metrics.

### When to use STACK (the `stack` MCP server)

- **Zero-shot embeddings**: For embedding tasks like clustering, cell-type annotation, batch integration, or disease probing, consider STACK embeddings. STACK's context-aware set attention captures inter-cell relationships and generally outperforms STATE EMB in zero-shot downstream tasks.

- **In-context learning / generation**: For donor-specific prediction, perturbation transfer to unseen cell types, or counterfactual cell state generation, use STACK's generation capabilities. STATE has no equivalent in-context learning capability.

### Cross-model workflows

- STACK embeddings can be used as input features for STATE TX training by setting `embed_key="X_stack"` in `state tx train`. This can improve TX model performance by leveraging STACK's richer embedding space.

## STATE Architecture

- **STATE EMB**: Single-cell embedding model producing per-cell embeddings. Subcommands: `preprocess`, `fit`, `transform`, `eval`, `query`.
- **STATE TX**: Perturbation prediction model that takes cell sets + perturbation labels and predicts expression changes. Subcommands: `preprocess_train`, `train`, `evaluate`, `infer`.

Model presets (via Hydra `model=<preset>`): `state`, `state_sm`, `state_lg`, `context_mean`, `perturb_mean`, `celltypemean`, `decoder_only`, `embedsum`, `globalsimplesum`, `pertsets`, `pseudobulk`, `tahoe_best`, `tahoe_llama_212693232`, `tahoe_llama_62089464`.

## Development

Install with: `uv sync --group dev`

Or use the devcontainer (VS Code → "Reopen in Container" / GitHub Codespaces) for a ready-made environment.
