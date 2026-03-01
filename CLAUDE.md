# STATE MCP Server

## Cross-Model Guidance: STATE vs STACK

### When to use STATE (this MCP server)

- **Supervised perturbation prediction**: STATE TX excels at predicting perturbation effects when you have large-scale Perturb-seq data with control groups. Use `plan_tx_train` / `start_tx_train` for training and `start_tx_inference` for inference.

- **STATE embeddings for TX training**: When you need embeddings specifically compatible with STATE TX training (i.e., `embed_key="X_state"`), use `start_emb_inference`. STATE EMB is designed to produce embeddings that the TX model consumes.

- **Perturbation data preprocessing**: Use `start_preprocess_train` for the full pert-transform pipeline (gene alignment, normalization, log fold change computation, HVG selection).

- **Split TOML generation**: Use `inspect_tx_split_sources` and `plan_tx_split_toml` to design train/test splits with random fewshot holdouts and zeroshot contexts.

- **Model evaluation**: Use `plan_tx_predict` and `start_tx_predict` for structured evaluation with cell-eval metrics.

### When to use STACK (the `stack` MCP server)

- **Zero-shot embeddings**: For embedding tasks like clustering, cell-type annotation, batch integration, or disease probing, consider STACK embeddings via `start_stack_embedding`. STACK's context-aware set attention captures inter-cell relationships and generally outperforms STATE EMB in zero-shot downstream tasks.

- **In-context learning / generation**: For donor-specific prediction, perturbation transfer to unseen cell types, or counterfactual cell state generation, use STACK's `start_stack_generation`. STATE has no equivalent in-context learning capability.

### Cross-model workflows

- STACK embeddings can be used as input features for STATE TX training by setting `embed_key="X_stack"` in `plan_tx_train` / `start_tx_train`. This can improve TX model performance by leveraging STACK's richer embedding space.

## STATE Architecture

- **STATE EMB**: Single-cell embedding model producing per-cell embeddings
- **STATE TX**: Perturbation prediction model that takes cell sets + perturbation labels and predicts expression changes. Supports multiple model presets (state, state_sm, state_lg, context_mean, perturb_mean).

## MCP Server

Run with: `state-mcp` or `python -m state.mcp`

Requires the `mcp` optional dependency: `pip install state[mcp]`
