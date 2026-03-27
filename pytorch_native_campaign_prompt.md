# Campaign: Port State Embedding Training to Native PyTorch

## Goal
Replace PyTorch Lightning with a native PyTorch training loop for the State Embedding model. Work autonomously until all success criteria are met. This is a focused port — DDP only, no FSDP. The deliverable is a complete Lightning-free training pipeline with MFU benchmarks across all scales.

## Context

### Why port away from Lightning?
Benchmarked during the FSDP2 campaign:
- **Init overhead**: Lightning adds ~60s to startup (model summary iterating 6.8B params, strategy setup, callback registration). Native init is ~30s.
- **Per-step overhead**: Lightning's callback dispatch, closure-based optimization, and logging hooks add ~30-40% to step time. Measured 41.2% MFU through Lightning vs expected ~45-50% native.
- **DDP 7B multi-GPU broken**: Lightning calls `model.cuda()` before assigning devices, so all 8 torchrun processes compete for GPU 0. Native PyTorch: `torch.cuda.set_device(local_rank); model.cuda()` — trivially works.
- **Future FSDP2/MoE**: A clean native PyTorch base makes future FSDP2 and expert parallelism integration straightforward, without fighting Lightning abstractions.

### Current architecture
- `StateEmbeddingModel(L.LightningModule)` — owns tokenizer, decoder head, loss, optimizer config
- `LatentTokenizer(nn.Module)` — cross-attention, transformer encoder, count encoding, ESM2 cache
- `FlashTransformerEncoderLayer/Encoder` — self-attention + FFN
- Training orchestrated by `trainer.py` → `L.Trainer`
- Inference via `inference.py` → `Inference` class (loads checkpoint, runs forward)

### Baseline MFU (Lightning, single H100, with torch.compile)

| Scale | Params | B | c/s | MFU |
|-------|--------|---|-----|-----|
| 30M | 31M | 256 | 2,678 | 17.2% |
| 100M | 100M | 256 | 1,579 | 29.7% |
| 600M | 622M | 128 | 400 | 42.8% |
| 1B | 1.09B | 64 | 165 | 43.5% |
| 2B | 1.71B | 32 | 99 | 41.9% |
| 4B | 3.82B | 16 | 38 | 46.7% |
| 7B | 6.77B | 8 | 20 | 43.7% |

DDP scaling (Lightning): 2B 8-GPU 99% efficiency, 4B 8-GPU 91% efficiency.

## Scope — DDP only, no FSDP

This campaign implements DDP only. Remove any FSDP code paths from the trainer (the `experiment.strategy=fsdp` branch). FSDP2 will be added in a future campaign on top of the clean native base. The only strategy is DDP (single-GPU is just DDP with world_size=1).

## What must be ported

### 1. `StateEmbeddingModel` — remove Lightning dependency
**File**: `src/state/emb/nn/model.py`

Currently `L.LightningModule`. Must become `nn.Module`. Changes needed:
- Remove `L.LightningModule` base → `nn.Module`
- Remove `self.save_hyperparameters()` — save cfg yaml in checkpoint instead
- Remove `self.log(...)` calls in `shared_step`, `training_step`, `validation_step` — return loss (and optionally a metrics dict) instead
- Remove `self.trainer.max_steps` / `self.trainer.estimated_stepping_batches` in `configure_optimizers` — move optimizer/scheduler construction to the trainer
- Remove `@torch.compile(disable=True)` decorators on `training_step`/`validation_step`
- Keep `forward()`, `shared_step()`, `_decode()`, `_compute_embedding_for_batch()`, `resize_batch()`, `get_gene_embedding()`, `update_config()`, `on_save_checkpoint()`
- Move `configure_optimizers` logic to a standalone function in trainer.py (e.g., `build_optimizer_and_scheduler(model, cfg, total_steps)`)

**Critical for inference**: `Inference.load_model()` uses `StateEmbeddingModel.load_from_checkpoint()` (Lightning API). Replace with:
```python
model = StateEmbeddingModel(...)  # construct from config
ckpt = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
```

### 2. `trainer.py` — replace with native DDP training loop
**File**: `src/state/emb/train/trainer.py`

Replace `L.Trainer` with a native training loop. DDP only.

**Initialization (single GPU)**:
```python
torch.set_float32_matmul_precision("high")
model = build_model(cfg).cuda()
all_pe = load_pe_embedding(cfg)
model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe.cuda().to(torch.bfloat16))
# Populate ESM2 cache before compile
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    model.tokenizer._get_esm2_proj_table(model.tokenizer.pe_embedding.weight.device)
if compiled:
    model.tokenizer = torch.compile(model.tokenizer)
    model._decode = torch.compile(model._decode)
```

**Initialization (multi-GPU DDP via torchrun)**:
```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
# Same model build as single GPU, then:
model = DDP(model, device_ids=[local_rank])
```

**Training loop**:
```python
for step in range(max_steps):
    batch = next(dataloader_iter)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch)  # or model.module.shared_step for DDP
    loss.backward()
    if (step + 1) % grad_accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    # logging, checkpointing, validation, profiling
```

**Features to implement** (all were handled by Lightning before):
- Mixed precision via `torch.amp.autocast` (replaces `precision="bf16-mixed"`)
- Gradient accumulation (config: `optimizer.gradient_accumulation_steps`)
- Gradient clipping (config: `optimizer.max_grad_norm`)
- Checkpoint save/resume (every N steps, save last, save top-k by val_loss)
- Wandb logging (train_loss, val_loss, learning_rate, cumulative_flops, mfu)
- Validation loop (every N steps, compute val_loss over `limit_val_batches` batches)
- Progress bar (tqdm)
- NSys profiling (push/pop NVTX ranges at configured steps)
- LR scheduling (linear warmup 3% of steps + cosine annealing, from current `configure_optimizers`)
- Analytical FLOPS tracking + rolling-window MFU computation
- Early stopping on `max_steps` (for profiling runs)

### 3. Checkpoint format
**New format** (simple dict, no Lightning wrapper):
```python
{
    "model": model.state_dict(),          # or model.module.state_dict() for DDP
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": global_step,
    "epoch": epoch,
    "cfg_yaml": OmegaConf.to_yaml(cfg),
    "best_val_loss": best_val_loss,
}
```

**Inference loading** must handle:
- New native checkpoints (load `"model"` key)
- Safetensors (already supported, no Lightning dependency)
- Old Lightning checkpoints do NOT need to be supported

### 4. Callbacks → inline logic or utility functions
**File**: `src/state/emb/train/callbacks.py`

All Lightning callbacks become inline logic in the training loop. The callbacks.py file can be repurposed to hold utility functions (e.g., `compute_forward_flops`, `CheckpointManager` class), or deleted if all logic is inline.

| Callback | Replacement |
|----------|-------------|
| `LogLR` | `if step % 100 == 0: wandb.log({"lr": scheduler.get_last_lr()[0]})` |
| `PerfProfilerCallback` | Track batch_times, compute IPM every 60s |
| `ProfilerCallback` | `torch.cuda.nvtx.range_push/pop` at configured steps |
| `ResumeCallback` | LR reset logic in training loop setup |
| `CumulativeFLOPSCallback` | `compute_forward_flops()` + rolling MFU, inline in loop |
| `ModelCheckpoint` | Save every N steps, track best val_loss, delete old checkpoints |

### 5. `_fit.py` CLI entry point
**File**: `src/state/_cli/_emb/_fit.py`

Minimal changes — still calls `trainer_main(cfg)`. Add env var guards:
```python
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = str(cfg.experiment.port)
if "SLURM_NTASKS_PER_NODE" not in os.environ:
    os.environ["SLURM_NTASKS_PER_NODE"] = str(cfg.experiment.num_gpus_per_node)
```

### 6. `inference.py` — update checkpoint loading
**File**: `src/state/emb/inference.py`

- Replace `StateEmbeddingModel.load_from_checkpoint(...)` with direct construction + `load_state_dict`
- Keep safetensors loading path (already Lightning-free)
- Keep all other methods unchanged (`encode`, `encode_adata`, `decode_from_adata`, etc.)

### 7. `finetune_decoder.py` — delete
**File**: `src/state/emb/finetune_decoder.py`

Remove this file. It uses old `from vci.nn.model import ...` import paths and is not part of the active pipeline.

## What must NOT change
- `LatentTokenizer`, `SentenceTokenizer` — pure `nn.Module`, no Lightning dependency
- `FlashTransformerEncoderLayer`, `FlashTransformerEncoder` — pure `nn.Module`
- `SkipBlock`, `CrossAttentionBlock` — pure `nn.Module`
- `LatentCollator`, `LatentBatch` — pure Python/PyTorch
- `H5adSentenceDataset`, `VCIDatasetSentenceCollator` — pure PyTorch Dataset
- `MuonWithAuxAdamW` — pure `torch.optim.Optimizer`
- `_split_muon_parameters` — pure function
- All config YAML files
- `_eval.py`, `_query.py`, `_transform.py`, `_preprocess.py` CLI entry points (use Inference class)
- `eval/emb.py` — uses wandb directly, not Lightning
- `utils.py`, `loss.py`, `vectordb.py` — no Lightning dependency

## Codebase

All code is in `/home/aadduri/state-mcp-local`. The repo is on branch `main` with a clean working tree (FSDP2 work is in stash@{0}, do NOT apply it).

Key files to modify:
- `src/state/emb/nn/model.py` — StateEmbeddingModel (L.LightningModule → nn.Module)
- `src/state/emb/train/trainer.py` — training orchestration (L.Trainer → native loop)
- `src/state/emb/train/callbacks.py` — Lightning callbacks → utility functions or delete
- `src/state/emb/inference.py` — checkpoint loading (load_from_checkpoint → load_state_dict)
- `src/state/_cli/_emb/_fit.py` — CLI entry point (env var guards)
- `src/state/emb/finetune_decoder.py` — DELETE

Key files for reference (do not modify):
- `src/state/tx/optim.py` — Muon optimizer
- `src/state/tx/models/state_transition.py` — `_split_muon_parameters`
- `src/state/emb/nn/tokenizer.py` — LatentTokenizer, SentenceTokenizer
- `src/state/emb/nn/flash_transformer.py` — transformer layers
- `src/state/configs/state-defaults.yaml` — default config
- `src/state/configs/scale/*.yaml` — scale presets (30m, 100m, 600m, 1b, 2b, 4b, 7b)

Python environment: `/home/aadduri/miniconda3/envs/state_env2/bin/python` (torch 2.9.0+cu128)

## Validation

### Step 1: Port and verify training works
Run 100m scale for 200 steps and verify loss decreases:
```bash
python -m src.state emb fit \
  scale=100m experiment.name=native_validation \
  experiment.strategy=ddp experiment.compiled=true \
  model.batch_size=256 wandb.enable=false \
  experiment.profile.enable_profiler=true \
  experiment.profile.max_steps=200 \
  experiment.val_check_interval=99999 \
  experiment.limit_val_batches=0 \
  experiment.checkpoint.every_n_train_steps=99999 \
  dataset.num_train_workers=4 dataset.num_val_workers=0
```
The loss should decrease from ~40+ to below 38 within 200 steps (rough expectation).

### Step 2: MFU benchmark across all scales (single GPU)
For each scale (30m, 100m, 600m, 1b, 2b, 4b, 7b), run with:
- `experiment.compiled=true`
- `wandb.enable=false`
- `experiment.profile.enable_profiler=true`
- `experiment.profile.max_steps=100` (enough for steady-state after compile warmup)
- Batch sizes from the respective YAML configs
- Record: cells/sec, MFU, peak GPU memory

### Step 3: Multi-GPU DDP benchmark
For scales that benefit from multi-GPU (2b, 4b, 7b), run with `torchrun --nproc_per_node=8`:
```bash
torchrun --nproc_per_node=8 -m src.state emb fit \
  scale=7b experiment.name=native_ddp_7b \
  experiment.strategy=ddp experiment.compiled=true \
  model.batch_size=8 wandb.enable=false \
  experiment.profile.enable_profiler=true \
  experiment.profile.max_steps=100 \
  experiment.val_check_interval=99999 \
  experiment.limit_val_batches=0 \
  experiment.checkpoint.every_n_train_steps=99999 \
  dataset.num_train_workers=4 dataset.num_val_workers=0
```
Record MFU and verify near-linear scaling (>90% efficiency for 2B/4B, >85% for 7B).

### Step 4: Verify inference
Save a checkpoint from the 100m validation run, then:
```bash
python -m src.state emb transform \
  --checkpoint /path/to/checkpoint.pt \
  --input /path/to/test.h5ad \
  --output /tmp/test_output.h5ad
```

## Success criteria

### Primary deliverable — MFU benchmark table
Produce a table with these columns, for all 7 scales:

| Scale | Params | B | 1-GPU c/s | 1-GPU MFU | 8-GPU c/s | 8-GPU MFU | 8-GPU efficiency |
|-------|--------|---|-----------|-----------|-----------|-----------|------------------|
| 30M | 31M | 256 | ? | ? | — | — | — |
| 100M | 100M | 256 | ? | ? | — | — | — |
| 600M | 622M | 128 | ? | ? | — | — | — |
| 1B | 1.09B | 64 | ? | ? | — | — | — |
| 2B | 1.71B | 32 | ? | ? | ? | ? | ? |
| 4B | 3.82B | 16 | ? | ? | ? | ? | ? |
| 7B | 6.77B | 8 | ? | ? | ? | ? | ? |

8-GPU numbers only needed for 2B, 4B, 7B (smaller models are memory-bound on single GPU and don't benefit from DDP).

### Additional criteria
1. No `lightning` import anywhere in `src/state/emb/`
2. `emb fit` works single-GPU and multi-GPU (torchrun) at all scales
3. `emb transform` works with new checkpoint format
4. Loss decreases during 200-step 100m validation run
5. MFU matches or exceeds Lightning baseline for each scale
6. Near-linear DDP scaling (>90% efficiency at 2B, >85% at 7B)
7. Commit and push all changes to main in `/home/aadduri/state-mcp-local`
