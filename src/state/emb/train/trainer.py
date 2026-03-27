"""Native PyTorch training loop for State Embedding model.

Replaces the former Lightning Trainer with a minimal, high-performance loop.
Supports DDP and FSDP2 strategies (single-GPU is just world_size=1 without wrapper).
"""

import os
import time
import logging
import collections
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from datetime import timedelta
from tqdm import tqdm
from omegaconf import OmegaConf

from ..nn.model import StateEmbeddingModel
from ..nn.tokenizer import SentenceTokenizer, LatentTokenizer
from ..data import H5adSentenceDataset
from ..utils import get_latest_checkpoint, get_embedding_cfg, get_dataset_cfg
from .callbacks import compute_forward_flops

log = logging.getLogger(__name__)

# H100 peak bf16 TFLOPS
H100_PEAK_TFLOPS = 989.5


def get_embeddings(cfg):
    """Load ESM2 embeddings and special tokens."""
    all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
    if isinstance(all_pe, dict):
        all_pe = torch.vstack(list(all_pe.values()))
    all_pe = all_pe.cuda()
    return all_pe


def build_optimizer_and_scheduler(model, cfg, total_steps):
    """Build optimizer and LR scheduler from config.

    Extracted from former StateEmbeddingModel.configure_optimizers.
    """
    max_lr = cfg.optimizer.max_lr
    weight_decay = cfg.optimizer.weight_decay
    optimizer_name = getattr(cfg.optimizer, "name", "adamw").lower()

    if optimizer_name == "muon":
        from ...tx.optim import MuonWithAuxAdamW
        from ...tx.models.state_transition import _split_muon_parameters

        muon_params, adamw_params = _split_muon_parameters(model)

        # Move MoE router params from Muon to AdamW — the gate matrix is highly
        # non-square (d_model × num_experts) and gets extreme Muon scaling.
        from ..nn.moe import TopKRouter
        router_param_ids = {id(p) for m in model.modules() if isinstance(m, TopKRouter) for p in m.parameters()}
        if router_param_ids:
            moved = [p for p in muon_params if id(p) in router_param_ids]
            muon_params = [p for p in muon_params if id(p) not in router_param_ids]
            adamw_params.extend(moved)
        print(f"Muon: {len(muon_params)} matrix params, {len(adamw_params)} scalar/bias/norm params")
        optimizer = MuonWithAuxAdamW(
            muon_params,
            adamw_params,
            lr=max_lr,
            weight_decay=weight_decay,
            momentum=getattr(cfg.optimizer, "muon_momentum", 0.95),
            nesterov=getattr(cfg.optimizer, "muon_nesterov", True),
            ns_steps=getattr(cfg.optimizer, "muon_ns_steps", 5),
            muon_eps=getattr(cfg.optimizer, "muon_eps", 1e-7),
            adamw_betas=(
                getattr(cfg.optimizer, "muon_adamw_beta1", 0.9),
                getattr(cfg.optimizer, "muon_adamw_beta2", 0.95),
            ),
            adamw_eps=getattr(cfg.optimizer, "muon_adamw_eps", 1e-8),
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=max_lr, weight_decay=weight_decay, foreach=True)

    warmup_steps = int(getattr(cfg.optimizer, "warmup_steps", 2000))
    min_lr = float(getattr(cfg.optimizer, "min_lr", 0.0))
    schedule_steps = int(getattr(cfg.optimizer, "schedule_steps", -1))
    if schedule_steps <= 0:
        schedule_steps = total_steps
    lr_schedulers = [
        LinearLR(
            optimizer,
            start_factor=float(getattr(cfg.optimizer, "start", 0.33)),
            end_factor=1.0,
            total_iters=warmup_steps,
        ),
        CosineAnnealingLR(optimizer, eta_min=min_lr, T_max=schedule_steps - warmup_steps),
    ]
    scheduler = ChainedScheduler(lr_schedulers)

    return optimizer, scheduler


class CheckpointManager:
    """Manages checkpoint saving with last + top-k by metric."""

    def __init__(self, dirpath, save_top_k=2):
        self.dirpath = dirpath
        self.save_top_k = save_top_k
        self._best = []  # list of (metric_value, path)
        os.makedirs(dirpath, exist_ok=True)

    def _build_checkpoint(self, model, optimizer, scheduler, step, epoch, best_val_loss):
        raw_model = model.module if isinstance(model, DDP) else model
        try:
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
            )
            full_opts = StateDictOptions(full_state_dict=True)
            model_sd = get_model_state_dict(raw_model, options=full_opts)
            optim_sd = get_optimizer_state_dict(raw_model, optimizer, options=full_opts)
        except (ImportError, TypeError):
            # Fallback for non-FSDP (DDP / single-GPU)
            model_sd = raw_model.state_dict()
            optim_sd = optimizer.state_dict()
        checkpoint = {
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        raw_model.on_save_checkpoint(checkpoint)
        return checkpoint

    def save(self, model, optimizer, scheduler, step, epoch, best_val_loss,
             metric_value=None, run_name="", rank=0):
        """Save checkpoint with optional metric tracking.

        All ranks must call this (FSDP2 all-gathers in _build_checkpoint),
        but only rank 0 writes to disk.
        """
        checkpoint = self._build_checkpoint(model, optimizer, scheduler, step, epoch, best_val_loss)

        if rank != 0:
            return None

        # Always save last
        last_path = os.path.join(self.dirpath, "last.pt")
        torch.save(checkpoint, last_path)

        # Save named checkpoint if metric provided
        if metric_value is not None:
            fname = f"{run_name}-epoch={epoch}-step={step}-val_loss={metric_value:.4f}.pt"
            path = os.path.join(self.dirpath, fname)
            torch.save(checkpoint, path)

            self._best.append((metric_value, path))
            self._best.sort(key=lambda x: x[0])  # ascending — best (lowest) first

            while len(self._best) > self.save_top_k:
                _, old_path = self._best.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)

        return last_path

    def save_periodic(self, model, optimizer, scheduler, step, epoch, best_val_loss, rank=0):
        """Save periodic checkpoint (last.pt only, no metric tracking).

        All ranks must call this (FSDP2 all-gathers in _build_checkpoint),
        but only rank 0 writes to disk.
        """
        checkpoint = self._build_checkpoint(model, optimizer, scheduler, step, epoch, best_val_loss)
        if rank != 0:
            return None
        last_path = os.path.join(self.dirpath, "last.pt")
        torch.save(checkpoint, last_path)
        return last_path


def apply_fsdp2(model):
    """Apply FSDP2 partial sharding to the model.

    Shards each transformer encoder layer individually (95% of params for 7B).
    For MoE layers, also shards the MoE FFN sub-module for finer granularity.
    Also shards cross-attention blocks if present (LatentTokenizer).
    The root fully_shard handles remaining params + gradient synchronization.
    """
    from torch.distributed._composable.fsdp import fully_shard

    # Shard each transformer layer (these hold 95%+ of params)
    # MoE layers use stacked expert weights — FSDP2 shards the whole layer
    # as one unit, which is simpler and produces fewer all-gather calls.
    for layer in model.tokenizer.transformer_encoder.layers:
        fully_shard(layer)

    # Shard cross-attention blocks if present (LatentTokenizer)
    if hasattr(model.tokenizer, "cross_attn_rounds"):
        for block in model.tokenizer.cross_attn_rounds:
            fully_shard(block)

    # Root handles remaining params (encoder, decoder, cls_token, etc.)
    # and coordinates gradient synchronization
    fully_shard(model)

    return model


def run_validation(model, val_dataloader, limit_val_batches):
    """Run validation loop, return mean val loss."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if 0 < limit_val_batches <= i:
                break
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)
            val_losses.append(loss.item())
    model.train()
    return sum(val_losses) / len(val_losses) if val_losses else float("inf")


def main(cfg):
    print(f"Starting training with Embedding {cfg.embeddings.current} and dataset {cfg.dataset.current}")
    torch.set_float32_matmul_precision("high")

    # Reduce CUDA memory fragmentation for large models (4B+).
    # torch >=2.9 renamed PYTORCH_CUDA_ALLOC_CONF → PYTORCH_ALLOC_CONF.
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if key not in os.environ:
            os.environ[key] = "expandable_segments:True"

    # --- DDP setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main = local_rank == 0

    os.environ["NCCL_LAUNCH_TIMEOUT"] = str(cfg.experiment.ddp_timeout)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=cfg.experiment.ddp_timeout))
    else:
        torch.cuda.set_device(0)

    strategy = cfg.experiment.get("strategy", "ddp").lower()
    use_fsdp = strategy in ("fsdp", "fsdp2") and is_distributed
    compiled = cfg.experiment.get("compiled", False)

    # --- Build tokenizer ---
    tokenizer_type = getattr(cfg.model, "tokenizer", "sentence")
    if tokenizer_type == "latent":
        n_latent = getattr(cfg.model, "n_latent", 128)
        tokenizer = LatentTokenizer(
            n_genes=get_embedding_cfg(cfg).num,
            n_latent=n_latent,
            token_dim=get_embedding_cfg(cfg).size,
            d_model=cfg.model.emsize,
            nhead=cfg.model.nhead,
            d_hid=cfg.model.d_hid,
            nlayers=cfg.model.nlayers,
            output_dim=cfg.model.output_dim,
            dropout=cfg.model.dropout,
            compiled=False,
            cfg=cfg,
        )
        print(f"Using LatentTokenizer: n_genes={get_embedding_cfg(cfg).num}, n_latent={n_latent}")
    else:
        tokenizer = SentenceTokenizer(
            token_dim=get_embedding_cfg(cfg).size,
            d_model=cfg.model.emsize,
            nhead=cfg.model.nhead,
            d_hid=cfg.model.d_hid,
            nlayers=cfg.model.nlayers,
            output_dim=cfg.model.output_dim,
            dropout=cfg.model.dropout,
            compiled=False,
            cfg=cfg,
        )

    # --- Build collators ---
    train_collator = tokenizer.make_collator(cfg, is_train=True)
    val_collator = tokenizer.make_collator(cfg, is_train=False)

    generator = torch.Generator()
    generator.manual_seed(cfg.dataset.seed)

    if get_dataset_cfg(cfg).ds_type == "h5ad":
        DatasetClass = H5adSentenceDataset
    else:
        raise ValueError(f"Unknown dataset type: {get_dataset_cfg(cfg).ds_type}")

    # --- Dataloaders ---
    train_dataset = DatasetClass(cfg)
    val_dataset = DatasetClass(cfg, test=True)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    n_train_workers = cfg.dataset.num_train_workers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=train_collator,
        num_workers=n_train_workers,
        persistent_workers=n_train_workers > 0,
        pin_memory=True,
        prefetch_factor=4 if n_train_workers > 0 else None,
        generator=generator if train_sampler is None else None,
    )

    n_val_workers = cfg.dataset.num_val_workers
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        collate_fn=val_collator,
        num_workers=n_val_workers,
        persistent_workers=n_val_workers > 0,
        pin_memory=True,
        prefetch_factor=4 if n_val_workers > 0 else None,
        generator=generator if val_sampler is None else None,
    )

    # --- Build model ---
    model = StateEmbeddingModel(
        token_dim=get_embedding_cfg(cfg).size,
        d_model=cfg.model.emsize,
        nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        warmup_steps=0,
        compiled=False,
        max_lr=cfg.optimizer.max_lr,
        emb_size=get_embedding_cfg(cfg).size,
        collater=val_collator,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    model.update_config(cfg)
    train_dataset.cfg = cfg
    val_dataset.cfg = cfg
    train_collator.cfg = cfg
    val_collator.cfg = cfg
    model.collater = val_collator

    # Store all parameters in bf16 to halve memory footprint.
    # Without this, 7B params + grads + optimizer state = ~84 GB (exceeds H100 80 GB).
    # With bf16: ~42 GB base, leaving room for activations.
    # Autocast still handles compute precision; Muon upcasts to fp32 for Newton-Schulz.
    model = model.to(torch.bfloat16).cuda()

    # Load frozen protein embeddings (bf16)
    all_pe = get_embeddings(cfg)
    all_pe = all_pe.to(torch.bfloat16)
    all_pe.requires_grad = False
    model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    # Measure FLOPS before compile (FlopCounterMode can't trace compiled kernels).
    # For large models (>50% GPU used by params alone), FLOPS measurement runs an
    # uncompiled forward+backward that fragments memory, preventing the compiled
    # model from fitting. In that case, skip and set flops=0 (cells/sec still works).
    flops_per_batch = None
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        if n_params > 2_000_000_000:
            # Large models: measure forward-only with batch_size=1 (minimal memory),
            # then scale to actual batch_size × 3 (backward ≈ 2× forward).
            try:
                mini_batch = next(iter(train_dataloader))
                # Slice all tensor fields to batch=1 (NamedTuple uses _replace)
                sliced = {
                    f: getattr(mini_batch, f)[:1] if isinstance(getattr(mini_batch, f), torch.Tensor)
                       and getattr(mini_batch, f).dim() > 0 else getattr(mini_batch, f)
                    for f in mini_batch._fields
                }
                mini_batch = type(mini_batch)(**sliced)
                flops_1 = compute_forward_flops(model, mini_batch, use_backward=False)
                flops_per_batch = flops_1 * cfg.model.batch_size * 3
                print(f"Measured FLOPs per batch ({n_params/1e9:.1f}B model, bs=1×{cfg.model.batch_size}×3): {flops_per_batch:,}")
                del mini_batch
            except Exception as e:
                print(f"Warning: FLOPS measurement failed for {n_params/1e9:.1f}B model: {e}")
                flops_per_batch = 0
            finally:
                import gc
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        else:
            try:
                flops_batch = next(iter(train_dataloader))
                flops_per_batch = compute_forward_flops(
                    model, flops_batch,
                    use_backward=cfg.experiment.cumulative_flops_use_backward,
                )
                print(f"Measured FLOPs per batch: {flops_per_batch:,}")
                del flops_batch
            except (torch.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                try:
                    flops_batch = next(iter(train_dataloader))
                    flops_fwd = compute_forward_flops(model, flops_batch, use_backward=False)
                    flops_per_batch = flops_fwd * 3
                    print(f"Measured FLOPs per batch (fwd×3): {flops_per_batch:,}")
                    del flops_batch
                except Exception:
                    torch.cuda.empty_cache()
                    model.zero_grad(set_to_none=True)
                    print(f"Warning: FLOPS measurement failed (OOM): {e}")
                    flops_per_batch = 0
            except Exception as e:
                print(f"Warning: FLOPS measurement failed: {e}")
                flops_per_batch = 0
            finally:
                import gc
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    # Populate ESM2 cache before compile/FSDP (needs full pe_embedding)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.tokenizer._get_esm2_proj_table(model.tokenizer.pe_embedding.weight.device)

    # --- Expert Parallelism (if enabled) ---
    moe_cfg = cfg.model.get("moe", None)
    use_ep = (moe_cfg is not None and getattr(moe_cfg, "enable", False)
              and getattr(moe_cfg, "expert_parallel", False)
              and is_distributed)
    if use_ep:
        from ..nn.moe import enable_expert_parallel
        ep_group = dist.group.WORLD
        enable_expert_parallel(model, ep_group)
        n_experts = getattr(moe_cfg, "num_experts", 8)
        print(f"Expert parallelism enabled: {n_experts} experts across {world_size} GPUs")

        # Correct FLOPS for EP: the measurement above counted all E experts on rank 0,
        # but with EP each GPU only runs E/world_size experts. Scale down accordingly.
        # The expert FFN FLOPs dominate in MoE, so scale by (top_k / num_experts).
        if is_main and flops_per_batch and flops_per_batch > 0:
            top_k = getattr(moe_cfg, "top_k", 2)
            # Pre-EP FLOPs counted all E experts; active FLOPs use only top_k
            scale = top_k / n_experts
            # But attention + shared experts are fully counted, not scaled.
            # Approximation: expert FFN is ~80% of total FLOPs for MoE models.
            # active_flops ≈ total * (0.2 + 0.8 * top_k/E)
            moe_frac = 0.8  # expert FFN fraction of total FLOPs (approximate)
            ep_scale = (1.0 - moe_frac) + moe_frac * scale
            flops_per_batch = int(flops_per_batch * ep_scale)
            print(f"EP-corrected FLOPs per batch (scale={ep_scale:.3f}): {flops_per_batch:,}")

    # --- Apply distributed strategy ---
    if use_fsdp:
        model = apply_fsdp2(model)
        print(f"Applied FSDP2 sharding across {world_size} GPUs")
        if compiled:
            model.tokenizer = torch.compile(model.tokenizer)
            model._decode = torch.compile(model._decode)
            print("Compiled tokenizer and _decode with torch.compile (FSDP2)")
    else:
        # Compile then wrap with DDP
        if compiled:
            model.tokenizer = torch.compile(model.tokenizer)
            model._decode = torch.compile(model._decode)
            print("Compiled tokenizer and _decode with torch.compile")
        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

    model.train()

    # --- Config values ---
    grad_accum = cfg.optimizer.gradient_accumulation_steps
    max_grad_norm = cfg.optimizer.max_grad_norm
    ckpt_interval = cfg.experiment.checkpoint.every_n_train_steps
    val_interval = int(cfg.experiment.val_check_interval * world_size)
    limit_val_batches = cfg.experiment.limit_val_batches

    max_steps = -1
    if cfg.experiment.profile.enable_profiler:
        max_steps = cfg.experiment.profile.max_steps

    # Total optimizer steps for scheduler
    if max_steps > 0:
        total_steps = max_steps
    else:
        steps_per_epoch = max(len(train_dataloader) // grad_accum, 1)
        total_steps = steps_per_epoch * cfg.experiment.num_epochs

    # --- Build optimizer and scheduler ---
    # For DDP: model.module gives the unwrapped model.
    # For FSDP2: model IS the model (fully_shard modifies in-place, no wrapper).
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer, scheduler = build_optimizer_and_scheduler(raw_model, cfg, total_steps)

    # --- Resume from checkpoint ---
    run_name, chk = get_latest_checkpoint(cfg)
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if chk:
        print(f"******** Loading checkpoint {run_name} {chk}...")
        ckpt = torch.load(chk, map_location=f"cuda:{local_rank}", weights_only=False)
        if use_fsdp:
            try:
                from torch.distributed.checkpoint.state_dict import (
                    set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
                )
                full_opts = StateDictOptions(full_state_dict=True)
                set_model_state_dict(raw_model, ckpt["model"], options=full_opts)
                if "optimizer" in ckpt:
                    set_optimizer_state_dict(raw_model, optimizer, ckpt["optimizer"], options=full_opts)
            except (ImportError, TypeError):
                raw_model.load_state_dict(ckpt["model"], strict=False)
        else:
            raw_model.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        if cfg.optimizer.get("reset_lr_on_restart", False):
            for param_group in optimizer.param_groups:
                original_lr = param_group.get("lr", None)
                param_group["lr"] = cfg.optimizer.max_lr
                print(f"Reset learning rate from {original_lr} to {param_group['lr']}")
    else:
        print(f"******** Initialized fresh {run_name}...")

    # --- Wandb ---
    wandb_run = None
    if cfg.wandb.enable and is_main:
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.experiment.name)
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")

    # --- Checkpoint manager ---
    ckpt_dir = os.path.join(cfg.experiment.checkpoint.path, cfg.experiment.name)
    ckpt_mgr = CheckpointManager(dirpath=ckpt_dir, save_top_k=cfg.experiment.checkpoint.save_top_k)

    # --- FLOPS tracking ---
    cumulative_flops = 0
    batch_times = collections.deque(maxlen=50)

    # --- Profiling config ---
    profiling = cfg.experiment.profile.enable_profiler
    profile_steps = cfg.experiment.profile.profile_steps if profiling else [0, 0]

    # --- Training loop ---
    log_interval = 100
    microstep = global_step * grad_accum
    done = False

    # Log dataset and training stats
    batches_per_epoch = len(train_dataloader)
    steps_per_epoch_actual = batches_per_epoch // grad_accum
    if is_main:
        n_train_cells = len(train_dataloader.dataset)
        print(f"Dataset: {n_train_cells:,} cells, {batches_per_epoch:,} batches/epoch "
              f"(bs={cfg.model.batch_size}×{world_size} GPUs), "
              f"{steps_per_epoch_actual:,} optimizer steps/epoch (grad_accum={grad_accum})")

    for epoch in range(start_epoch, cfg.experiment.num_epochs):
        if done:
            break
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar_total = max_steps - global_step if max_steps > 0 else steps_per_epoch_actual
        pbar = tqdm(total=pbar_total,
                    desc=f"Epoch {epoch}", disable=not is_main, dynamic_ncols=True,
                    initial=global_step if max_steps > 0 else 0)
        accum_start = None

        for batch in train_dataloader:
            # NSys profiling start
            if profiling and microstep == profile_steps[0]:
                log.info(f"Starting NSys profiling at microstep {microstep}")
                torch.cuda.nvtx.range_push("VCIProfiledSection")

            # Track time for the accumulation window
            if microstep % grad_accum == 0:
                accum_start = time.time()

            # Skip gradient sync on non-final accumulation microsteps
            is_accum_step = (microstep + 1) % grad_accum != 0

            if use_fsdp and is_distributed:
                # FSDP2: toggle gradient sync via FSDPModule method
                model.set_requires_gradient_sync(not is_accum_step)
                sync_context = nullcontext()
            elif is_distributed and is_accum_step:
                # DDP: use no_sync context manager
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch)

                loss_scaled = loss / grad_accum
                loss_scaled.backward()
            microstep += 1

            # NSys profiling end
            if profiling and microstep == profile_steps[1]:
                log.info(f"Stopping NSys profiling at microstep {microstep}")
                torch.cuda.nvtx.range_pop()

            if microstep % grad_accum == 0:
                if not use_fsdp:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Reset MoE global-batch balance stats after each optimizer step
                from ..nn.moe import reset_moe_balance_stats
                reset_moe_balance_stats(model)

                global_step += 1

                step_time = time.time() - accum_start
                batch_times.append(step_time)

                # Track cumulative FLOPS (grad_accum forward+backward passes per step)
                if flops_per_batch and flops_per_batch > 0:
                    cumulative_flops += flops_per_batch * grad_accum * world_size

                # Compute metrics
                avg_step_time = sum(batch_times) / len(batch_times) if batch_times else 1.0
                cells_per_sec = cfg.model.batch_size * world_size * grad_accum / avg_step_time

                mfu = 0.0
                if flops_per_batch and flops_per_batch > 0:
                    flops_per_step = flops_per_batch * grad_accum
                    flops_per_sec = flops_per_step / avg_step_time
                    mfu = flops_per_sec / (H100_PEAK_TFLOPS * 1e12) * 100

                lr = scheduler.get_last_lr()[0]

                # Update progress bar (one tick per optimizer step)
                if is_main:
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr:.2e}",
                        "c/s": f"{cells_per_sec:.0f}",
                        "mfu": f"{mfu:.1f}%",
                    })

                # Wandb logging
                if is_main and global_step % log_interval == 0 and wandb_run:
                    import wandb

                    log_dict = {
                        "trainer/train_loss": loss.item(),
                        "trainer/learning_rate": lr,
                        "perf/cells_per_sec": cells_per_sec,
                        "perf/mfu": mfu,
                        "cumulative_flops": float(cumulative_flops),
                    }
                    moe_cfg = cfg.model.get("moe", None)
                    if moe_cfg is not None and getattr(moe_cfg, "enable", False):
                        from ..nn.moe import collect_moe_aux_losses
                        moe_losses = collect_moe_aux_losses(raw_model)
                        log_dict["moe/load_balance_loss"] = moe_losses["moe_load_balance"].item()
                        log_dict["moe/router_z_loss"] = moe_losses["moe_router_z"].item()
                    wandb.log(log_dict, step=global_step)

                # Validation
                if val_interval > 0 and limit_val_batches > 0 and global_step % val_interval == 0:
                    val_loss = run_validation(model, val_dataloader, limit_val_batches)

                    if is_distributed:
                        val_loss_tensor = torch.tensor(val_loss, device=f"cuda:{local_rank}")
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                        val_loss = val_loss_tensor.item()

                    if is_main:
                        print(f"\n[Step {global_step}] val_loss={val_loss:.4f} (best={best_val_loss:.4f})")
                        if wandb_run:
                            import wandb

                            wandb.log(
                                {
                                    "validation/val_loss": val_loss,
                                    "cumulative_flops": float(cumulative_flops),
                                },
                                step=global_step,
                            )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    # All ranks must call save (FSDP2 all-gathers state dicts);
                    # only rank 0 writes to disk.
                    ckpt_mgr.save(
                        model, optimizer, scheduler, global_step, epoch,
                        best_val_loss, metric_value=val_loss, run_name=run_name,
                        rank=local_rank,
                    )

                    model.train()

                # Periodic checkpoint — all ranks participate for FSDP2, rank 0 writes
                if ckpt_interval > 0 and global_step % ckpt_interval == 0:
                    ckpt_mgr.save_periodic(model, optimizer, scheduler, global_step, epoch, best_val_loss, rank=local_rank)

                # Max steps check
                if 0 < max_steps <= global_step:
                    done = True
                    break

        pbar.close()

    # --- Final summary ---
    if is_main:
        avg_step_time = sum(batch_times) / len(batch_times) if batch_times else 1.0
        cells_per_sec = cfg.model.batch_size * world_size * grad_accum / avg_step_time
        mfu = 0.0
        if flops_per_batch and flops_per_batch > 0:
            flops_per_step = flops_per_batch * grad_accum
            mfu = (flops_per_step / avg_step_time) / (H100_PEAK_TFLOPS * 1e12) * 100
        print(f"\n{'='*60}")
        print(f"Training complete: {global_step} steps")
        print(f"  cells/sec: {cells_per_sec:.0f}")
        print(f"  MFU: {mfu:.1f}%")
        print(f"  peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        print(f"{'='*60}")

    # --- Final checkpoint (all ranks gather state dicts for FSDP2, rank 0 writes) ---
    final_path = os.path.join(ckpt_dir, f"{run_name}_final.pt")
    raw_model = model.module if isinstance(model, DDP) else model
    try:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
        )
        full_opts = StateDictOptions(full_state_dict=True)
        model_sd = get_model_state_dict(raw_model, options=full_opts)
        optim_sd = get_optimizer_state_dict(raw_model, optimizer, options=full_opts)
    except (ImportError, TypeError):
        model_sd = raw_model.state_dict()
        optim_sd = optimizer.state_dict()
    checkpoint = {
        "model": model_sd,
        "optimizer": optim_sd,
        "scheduler": scheduler.state_dict(),
        "step": global_step,
        "epoch": epoch if 'epoch' in dir() else 0,
        "best_val_loss": best_val_loss,
    }
    raw_model.on_save_checkpoint(checkpoint)
    if is_main:
        torch.save(checkpoint, final_path)
        print(f"Saved final checkpoint to {final_path}")

        if wandb_run:
            import wandb

            wandb.finish()

    # --- Cleanup ---
    if is_distributed:
        dist.destroy_process_group()
