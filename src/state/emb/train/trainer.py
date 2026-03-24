import os
import torch
import lightning as L

from torch import nn
from torch.utils.data import DataLoader
from datetime import timedelta

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

from ..nn.model import StateEmbeddingModel
from ..nn.tokenizer import SentenceTokenizer, LatentTokenizer
from ..data import H5adSentenceDataset
from ..train.callbacks import (
    LogLR,
    ProfilerCallback,
    ResumeCallback,
    PerfProfilerCallback,
    CumulativeFLOPSCallback,
)
from ..utils import get_latest_checkpoint, get_embedding_cfg, get_dataset_cfg


def get_embeddings(cfg):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
    if isinstance(all_pe, dict):
        all_pe = torch.vstack(list(all_pe.values()))

    all_pe = all_pe.cuda()
    return all_pe


def main(cfg):
    print(f"Starting training with Embedding {cfg.embeddings.current} and dataset {cfg.dataset.current}")
    torch.set_float32_matmul_precision("high")
    os.environ["NCCL_LAUNCH_TIMEOUT"] = str(cfg.experiment.ddp_timeout)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    TOTAL_N_CELL = cfg.dataset.num_cells
    EPOCH_LENGTH = int(TOTAL_N_CELL // cfg.model.batch_size // 24)
    warmup_steps = EPOCH_LENGTH * 6

    # --- Build tokenizer ---
    tokenizer_type = getattr(cfg.model, "tokenizer", "sentence")
    compiled = cfg.experiment.get("compiled", False)
    # When compiled=True, we compile the full tokenizer in the trainer (after pe_embedding
    # is set) rather than individual submodules, since full-module compile gives better
    # kernel fusion (1.25x vs 1.16x in benchmarks).
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
            compiled=False,  # compile full tokenizer below instead
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

    # --- Build collators from tokenizer ---
    train_collator = tokenizer.make_collator(cfg, is_train=True)
    val_collator = tokenizer.make_collator(cfg, is_train=False)

    generator = torch.Generator()
    generator.manual_seed(cfg.dataset.seed)

    if get_dataset_cfg(cfg).ds_type == "h5ad":
        DatasetClass = H5adSentenceDataset
    else:
        raise ValueError(f"Unknown dataset type: {get_dataset_cfg(cfg).ds_type}")

    # Training dataloader
    train_dataset = DatasetClass(cfg)
    n_train_workers = cfg.dataset.num_train_workers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=n_train_workers,
        persistent_workers=n_train_workers > 0,
        pin_memory=True,
        prefetch_factor=4 if n_train_workers > 0 else None,
        generator=generator,
    )

    val_dataset = DatasetClass(cfg, test=True)
    n_val_workers = cfg.dataset.num_val_workers
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        collate_fn=val_collator,
        num_workers=n_val_workers,
        persistent_workers=n_val_workers > 0,
        pin_memory=True,
        prefetch_factor=4 if n_val_workers > 0 else None,
        generator=generator,
    )

    model = StateEmbeddingModel(
        token_dim=get_embedding_cfg(cfg).size,
        d_model=cfg.model.emsize,
        nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        warmup_steps=warmup_steps,
        compiled=False,  # compile in trainer after pe_embedding is set
        max_lr=cfg.optimizer.max_lr,
        emb_size=get_embedding_cfg(cfg).size,
        collater=val_collator,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    # Ensure model always uses the current config, even after checkpoint loading
    model.update_config(cfg)
    train_dataset.cfg = cfg
    val_dataset.cfg = cfg
    train_collator.cfg = cfg
    val_collator.cfg = cfg
    model.collater = val_collator

    strategy_name = cfg.experiment.get("strategy", "ddp").lower()
    use_fsdp = strategy_name == "fsdp"

    if use_fsdp:
        # FSDP: keep model on CPU — Lightning's FSDPStrategy handles dist init,
        # process spawning, and sharding across GPUs.
        # use_orig_params=True preserves parameter shapes so Muon sees ndim>=2 matrices.
        all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))
        all_pe = all_pe.to(torch.bfloat16)
        all_pe.requires_grad = False
        model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe)

        # Pre-split Muon params BEFORE FSDP wrapping — FSDP flattens parameters
        # to 1-D, making ndim-based splitting fail in configure_optimizers.
        # Store param names so we can match by name after FSDP.
        optimizer_name = getattr(cfg.optimizer, "name", "adamw").lower()
        if optimizer_name == "muon":
            from ...tx.models.state_transition import _split_muon_parameters
            muon_params, adamw_params = _split_muon_parameters(model)
            # Build name→group mapping
            muon_names = set()
            param_to_name = {id(p): n for n, p in model.named_parameters()}
            for p in muon_params:
                muon_names.add(param_to_name[id(p)])
            model._muon_param_names = muon_names
            print(f"Pre-FSDP Muon split: {len(muon_params)} matrix, {len(adamw_params)} adamw")

        # Skip compile — FSDP + compile requires FSDP2 composable API which
        # can't be used through Lightning's strategy (needs manual dist init).
        print(f"FSDP mode: model on CPU, pe_embedding loaded, compile skipped")
    else:
        model = model.cuda()

        # Load frozen protein embeddings onto the tokenizer (bf16 to avoid dtype conversion)
        all_pe = get_embeddings(cfg)
        all_pe = all_pe.to(torch.bfloat16)
        all_pe.requires_grad = False
        model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe)

        # Compile after pe_embedding is set so the cache can be populated first.
        # Full-module compile gives 1.25x vs 1.16x for individual submodules.
        if compiled:
            # Populate the ESM2 projection cache before compile to avoid graph breaks
            # Use autocast since pe_embedding is bf16 but encoder weights are fp32
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model.tokenizer._get_esm2_proj_table(model.tokenizer.pe_embedding.weight.device)
            model.tokenizer = torch.compile(model.tokenizer)
            model._decode = torch.compile(model._decode)
            print(f"Compiled tokenizer and _decode with torch.compile")

    model = model.train()

    run_name, chk = get_latest_checkpoint(cfg)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=cfg.experiment.checkpoint.every_n_train_steps,
        dirpath=os.path.join(cfg.experiment.checkpoint.path, cfg.experiment.name),
        filename=f"{run_name}" + "-{epoch}-{step}",
        save_last=True,
        save_top_k=cfg.experiment.checkpoint.save_top_k,
        monitor=cfg.experiment.checkpoint.monitor,
    )

    if cfg.wandb.enable:
        try:
            import wandb

            exp_logger = WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.experiment.name)
            exp_logger.watch(model, log_freq=1000)
        except ImportError:
            print("Warning: wandb is not installed. Skipping wandb logging.")
            print("To enable wandb logging, install it with: pip install wandb")
            exp_logger = None
        except Exception as e:
            print(f"Warning: Failed to initialize wandb logger: {e}")
            print("Continuing without wandb logging.")
            exp_logger = None
    else:
        exp_logger = None

    callbacks = [checkpoint_callback, LogLR(100), ResumeCallback(cfg), PerfProfilerCallback()]

    # Add cumulative FLOPS callback
    callbacks.append(CumulativeFLOPSCallback(use_backward=cfg.experiment.cumulative_flops_use_backward))

    max_steps = -1
    if cfg.experiment.profile.enable_profiler:
        callbacks.append(ProfilerCallback(cfg=cfg))
        max_steps = cfg.experiment.profile.max_steps

    val_interval = int(cfg.experiment.val_check_interval * cfg.experiment.num_gpus_per_node * cfg.experiment.num_nodes)

    if use_fsdp:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from ..nn.flash_transformer import FlashTransformerEncoderLayer
        from ..nn.tokenizer import CrossAttentionBlock

        # Detach frozen pe_embedding (bf16) before FSDP wrapping — mixed dtype
        # in the same FSDP unit causes ValueError. Re-attach after setup.
        _pe_embedding = model.tokenizer.pe_embedding
        model.tokenizer.pe_embedding = None

        class _ReattachPECallback(L.Callback):
            """Re-attach pe_embedding after FSDP wraps the model."""
            def __init__(self, pe_emb):
                self._pe_emb = pe_emb
            def on_fit_start(self, trainer, pl_module):
                device = next(pl_module.parameters()).device
                pl_module.tokenizer.pe_embedding = self._pe_emb.to(device)
                print(f"Re-attached pe_embedding on {device}")

        callbacks.append(_ReattachPECallback(_pe_embedding))

        strategy = FSDPStrategy(
            auto_wrap_policy=ModuleWrapPolicy({FlashTransformerEncoderLayer, CrossAttentionBlock}),
            sharding_strategy="FULL_SHARD",
            process_group_backend="nccl",
            timeout=timedelta(seconds=cfg.experiment.get("ddp_timeout", 3600)),
            use_orig_params=True,  # preserve param shapes for Muon optimizer
        )
        print(f"Using FSDP (FULL_SHARD, use_orig_params=True)")
    else:
        strategy = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            timeout=timedelta(seconds=cfg.experiment.get("ddp_timeout", 3600)),
        )

    trainer = L.Trainer(
        max_epochs=cfg.experiment.num_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        devices=cfg.experiment.num_gpus_per_node,
        num_nodes=cfg.experiment.num_nodes,
        # Accumulation
        gradient_clip_val=cfg.optimizer.max_grad_norm,
        accumulate_grad_batches=cfg.optimizer.gradient_accumulation_steps,
        precision="bf16-mixed",
        strategy=strategy,
        val_check_interval=val_interval,
        # Logging
        logger=exp_logger,
        fast_dev_run=False,
        limit_val_batches=cfg.experiment.limit_val_batches,
    )

    if chk:
        print(f"******** Loading chkpoint {run_name} {chk}...")
    else:
        print(f"******** Initialized fresh {run_name}...")

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=chk)

    trainer.save_checkpoint(os.path.join(cfg.experiment.checkpoint.path, f"{run_name}_final.pt"))
