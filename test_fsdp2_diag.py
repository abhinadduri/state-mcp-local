"""Minimal FSDP2 diagnostic: isolate forward/backward from optimizer."""
import os
import time
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build model (same as trainer)
    import sys
    sys.path.insert(0, "/home/aadduri/state-mcp-local")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir="/home/aadduri/state-mcp-local/src/state/configs", version_base=None):
        cfg = compose(config_name="state-defaults", overrides=["+scale=7b", "model.batch_size=16"])

    from src.state.emb.nn.model import StateEmbeddingModel
    from src.state.emb.nn.tokenizer import LatentTokenizer
    from src.state.emb.train.trainer import get_embedding_cfg
    import torch.nn as nn

    n_latent = getattr(cfg.model, "n_latent", 128)
    tokenizer = LatentTokenizer(
        n_genes=get_embedding_cfg(cfg).num, n_latent=n_latent,
        token_dim=get_embedding_cfg(cfg).size, d_model=cfg.model.emsize,
        nhead=cfg.model.nhead, d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers, output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout, compiled=False, cfg=cfg,
    )
    model = StateEmbeddingModel(
        token_dim=get_embedding_cfg(cfg).size,
        d_model=cfg.model.emsize, nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid, nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim, dropout=cfg.model.dropout,
        warmup_steps=0, compiled=False, max_lr=cfg.optimizer.max_lr,
        emb_size=get_embedding_cfg(cfg).size, collater=None, cfg=cfg,
        tokenizer=tokenizer,
    )
    model = model.to(torch.bfloat16).cuda()

    # Load PE embeddings
    from src.state.emb.train.trainer import get_embeddings
    all_pe = get_embeddings(cfg).to(torch.bfloat16)
    all_pe.requires_grad = False
    model.tokenizer.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    # Populate ESM2 cache
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.tokenizer._get_esm2_proj_table(model.tokenizer.pe_embedding.weight.device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params/1e9:.1f}B params, {torch.cuda.memory_allocated()/1e9:.1f} GB before FSDP")

    # Apply FSDP2
    for layer in model.tokenizer.transformer_encoder.layers:
        fully_shard(layer)
    if hasattr(model.tokenizer, "cross_attn_rounds"):
        for block in model.tokenizer.cross_attn_rounds:
            fully_shard(block)
    fully_shard(model)

    if rank == 0:
        print(f"After FSDP2: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Get a batch from dataloader
    from src.state.emb.train.trainer import build_datasets
    train_ds, val_ds, train_collator, val_collator, _ = build_datasets(cfg)
    model.collater = val_collator
    model.update_config(cfg)
    train_ds.cfg = cfg
    train_collator.cfg = cfg

    from torch.utils.data import DataLoader, DistributedSampler
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=16, sampler=sampler, collate_fn=train_collator,
                       num_workers=2, pin_memory=True)
    batch = next(iter(loader))

    if rank == 0:
        print(f"Batch loaded, {torch.cuda.memory_allocated()/1e9:.1f} GB")

    model.train()
    dist.barrier()

    # --- Phase 1: Forward only ---
    if rank == 0:
        print("\n--- Phase 1: Forward only ---", flush=True)
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        torch.cuda.synchronize()
        t1 = time.time()
        if rank == 0:
            print(f"  Forward {i}: {t1-t0:.3f}s  loss={loss.item():.4f}  mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
        loss.detach()
        model.zero_grad(set_to_none=True)

    # --- Phase 2: Forward + Backward ---
    if rank == 0:
        print("\n--- Phase 2: Forward + Backward ---", flush=True)
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        torch.cuda.synchronize()
        t1 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.time()
        if rank == 0:
            print(f"  Fwd {i}: {t1-t0:.3f}s  Bwd: {t2-t1:.3f}s  total={t2-t0:.3f}s  mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
        model.zero_grad(set_to_none=True)

    # --- Phase 3: Forward + Backward + clip_grad_norm ---
    if rank == 0:
        print("\n--- Phase 3: + clip_grad_norm ---", flush=True)
    for i in range(2):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize()
        t2 = time.time()
        if rank == 0:
            print(f"  Fwd+Bwd {i}: {t1-t0:.3f}s  clip: {t2-t1:.3f}s", flush=True)
        model.zero_grad(set_to_none=True)

    # --- Phase 4: + Muon optimizer step ---
    if rank == 0:
        print("\n--- Phase 4: + Muon optimizer step ---", flush=True)
    from src.state.tx.optim import MuonWithAuxAdamW
    from src.state.tx.models.state_transition import _split_muon_parameters
    muon_params, adamw_params = _split_muon_parameters(model)
    optimizer = MuonWithAuxAdamW(
        muon_params, adamw_params,
        lr=cfg.optimizer.max_lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    for i in range(2):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize()
        t2 = time.time()
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.time()
        optimizer.zero_grad()
        if rank == 0:
            print(f"  Fwd+Bwd {i}: {t1-t0:.3f}s  clip: {t2-t1:.3f}s  optim: {t3-t2:.3f}s  total: {t3-t0:.3f}s", flush=True)

    if rank == 0:
        print(f"\nPeak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        print("Done!", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
