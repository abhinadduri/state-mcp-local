#!/usr/bin/env python3
"""
Knowledge distillation: Train a LatentToGeneDecoder MLP to mimic the SE binary decoder.

This script:
1. Loads SE embeddings from the dataset (real cell embeddings)
2. Runs the SE binary decoder on each cell to get gene expression predictions
3. Trains a LatentToGeneDecoder MLP to match these predictions
4. Saves the pretrained decoder weights for use as initialization in TX training

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/distill_binary_decoder.py \
        --se-checkpoint /home/aadduri/SE-600M/se600m_epoch16.ckpt \
        --se-config /home/aadduri/SE-600M/config.yaml \
        --data-path /data/replogle_llm/replogle_concat_with_llm_claude.h5ad \
        --output /data/replogle_llm/distilled_decoder.pt \
        --n-cells 50000 --epochs 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_se_and_embeddings(se_checkpoint, se_config, data_path, n_cells, gene_col, device):
    """Load SE model and extract embeddings + binary decoder outputs for training cells."""
    import anndata as ad
    from omegaconf import OmegaConf
    from state.emb.nn.model import StateEmbeddingModel

    print("Loading SE checkpoint...")
    cfg = OmegaConf.load(se_config)
    ckpt = torch.load(se_checkpoint, map_location="cpu", weights_only=False)

    model = StateEmbeddingModel(
        token_dim=cfg.tokenizer.token_dim,
        d_model=cfg.model.emsize,
        nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim,
        dropout=0.0,
        cfg=cfg,
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)

    # Protein embeddings
    if "protein_embeds_dict" in ckpt:
        protein_embeds = ckpt["protein_embeds_dict"]
    else:
        pe_path = str(Path(se_checkpoint).parent / "protein_embeddings.pt")
        protein_embeds = torch.load(pe_path, weights_only=False)

    print("Loading dataset...")
    adata = ad.read_h5ad(data_path, backed="r")

    # Get gene names
    if gene_col and gene_col in adata.var.columns:
        gene_names = list(adata.var[gene_col].astype(str))
    else:
        gene_names = list(adata.var_names)

    # Get SE embeddings (X_state)
    if "X_state" in adata.obsm:
        all_embs = adata.obsm["X_state"]
        if hasattr(all_embs, "toarray"):
            all_embs = all_embs.toarray()
        # Sample
        n_total = all_embs.shape[0]
        indices = np.random.choice(n_total, size=min(n_cells, n_total), replace=False)
        indices.sort()
        embs = torch.tensor(all_embs[indices], dtype=torch.float32)
    else:
        raise ValueError("Dataset does not have X_state in obsm")

    print(f"  Loaded {embs.shape[0]} cell embeddings of dim {embs.shape[1]}")
    print(f"  Gene count: {len(gene_names)}")

    # Build gene embeddings
    embed_size = next(iter(protein_embeds.values())).shape[-1]
    raw_list = []
    for g in gene_names:
        if g in protein_embeds:
            raw_list.append(protein_embeds[g])
        else:
            raw_list.append(torch.zeros(embed_size))

    raw_tensor = torch.stack(raw_list).to(device)
    with torch.no_grad():
        gene_embeds = model.gene_embedding_layer(raw_tensor)  # [G, d_model]

    n_matched = sum(1 for g in gene_names if g in protein_embeds)
    print(f"  Matched genes: {n_matched}/{len(gene_names)}")

    return model, embs, gene_embeds, gene_names


def compute_binary_decoder_targets(se_model, cell_embs, gene_embeds, device, batch_size=8):
    """Run the binary decoder to generate target gene expression for all cells."""
    n_cells = cell_embs.shape[0]
    n_genes = gene_embeds.shape[0]
    cell_dim = 2048
    ds_dim = cell_embs.shape[1] - cell_dim

    all_targets = []

    print(f"Computing binary decoder targets for {n_cells} cells x {n_genes} genes...")
    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            end = min(i + batch_size, n_cells)
            cell_batch = cell_embs[i:end].to(device)

            cell_part = cell_batch[:, :cell_dim]
            ds_part = cell_batch[:, cell_dim:] if ds_dim > 0 else None

            bs = cell_part.shape[0]
            rda = torch.tensor(4.0, device=device).expand(bs)

            # Build pairwise features
            A = gene_embeds.unsqueeze(0).expand(bs, -1, -1)
            B = cell_part.unsqueeze(1).expand(-1, n_genes, -1)
            rda_exp = rda.unsqueeze(1).unsqueeze(2).expand(-1, n_genes, 1)
            combined = torch.cat([A, B, rda_exp], dim=2)
            if ds_part is not None:
                ds_exp = ds_part.unsqueeze(1).expand(-1, n_genes, -1)
                combined = torch.cat([combined, ds_exp], dim=2)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = se_model.binary_decoder(combined).squeeze(-1)  # [bs, G]

            all_targets.append(logits.float().cpu())

            if (i // batch_size) % 100 == 0:
                print(f"  Processed {end}/{n_cells} cells")

    return torch.cat(all_targets, dim=0)


def train_distilled_decoder(cell_embs, targets, latent_dim, gene_dim, hidden_dims, epochs, lr, device):
    """Train a LatentToGeneDecoder MLP to match binary decoder outputs."""
    from state.tx.models.base import LatentToGeneDecoder

    decoder = LatentToGeneDecoder(
        latent_dim=latent_dim,
        gene_dim=gene_dim,
        hidden_dims=hidden_dims,
        dropout=0.1,
    ).to(device)

    # Remove ReLU from the end since targets may have negative logits
    # We'll train on logits and add ReLU back for final usage
    # Actually LatentToGeneDecoder has ReLU at the end — let's train with ReLU targets
    targets_relu = torch.relu(targets)

    dataset = TensorDataset(cell_embs, targets_relu)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTraining distilled decoder: {latent_dim} -> {hidden_dims} -> {gene_dim}")
    print(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        decoder.train()
        total_loss = 0
        n_batches = 0

        for emb_batch, target_batch in loader:
            emb_batch = emb_batch.to(device)
            target_batch = target_batch.to(device)

            pred = decoder(emb_batch)
            loss = F.mse_loss(pred, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            # Compute correlation on a sample
            decoder.eval()
            with torch.no_grad():
                sample_emb = cell_embs[:100].to(device)
                sample_target = targets_relu[:100].to(device)
                sample_pred = decoder(sample_emb)
                corr = torch.corrcoef(
                    torch.stack([sample_pred.flatten(), sample_target.flatten()])
                )[0, 1].item()
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.6f}, best={best_loss:.6f}, corr={corr:.4f}")

    return best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--se-checkpoint", default="/home/aadduri/SE-600M/se600m_epoch16.ckpt")
    parser.add_argument("--se-config", default="/home/aadduri/SE-600M/config.yaml")
    parser.add_argument("--data-path", default="/data/replogle_llm/replogle_concat_with_llm_claude.h5ad")
    parser.add_argument("--output", default="/data/replogle_llm/distilled_decoder.pt")
    parser.add_argument("--n-cells", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 1024, 512])
    parser.add_argument("--gene-col", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    se_model, cell_embs, gene_embeds, gene_names = load_se_and_embeddings(
        args.se_checkpoint, args.se_config, args.data_path, args.n_cells, args.gene_col, args.device
    )

    targets = compute_binary_decoder_targets(se_model, cell_embs, gene_embeds, args.device)

    # Free SE model memory
    del se_model
    torch.cuda.empty_cache()

    latent_dim = cell_embs.shape[1]
    gene_dim = len(gene_names)

    best_state = train_distilled_decoder(
        cell_embs, targets, latent_dim, gene_dim, args.hidden_dims, args.epochs, args.lr, args.device
    )

    # Save with metadata
    output = {
        "state_dict": best_state,
        "latent_dim": latent_dim,
        "gene_dim": gene_dim,
        "hidden_dims": args.hidden_dims,
        "gene_names": gene_names,
        "n_training_cells": cell_embs.shape[0],
    }
    torch.save(output, args.output)
    print(f"\nSaved distilled decoder to {args.output}")


if __name__ == "__main__":
    main()
