#!/usr/bin/env python3
"""
Diagnostic script to quantify the representation gap between TX model outputs
and SE cell embeddings.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_representation_gap.py \
        --tx-run /data/replogle_llm/tx_runs/k562_state_emb \
        --se-checkpoint /home/aadduri/SE-600M/se600m_epoch16.ckpt \
        --se-config /home/aadduri/SE-600M/config.yaml
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Allow old checkpoints with OmegaConf objects to load
import omegaconf
torch.serialization.add_safe_globals(
    [omegaconf.dictconfig.DictConfig, omegaconf.listconfig.ListConfig]
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_tx_model(run_dir, device="cuda"):
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    run_dir = Path(run_dir)
    config = OmegaConf.load(run_dir / "config.yaml")
    ckpt_path = run_dir / "checkpoints" / "best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"

    var_dims = pickle.load(open(run_dir / "var_dims.pkl", "rb"))
    model_kwargs = OmegaConf.to_container(config.model.kwargs, resolve=True)
    model_kwargs["input_dim"] = int(var_dims["input_dim"])
    model_kwargs["output_dim"] = int(var_dims["output_dim"])
    model_kwargs["pert_dim"] = int(var_dims["pert_dim"])
    model_kwargs["gene_dim"] = int(var_dims.get("gene_dim", var_dims["output_dim"]))
    model_kwargs["hvg_dim"] = int(var_dims.get("hvg_dim", 2001))
    model_kwargs["embed_key"] = config.data.kwargs.embed_key
    model_kwargs["output_space"] = config.data.kwargs.output_space
    model_kwargs["gene_names"] = var_dims.get("gene_names", None)
    model_kwargs["basal_mapping_strategy"] = config.data.kwargs.get("basal_mapping_strategy", "random")

    if config.model.get("decoder_cfg"):
        model_kwargs["decoder_cfg"] = OmegaConf.to_container(config.model.decoder_cfg, resolve=True)

    model = StateTransitionPerturbationModel.load_from_checkpoint(
        str(ckpt_path), strict=False, **model_kwargs
    )
    model.eval().to(device)
    return model, config, var_dims


def load_se_model(checkpoint, config_path, device="cuda"):
    from state.emb.nn.model import StateEmbeddingModel
    from state.emb.finetune_decoder import Finetune

    cfg = OmegaConf.load(config_path)

    # Load checkpoint manually (avoid weights_only issues)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
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

    # Load protein embeddings
    if "protein_embeds_dict" in ckpt:
        protein_embeds = ckpt["protein_embeds_dict"]
    else:
        # Try config path, then fallback to SE directory
        pe_path = cfg.embeddings[cfg.embeddings.current].all_embeddings
        if not Path(pe_path).exists():
            pe_path = str(Path(checkpoint).parent / "protein_embeddings.pt")
        protein_embeds = torch.load(pe_path, weights_only=False)
    model.protein_embeds = protein_embeds

    # Set up pe_embedding
    stacked = torch.vstack(list(protein_embeds.values()))
    stacked.requires_grad = False
    import torch.nn as nn
    model.pe_embedding = nn.Embedding.from_pretrained(stacked)

    # Create Finetune wrapper for gene embedding access
    ft = Finetune.__new__(Finetune)
    nn.Module.__init__(ft)
    ft.model = model
    ft.protein_embeds = protein_embeds
    ft._vci_conf = cfg
    ft.device = torch.device(device)
    ft.train_binary_decoder = False
    ft.read_depth = nn.Parameter(torch.tensor(4.0))
    ft.cached_gene_embeddings = {}
    ft.missing_table = None
    ft._last_missing_count = 0
    ft._last_missing_dim = 0
    ft._present_mask_cache = {}
    ft._missing_index_map_cache = {}
    ft.to(device)

    return ft


def load_data(config, run_dir):
    from cell_load.data_modules import PerturbationDataModule
    from cell_load.utils.modules import get_datamodule

    run_dir = Path(run_dir)
    data_kwargs = OmegaConf.to_container(config.data.kwargs, resolve=True)
    sentence_len = config.model.kwargs.cell_set_len

    dm = get_datamodule(
        config.data.name,
        data_kwargs,
        batch_size=config.training.get("batch_size", 16),
        cell_sentence_len=sentence_len,
    )
    dm.setup(stage="test")

    # Restore maps
    dm.pert_onehot_map = torch.load(run_dir / "pert_onehot_map.pt", weights_only=False)
    if (run_dir / "batch_onehot_map.torch").exists():
        dm.batch_onehot_map = torch.load(run_dir / "batch_onehot_map.torch", weights_only=False)
    if (run_dir / "cell_type_onehot_map.torch").exists():
        dm.cell_type_onehot_map = torch.load(run_dir / "cell_type_onehot_map.torch", weights_only=False)

    return dm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx-run", default="/data/replogle_llm/tx_runs/k562_state_emb")
    parser.add_argument("--se-checkpoint", default="/home/aadduri/SE-600M/se600m_epoch16.ckpt")
    parser.add_argument("--se-config", default="/home/aadduri/SE-600M/config.yaml")
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device

    print("Loading TX model...")
    tx_model, config, var_dims = load_tx_model(args.tx_run, device)
    print(f"  TX output_dim={tx_model.output_dim}, input_dim={tx_model.input_dim}")

    print("Loading SE model...")
    se_ft = load_se_model(args.se_checkpoint, args.se_config, device)
    se_model = se_ft.model
    se_output_dim = se_model.hparams["output_dim"]
    se_d_model = se_model.d_model
    se_z_dim = se_model.z_dim
    print(f"  SE output_dim={se_output_dim}, d_model={se_d_model}, z_dim={se_z_dim}")
    print(f"  Binary decoder input dim: {se_output_dim + se_d_model + se_z_dim}")

    print("Loading data...")
    dm = load_data(config, args.tx_run)
    test_dl = dm.test_dataloader()

    # Collect statistics
    all_tx = []
    all_se = []

    print(f"\nProcessing {args.n_batches} test batches...")
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            if i >= args.n_batches:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            tx_out = tx_model(batch, padded=False)
            se_embs = batch["pert_cell_emb"]
            all_tx.append(tx_out.cpu())
            all_se.append(se_embs.cpu())
            print(f"  Batch {i}: {tx_out.shape[0]} cells")

    all_tx = torch.cat(all_tx, dim=0).float()
    all_se = torch.cat(all_se, dim=0).float()
    n_cells = all_tx.shape[0]
    print(f"\nTotal cells: {n_cells}")

    cell_dim = se_output_dim  # 2048
    ds_dim = se_model.z_dim_ds  # 10

    # Split into cell embedding and dataset embedding parts
    tx_cell = all_tx[:, :cell_dim]
    tx_ds = all_tx[:, cell_dim:cell_dim + ds_dim]
    se_cell = all_se[:, :cell_dim]
    se_ds = all_se[:, cell_dim:cell_dim + ds_dim]

    print("\n" + "=" * 80)
    print("REPRESENTATION GAP ANALYSIS")
    print("=" * 80)

    # 1. L2 Norms
    tx_cell_norms = tx_cell.norm(dim=-1)
    se_cell_norms = se_cell.norm(dim=-1)
    tx_ds_norms = tx_ds.norm(dim=-1)
    se_ds_norms = se_ds.norm(dim=-1)

    print(f"\n1. L2 Norms (cell embedding, first {cell_dim} dims):")
    print(f"   TX:  mean={tx_cell_norms.mean():.4f}  std={tx_cell_norms.std():.4f}  "
          f"range=[{tx_cell_norms.min():.4f}, {tx_cell_norms.max():.4f}]")
    print(f"   SE:  mean={se_cell_norms.mean():.4f}  std={se_cell_norms.std():.4f}  "
          f"range=[{se_cell_norms.min():.4f}, {se_cell_norms.max():.4f}]")
    print(f"   SE cell embeddings are L2-normalized: {se_cell_norms.mean():.6f} ≈ 1.0")

    print(f"\n   Dataset embedding (last {ds_dim} dims) norms:")
    print(f"   TX:  mean={tx_ds_norms.mean():.4f}  std={tx_ds_norms.std():.4f}")
    print(f"   SE:  mean={se_ds_norms.mean():.4f}  std={se_ds_norms.std():.4f}")

    # 2. Cosine similarities
    cos_raw = F.cosine_similarity(tx_cell, se_cell, dim=-1)
    tx_cell_normed = F.normalize(tx_cell, dim=-1)
    se_cell_normed = F.normalize(se_cell, dim=-1)  # should be identity
    cos_norm = F.cosine_similarity(tx_cell_normed, se_cell_normed, dim=-1)

    print(f"\n2. Cosine Similarity (TX cell emb vs SE cell emb):")
    print(f"   Raw:         mean={cos_raw.mean():.4f}  std={cos_raw.std():.4f}")
    print(f"   Both normed: mean={cos_norm.mean():.4f}  std={cos_norm.std():.4f}")

    # 3. MSE analysis
    mse_raw = F.mse_loss(all_tx, all_se).item()
    mse_cell = F.mse_loss(tx_cell, se_cell).item()
    mse_cell_normed = F.mse_loss(tx_cell_normed, se_cell_normed).item()
    mse_ds = F.mse_loss(tx_ds, se_ds).item()

    print(f"\n3. MSE (TX vs SE):")
    print(f"   Full (2058 dim):      {mse_raw:.6f}")
    print(f"   Cell part (2048 dim): {mse_cell:.6f}")
    print(f"   Cell normed:          {mse_cell_normed:.6f}")
    print(f"   DS part (10 dim):     {mse_ds:.6f}")

    # 4. Per-dimension statistics
    print(f"\n4. Per-dimension statistics:")
    tx_mean = tx_cell.mean(dim=0)
    se_mean = se_cell.mean(dim=0)
    tx_std = tx_cell.std(dim=0)
    se_std = se_cell.std(dim=0)
    print(f"   TX cell mean: range=[{tx_mean.min():.6f}, {tx_mean.max():.6f}], abs_mean={tx_mean.abs().mean():.6f}")
    print(f"   SE cell mean: range=[{se_mean.min():.6f}, {se_mean.max():.6f}], abs_mean={se_mean.abs().mean():.6f}")
    print(f"   TX cell std:  range=[{tx_std.min():.6f}, {tx_std.max():.6f}], mean={tx_std.mean():.6f}")
    print(f"   SE cell std:  range=[{se_std.min():.6f}, {se_std.max():.6f}], mean={se_std.mean():.6f}")

    # 5. Binary decoder comparison
    print(f"\n5. Binary Decoder Output Comparison:")
    # Use first 100 genes and 20 cells
    genes = list(se_ft.protein_embeds.keys())[:100]
    gene_embeds = se_ft.get_gene_embedding(genes).to(device)  # [100, d_model]

    n_test = min(20, n_cells)
    se_cell_dev = se_cell[:n_test].to(device)
    tx_cell_dev = tx_cell[:n_test].to(device)
    tx_cell_norm_dev = tx_cell_normed[:n_test].to(device)
    se_ds_dev = se_ds[:n_test].to(device)
    tx_ds_dev = tx_ds[:n_test].to(device)

    rda = se_ft.read_depth.expand(n_test).to(device)

    from state.emb.nn.model import StateEmbeddingModel

    with torch.no_grad():
        # With real SE cell embeddings
        merged_se = StateEmbeddingModel.resize_batch(se_cell_dev, gene_embeds, task_counts=rda, ds_emb=se_ds_dev)
        out_se = se_model.binary_decoder(merged_se).squeeze(-1)

        # With raw TX cell embeddings
        merged_tx_raw = StateEmbeddingModel.resize_batch(tx_cell_dev, gene_embeds, task_counts=rda, ds_emb=tx_ds_dev)
        out_tx_raw = se_model.binary_decoder(merged_tx_raw).squeeze(-1)

        # With L2-normalized TX cell embeddings
        merged_tx_norm = StateEmbeddingModel.resize_batch(tx_cell_norm_dev, gene_embeds, task_counts=rda, ds_emb=tx_ds_dev)
        out_tx_norm = se_model.binary_decoder(merged_tx_norm).squeeze(-1)

    print(f"   With real SE cell embs:    mean={out_se.mean():.4f}  std={out_se.std():.4f}  range=[{out_se.min():.2f}, {out_se.max():.2f}]")
    print(f"   With raw TX cell embs:     mean={out_tx_raw.mean():.4f}  std={out_tx_raw.std():.4f}  range=[{out_tx_raw.min():.2f}, {out_tx_raw.max():.2f}]")
    print(f"   With L2-normed TX cell:    mean={out_tx_norm.mean():.4f}  std={out_tx_norm.std():.4f}  range=[{out_tx_norm.min():.2f}, {out_tx_norm.max():.2f}]")

    # Correlation
    corr_raw = torch.corrcoef(torch.stack([out_se.flatten(), out_tx_raw.flatten()]))[0, 1].item()
    corr_norm = torch.corrcoef(torch.stack([out_se.flatten(), out_tx_norm.flatten()]))[0, 1].item()
    print(f"\n   Pearson correlation (SE-based vs TX-based predictions across 100 genes x 20 cells):")
    print(f"   Raw TX:    r={corr_raw:.4f}")
    print(f"   Normed TX: r={corr_norm:.4f}")

    # Per-cell correlation (how well does binary decoder rank genes for a given cell?)
    per_cell_corrs_raw = []
    per_cell_corrs_norm = []
    for c in range(n_test):
        r_raw = torch.corrcoef(torch.stack([out_se[c], out_tx_raw[c]]))[0, 1].item()
        r_norm = torch.corrcoef(torch.stack([out_se[c], out_tx_norm[c]]))[0, 1].item()
        per_cell_corrs_raw.append(r_raw)
        per_cell_corrs_norm.append(r_norm)

    print(f"\n   Per-cell gene-ranking correlation (mean across {n_test} cells):")
    print(f"   Raw TX:    r={np.mean(per_cell_corrs_raw):.4f}")
    print(f"   Normed TX: r={np.mean(per_cell_corrs_norm):.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Key findings:
- SE cell embeddings are L2-normalized (norm ≈ {se_cell_norms.mean():.4f})
- TX cell embeddings have norm ≈ {tx_cell_norms.mean():.4f} (ratio: {tx_cell_norms.mean()/se_cell_norms.mean():.2f}x)
- Raw cosine similarity: {cos_raw.mean():.4f}
- The binary decoder was trained on L2-normalized cell embeddings
- L2-normalizing TX outputs {'significantly' if corr_norm > corr_raw + 0.05 else 'modestly'} improves decoder correlation: {corr_raw:.4f} -> {corr_norm:.4f}
""")


if __name__ == "__main__":
    main()
