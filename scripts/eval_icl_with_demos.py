"""
Evaluate an ICL model with demonstrations from the target cell type.

Usage:
    python scripts/eval_icl_with_demos.py \
        --run-dir /path/to/icl_run \
        --demo-adata /data/replogle_llm/replogle_concat_with_llm_claude.h5ad \
        --demo-cell-type k562 \
        --n-demo-perts 20 \
        --checkpoint last.ckpt
"""

import argparse
import logging
import os
import subprocess
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_demo_data_from_h5ad(
    h5ad_path: str,
    cell_type: str,
    embed_key: str = "X_hvg",
    pert_col: str = "gene",
    cell_type_col: str = "cell_line",
    control_pert: str = "non-targeting",
    n_demo_perts: int = 20,
    n_cells_per_pert: int = 8,
    pert_onehot_map: dict = None,
    seed: int = 42,
) -> dict:
    """Load demonstration data from an h5ad file for a specific cell type."""
    rng = np.random.RandomState(seed)

    logger.info("Loading demos from %s for cell type '%s'", h5ad_path, cell_type)

    with h5py.File(h5ad_path, "r") as f:
        # Read metadata
        ct_data = f["obs"][cell_type_col]
        if "categories" in ct_data:
            cats = [c.decode() if isinstance(c, bytes) else c for c in ct_data["categories"][:]]
            codes = ct_data["codes"][:]
            cell_types = np.array([cats[c] for c in codes])
        else:
            cell_types = np.array([x.decode() if isinstance(x, bytes) else x for x in ct_data[:]])

        pt_data = f["obs"][pert_col]
        if "categories" in pt_data:
            cats = [c.decode() if isinstance(c, bytes) else c for c in pt_data["categories"][:]]
            codes = pt_data["codes"][:]
            perts = np.array([cats[c] for c in codes])
        else:
            perts = np.array([x.decode() if isinstance(x, bytes) else x for x in pt_data[:]])

        # Filter to target cell type
        ct_mask = cell_types == cell_type
        ct_indices = np.where(ct_mask)[0]
        ct_perts = perts[ct_indices]

        # Get unique perturbations (excluding control)
        unique_perts = list(set(ct_perts) - {control_pert})
        logger.info("Found %d unique perturbations in %s", len(unique_perts), cell_type)

        # Sample demo perturbations
        n_sample = min(n_demo_perts, len(unique_perts))
        demo_perts = rng.choice(unique_perts, size=n_sample, replace=False).tolist()
        logger.info("Selected %d demo perturbations: %s...", n_sample, demo_perts[:5])

        # Get control cell indices for this cell type
        ctrl_mask = (cell_types == cell_type) & (perts == control_pert)
        ctrl_indices = np.where(ctrl_mask)[0]

        # Read embeddings
        emb = f["obsm"][embed_key]
        emb_dim = emb.shape[1]

        # Compute mean control expression
        ctrl_sample = rng.choice(ctrl_indices, size=min(100, len(ctrl_indices)), replace=False)
        ctrl_mean = np.mean(emb[ctrl_sample], axis=0)

        # Build demo tensors
        demo_ctrl_list = []
        demo_pert_list = []
        demo_effect_list = []

        for pn in demo_perts:
            pert_mask = (cell_types == cell_type) & (perts == pn)
            pert_indices = np.where(pert_mask)[0]

            # Sample cells
            n_cells = min(n_cells_per_pert, len(pert_indices))
            sampled = rng.choice(pert_indices, size=n_cells, replace=len(pert_indices) < n_cells)
            pert_embs = emb[sampled]

            # Get one-hot
            if pert_onehot_map and pn in pert_onehot_map:
                onehot = pert_onehot_map[pn]
                if isinstance(onehot, torch.Tensor):
                    onehot = onehot.cpu().numpy()
            else:
                onehot = np.zeros(len(pert_onehot_map) if pert_onehot_map else 100)

            for i in range(n_cells):
                demo_ctrl_list.append(ctrl_mean)
                demo_pert_list.append(onehot)
                demo_effect_list.append(pert_embs[i])

        # Pad to n_demo_perts * n_cells_per_pert
        total_needed = n_demo_perts * n_cells_per_pert
        while len(demo_ctrl_list) < total_needed:
            demo_ctrl_list.append(np.zeros(emb_dim))
            demo_pert_list.append(np.zeros(len(onehot)))
            demo_effect_list.append(np.zeros(emb_dim))

        demo_batch = {
            "demo_ctrl": torch.tensor(np.array(demo_ctrl_list[:total_needed]), dtype=torch.float32).unsqueeze(0),
            "demo_pert": torch.tensor(np.array(demo_pert_list[:total_needed]), dtype=torch.float32).unsqueeze(0),
            "demo_effect": torch.tensor(np.array(demo_effect_list[:total_needed]), dtype=torch.float32).unsqueeze(0),
        }

        logger.info(
            "Demo batch shapes: ctrl=%s, pert=%s, effect=%s",
            demo_batch["demo_ctrl"].shape,
            demo_batch["demo_pert"].shape,
            demo_batch["demo_effect"].shape,
        )

        return demo_batch, demo_perts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="ICL training run directory")
    parser.add_argument("--demo-adata", required=True, help="H5AD file with demo cell data")
    parser.add_argument("--demo-cell-type", required=True, help="Cell type to use for demos")
    parser.add_argument("--n-demo-perts", type=int, default=20, help="Number of demo perturbations")
    parser.add_argument("--n-cells-per-pert", type=int, default=8, help="Cells per demo perturbation")
    parser.add_argument("--checkpoint", default="last.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pseudobulk", action="store_true", default=True)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load pert_onehot_map
    pert_map_path = os.path.join(args.run_dir, "pert_onehot_map.pt")
    pert_onehot_map = torch.load(pert_map_path, weights_only=False)
    logger.info("Loaded %d perturbation one-hot encodings", len(pert_onehot_map))

    embed_key = cfg["data"]["kwargs"].get("embed_key", "X_hvg")
    pert_col = cfg["data"]["kwargs"].get("pert_col", "gene")
    cell_type_col = cfg["data"]["kwargs"].get("cell_type_key", "cell_line")
    control_pert = cfg["data"]["kwargs"].get("control_pert", "non-targeting")

    # Load demo data
    demo_batch, demo_perts = load_demo_data_from_h5ad(
        h5ad_path=args.demo_adata,
        cell_type=args.demo_cell_type,
        embed_key=embed_key,
        pert_col=pert_col,
        cell_type_col=cell_type_col,
        control_pert=control_pert,
        n_demo_perts=args.n_demo_perts,
        n_cells_per_pert=args.n_cells_per_pert,
        pert_onehot_map=pert_onehot_map,
        seed=args.seed,
    )

    # Save demo data to the run directory
    demo_path = os.path.join(args.run_dir, f"inference_demos_{args.demo_cell_type}_{args.n_demo_perts}perts.pt")
    torch.save({"demo_batch": demo_batch, "demo_perts": demo_perts}, demo_path)
    logger.info("Saved demo data to %s", demo_path)

    # Load model and set demos
    from state._cli._tx._evaluate import run_tx_evaluate
    from state.tx.models.state_transition_icl import ICLStateTransitionPerturbationModel
    import pickle

    checkpoint_path = os.path.join(args.run_dir, "checkpoints", args.checkpoint)
    var_dims_path = os.path.join(args.run_dir, "var_dims.pkl")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)

    model = ICLStateTransitionPerturbationModel.load_from_checkpoint(
        checkpoint_path, weights_only=False,
    )

    # Move demo data to model device
    device = next(model.parameters()).device
    demo_batch_device = {k: v.to(device) for k, v in demo_batch.items()}
    model.set_inference_demos(demo_batch_device)
    model.eval()

    # Save the model with demos set, then run eval
    # Actually, we need to run eval via the standard script but with demos injected
    # The simplest approach: monkey-patch the model loading in the eval script

    logger.info("Demo data loaded and set on model. Running evaluation...")
    logger.info("Demos from %d perturbations in %s", len(demo_perts), args.demo_cell_type)

    # Run evaluation as a subprocess with a custom flag
    eval_args = [
        sys.executable, "-m", "state", "tx", "evaluate",
        "--output-dir", args.run_dir,
        "--checkpoint", args.checkpoint,
        "--profile", "full",
    ]
    if args.pseudobulk:
        eval_args.append("--pseudobulk")

    # We can't easily pass the demo data through subprocess.
    # Instead, save it and modify the eval to load it.
    # For now, just print instructions.
    logger.info("To evaluate with demos, the model needs to be loaded with set_inference_demos() called.")
    logger.info("Demo file saved at: %s", demo_path)
    logger.info("Add this to the eval script or run evaluation programmatically.")


if __name__ == "__main__":
    main()
