import argparse
from typing import Any, Dict, List, Optional
import pandas as pd


class InferenceCancelledError(RuntimeError):
    """Raised when tx inference is cancelled by an external caller."""


def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint (.ckpt). If not provided, defaults to model_dir/checkpoints/final.ckpt",
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Key in adata.obsm for input features (if None, uses adata.X). If provided, .X will be left untouched in the output file.",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="drugname_drugconc",
        help="Column in adata.obs for perturbation labels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file (.h5ad or .npy). Defaults to <input>_simulated.h5ad",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help=(
            "Path to the training run directory. Must contain config.yaml, var_dims.pkl, pert_onehot_map.pt, and "
            "batch_onehot_map.torch (legacy batch_onehot_map.pkl is also supported). "
            "cell_type_onehot_map.torch is optional (legacy cell_type_onehot_map.pkl is also supported)."
        ),
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        default=None,
        help="Column in adata.obs to group by (defaults to auto-detected cell type column).",
    )
    parser.add_argument(
        "--celltypes",
        type=str,
        default=None,
        help="Comma-separated list of cell types to include (optional).",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help="Batch column name in adata.obs. If omitted, tries config['data']['kwargs']['batch_col'] then common fallbacks.",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=None,
        help="Override the control perturbation label. If omitted, read from config; for 'drugname_drugconc', defaults to \"[('DMSO_TF', 0.0, 'uM')]\".",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for control sampling (default: 42)",
    )
    parser.add_argument(
        "--max-set-len",
        type=int,
        default=None,
        help="Maximum set length per forward pass. If omitted, uses the model's trained cell_set_len.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )
    parser.add_argument(
        "--tsv",
        type=str,
        default=None,
        help="Path to TSV file with columns 'perturbation' and 'num_cells' to pad the adata with additional perturbation cells copied from random controls.",
    )
    parser.add_argument(
        "--all-perts",
        action="store_true",
        help="If set, add virtual copies of control cells for every perturbation in the saved one-hot map so all perturbations are simulated.",
    )
    parser.add_argument(
        "--cells-per-pert",
        type=int,
        default=None,
        help="Number of cells per perturbation. With --all-perts, limits control cells cloned for each virtual perturbation. With --tsv, this value is ignored.",
    )
    parser.add_argument(
        "--set-batch-size",
        type=int,
        default=None,
        help=(
            "Number of fixed-length cell sets to process per forward pass. Defaults to training.batch_size from config."
        ),
    )
    parser.add_argument(
        "--pseudobulk",
        action="store_true",
        help="Aggregate predictions into pseudobulk by (cell_type, perturbation) instead of writing cell-level output.",
    )
    parser.add_argument(
        "--deseq2",
        action="store_true",
        help="Run DESeq2 on pseudobulk predictions (implies --pseudobulk). Compares each perturbation against control.",
    )


def run_tx_infer(args: argparse.Namespace):
    import logging
    import os
    import pickle
    import sys
    import warnings

    import numpy as np
    import scanpy as sc
    import torch
    import yaml
    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel

    logger = logging.getLogger(__name__)
    progress_callback = getattr(args, "progress_callback", None)
    cancel_check = getattr(args, "cancel_check", None)

    def emit_event(kind: str, **payload: Any) -> None:
        if not callable(progress_callback):
            return
        event = {"kind": kind, **payload}
        try:
            progress_callback(event)
        except Exception as exc:
            warnings.warn(f"Progress callback failed ({type(exc).__name__}: {exc})")

    def ensure_not_cancelled() -> None:
        if not callable(cancel_check):
            return
        try:
            should_cancel = bool(cancel_check())
        except Exception as exc:
            warnings.warn(f"cancel_check failed ({type(exc).__name__}: {exc})")
            should_cancel = False
        if should_cancel:
            emit_event("cancelled", phase="inference", message="Inference cancelled.")
            raise InferenceCancelledError("Inference cancelled by request.")

    emit_event("phase", phase="initializing", message="Initializing tx inference.")

    # --deseq2 implies --pseudobulk
    if getattr(args, "deseq2", False) and not getattr(args, "pseudobulk", False):
        args.pseudobulk = True

    def info(msg: str, *fmt: Any) -> None:
        if args.quiet:
            return
        logger.info(msg, *fmt)

    def warn(msg: str, *fmt: Any) -> None:
        if args.quiet:
            return
        logger.warning(msg, *fmt)

    # -----------------------
    # Helpers
    # -----------------------
    def load_config(cfg_path: str) -> dict:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def to_dense(mat):
        """Return a dense numpy array for a variety of AnnData .X backends."""
        try:
            import scipy.sparse as sp

            if sp.issparse(mat):
                return mat.toarray()
        except Exception:
            pass
        return np.asarray(mat)

    def clip_array(arr, min_value: float = 0.0, max_value: float = 14.0) -> None:
        """Clip array values in-place for stability in gene-space outputs."""
        if arr is None:
            return
        np.clip(arr, min_value, max_value, out=arr)

    def pick_first_present(d: "sc.AnnData", candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in d.obs:
                return c
        return None

    def argmax_index_from_any(v, expected_dim: Optional[int]) -> Optional[int]:
        """
        Convert a saved mapping value (one-hot tensor, numpy array, or int) to an index.
        """
        if v is None:
            return None
        try:
            if torch.is_tensor(v):
                if v.ndim == 1:
                    return int(torch.argmax(v).item())
                else:
                    return None
        except Exception:
            pass
        try:
            import numpy as _np

            if isinstance(v, _np.ndarray):
                if v.ndim == 1:
                    return int(v.argmax())
                else:
                    return None
        except Exception:
            pass
        if isinstance(v, (int, np.integer)):
            return int(v)
        return None

    def load_onehot_map(model_dir: str, basename: str):
        candidates = [
            f"{basename}.torch",
            f"{basename}.pt",
            f"{basename}.pkl",
        ]
        resolved_paths = [os.path.join(model_dir, name) for name in candidates]
        for path in resolved_paths:
            if not os.path.exists(path):
                continue
            if path.endswith(".pkl"):
                with open(path, "rb") as f:
                    mapping = pickle.load(f)
            else:
                mapping = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(mapping, dict):
                raise TypeError(f"Expected dict in {path}, got {type(mapping).__name__}")
            return mapping, path, resolved_paths
        return None, None, resolved_paths

    def prepare_batch(
        ctrl_basal_np: np.ndarray,
        pert_onehots_np: np.ndarray,
        batch_indices_np: Optional[np.ndarray],
        pert_names_by_set: List[str],
        cell_set_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor | List[str]]:
        """
        Construct a model batch with fixed-length padded sets for batched inference.

        Args:
            ctrl_basal_np: Array of control features with shape [B, S, E].
            pert_onehots_np: Array of perturbation one-hots with shape [B, S, P].
            batch_indices_np: Optional integer batch indices with shape [B, S].
            pert_names_by_set: Perturbation name for each set (length B).
            cell_set_len: Sequence length S expected by the model in padded mode.
            device: Target torch device.
        """
        if ctrl_basal_np.ndim != 3:
            raise ValueError(f"Expected ctrl_basal_np shape [B, S, E], got shape {ctrl_basal_np.shape}")
        if pert_onehots_np.ndim != 3:
            raise ValueError(f"Expected pert_onehots_np shape [B, S, P], got shape {pert_onehots_np.shape}")
        if ctrl_basal_np.shape[:2] != pert_onehots_np.shape[:2]:
            raise ValueError(
                "ctrl_basal_np and pert_onehots_np must have matching [B, S] dimensions; "
                f"got {ctrl_basal_np.shape[:2]} vs {pert_onehots_np.shape[:2]}"
            )

        bsz, seq_len, emb_dim = ctrl_basal_np.shape
        if seq_len != cell_set_len:
            raise ValueError(f"Expected sequence length {cell_set_len}, got {seq_len}")
        if len(pert_names_by_set) != bsz:
            raise ValueError(f"Expected {bsz} perturbation names, got {len(pert_names_by_set)}")

        X_batch = torch.tensor(
            ctrl_basal_np.reshape(bsz * seq_len, emb_dim),
            dtype=torch.float32,
            device=device,
        )
        pert_batch = torch.tensor(
            pert_onehots_np.reshape(bsz * seq_len, pert_onehots_np.shape[-1]),
            dtype=torch.float32,
            device=device,
        )
        pert_names = [p for p in pert_names_by_set for _ in range(seq_len)]
        batch = {
            "ctrl_cell_emb": X_batch,
            "pert_emb": pert_batch,
            "pert_name": pert_names,
        }
        if batch_indices_np is not None:
            if batch_indices_np.shape != (bsz, seq_len):
                raise ValueError(
                    "batch_indices_np must have shape [B, S] to match inputs; "
                    f"got {batch_indices_np.shape} for expected {(bsz, seq_len)}"
                )
            batch["batch"] = torch.tensor(
                batch_indices_np.reshape(bsz * seq_len),
                dtype=torch.long,
                device=device,
            )
        return batch

    def pad_adata_with_tsv(
        adata: "sc.AnnData",
        tsv_path: str,
        pert_col: str,
        control_pert: str,
        rng: np.random.RandomState,
        quiet: bool = False,
    ) -> "sc.AnnData":
        """
        Pad AnnData with additional perturbation cells by copying random control cells
        and updating their perturbation labels according to the TSV specification.

        Args:
            adata: Input AnnData object
            tsv_path: Path to TSV file with 'perturbation' and 'num_cells' columns
            pert_col: Name of perturbation column in adata.obs
            control_pert: Label for control perturbation
            rng: Random number generator for sampling
            quiet: Whether to suppress logging

        Returns:
            AnnData object with padded cells
        """
        # Load TSV file
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        try:
            tsv_df = pd.read_csv(tsv_path, sep="\t")
        except Exception as e:
            raise ValueError(f"Error reading TSV file {tsv_path}: {e}")

        # Validate TSV format
        required_cols = ["perturbation", "num_cells"]
        missing_cols = [col for col in required_cols if col not in tsv_df.columns]
        if missing_cols:
            raise ValueError(
                f"TSV file missing required columns: {missing_cols}. Found columns: {list(tsv_df.columns)}"
            )

        # Find control cells
        ctl_mask = adata.obs[pert_col].astype(str) == str(control_pert)
        control_indices = np.where(ctl_mask)[0]

        if len(control_indices) == 0:
            raise ValueError(f"No control cells found with perturbation '{control_pert}' in column '{pert_col}'")

        if not quiet:
            info("Found %d control cells for padding", len(control_indices))

        # Collect cells to add
        new_cells_data = []
        total_to_add = 0

        for _, row in tsv_df.iterrows():
            pert_name = str(row["perturbation"])
            num_cells = int(row["num_cells"])
            total_to_add += num_cells

            if num_cells <= 0:
                continue

            # Sample control cells with replacement
            sampled_indices = rng.choice(control_indices, size=num_cells, replace=True)

            for idx in sampled_indices:
                new_cells_data.append({"original_index": idx, "new_perturbation": pert_name})

        if len(new_cells_data) == 0:
            if not quiet:
                info("No cells to add from TSV file")
            return adata

        if not quiet:
            info("Adding %d cells from TSV specification", total_to_add)

        # Create new AnnData with padded cells
        original_n_obs = adata.n_obs
        new_n_obs = original_n_obs + len(new_cells_data)

        # Copy X data
        if hasattr(adata.X, "toarray"):  # sparse matrix
            new_X = np.vstack(
                [adata.X.toarray(), adata.X[np.array([cell["original_index"] for cell in new_cells_data])].toarray()]
            )
        else:  # dense matrix
            new_X = np.vstack([adata.X, adata.X[np.array([cell["original_index"] for cell in new_cells_data])]])

        # Copy obs data
        new_obs = adata.obs.copy()
        for i, cell_data in enumerate(new_cells_data):
            orig_idx = cell_data["original_index"]
            new_pert = cell_data["new_perturbation"]

            # Copy the original control cell's metadata
            new_row = adata.obs.iloc[orig_idx].copy()
            # Update perturbation label
            new_row[pert_col] = new_pert

            new_obs.loc[original_n_obs + i] = new_row

        # Copy obsm data
        new_obsm = {}
        for key, matrix in adata.obsm.items():
            padded_matrix = np.vstack([matrix, matrix[np.array([cell["original_index"] for cell in new_cells_data])]])
            new_obsm[key] = padded_matrix

        # Copy varm, uns, var (unchanged)
        new_varm = adata.varm.copy()
        new_uns = adata.uns.copy()
        new_var = adata.var.copy()

        # Create new AnnData object
        import scanpy as sc

        new_adata = sc.AnnData(X=new_X, obs=new_obs, var=new_var, obsm=new_obsm, varm=new_varm, uns=new_uns)

        if not quiet:
            info("Padded AnnData: %d -> %d cells", original_n_obs, new_n_obs)

        return new_adata

    # -----------------------
    # Logging
    # -----------------------
    if not args.quiet:
        info("==> STATE: tx infer (virtual experiment)")

    # -----------------------
    # 1) Load config + dims + mappings
    # -----------------------
    config_path = os.path.join(args.model_dir, "config.yaml")
    ensure_not_cancelled()
    cfg = load_config(config_path)
    if not args.quiet:
        info("Loaded config: %s", config_path)
    emit_event("phase", phase="config_loaded", message=f"Loaded config: {config_path}")

    # control_pert
    control_pert = args.control_pert
    if control_pert is None:
        try:
            control_pert = cfg["data"]["kwargs"]["control_pert"]
        except Exception:
            control_pert = None
    if control_pert is None:
        raise ValueError(
            "Could not determine control perturbation. Provide --control-pert explicitly, "
            "or ensure data.kwargs.control_pert is set in the training config."
        )
    if not args.quiet:
        info("Control perturbation: %s", control_pert)
    control_pert_str = str(control_pert)

    # choose cell type column
    if args.celltype_col is None:
        ct_from_cfg = None
        try:
            ct_from_cfg = cfg["data"]["kwargs"].get("cell_type_key", None)
        except Exception:
            pass
        guess = pick_first_present(
            sc.read_h5ad(args.adata),
            candidates=[ct_from_cfg, "cell_type", "celltype", "cellType", "ctype", "celltype_col"]
            if ct_from_cfg
            else ["cell_type", "celltype", "cellType", "ctype", "celltype_col"],
        )
        args.celltype_col = guess
    if not args.quiet:
        info(
            "Grouping by cell type column: %s",
            args.celltype_col if args.celltype_col else "(not found; no grouping)",
        )

    # choose batch column
    if args.batch_col is None:
        try:
            args.batch_col = cfg["data"]["kwargs"].get("batch_col", None)
        except Exception:
            args.batch_col = None

    # dimensionalities
    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    if not os.path.exists(var_dims_path):
        raise FileNotFoundError(f"Missing var_dims.pkl at {var_dims_path}")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)

    pert_dim = var_dims.get("pert_dim")
    batch_dim = var_dims.get("batch_dim", None)

    # mappings
    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    if not os.path.exists(pert_onehot_map_path):
        raise FileNotFoundError(f"Missing pert_onehot_map.pt at {pert_onehot_map_path}")
    pert_onehot_map: Dict[str, torch.Tensor] = torch.load(pert_onehot_map_path, weights_only=False)
    pert_name_lookup: Dict[str, object] = {str(k): k for k in pert_onehot_map.keys()}
    pert_names_in_map: List[str] = list(pert_name_lookup.keys())

    batch_onehot_map, loaded_batch_onehot_map_path, batch_onehot_map_candidates = load_onehot_map(
        args.model_dir, "batch_onehot_map"
    )
    if loaded_batch_onehot_map_path is not None and not args.quiet:
        info("Loaded batch one-hot map from: %s", loaded_batch_onehot_map_path)
    cell_type_onehot_map, loaded_cell_type_onehot_map_path, _ = load_onehot_map(args.model_dir, "cell_type_onehot_map")
    if loaded_cell_type_onehot_map_path is not None and not args.quiet:
        info("Loaded cell type one-hot map from: %s", loaded_cell_type_onehot_map_path)

    # -----------------------
    # 2) Load model
    # -----------------------
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.model_dir, "checkpoints", "final.ckpt")
        if not args.quiet:
            info("No --checkpoint given, using %s", checkpoint_path)

    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred_device.type == "cuda":
        try:
            model = StateTransitionPerturbationModel.load_from_checkpoint(
                checkpoint_path,
                map_location=preferred_device,
                weights_only=False,
            )
            model = model.to(preferred_device)
        except Exception as e:
            warnings.warn(f"Failed to load model on CUDA ({e}); falling back to CPU.")
            preferred_device = torch.device("cpu")
            model = StateTransitionPerturbationModel.load_from_checkpoint(
                checkpoint_path,
                map_location=preferred_device,
                weights_only=False,
            )
            model = model.to(preferred_device)
    else:
        model = StateTransitionPerturbationModel.load_from_checkpoint(
            checkpoint_path,
            map_location=preferred_device,
            weights_only=False,
        )
        model = model.to(preferred_device)

    model.eval()
    device = next(model.parameters()).device
    cell_set_len = args.max_set_len if args.max_set_len is not None else getattr(model, "cell_sentence_len", 256)
    if cell_set_len <= 0:
        raise ValueError(f"Resolved cell_set_len must be a positive integer, got {cell_set_len}")
    set_batch_size = args.set_batch_size
    if set_batch_size is None:
        set_batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    if set_batch_size <= 0:
        raise ValueError(f"--set-batch-size must be a positive integer, got {set_batch_size}")
    uses_batch_encoder = getattr(model, "batch_encoder", None) is not None
    _OUTPUT_SPACE_ALIASES = {"hvg": "gene", "transcriptome": "all"}
    output_space = getattr(model, "output_space", cfg.get("data", {}).get("kwargs", {}).get("output_space", "gene"))
    output_space = _OUTPUT_SPACE_ALIASES.get(output_space.strip().lower(), output_space)
    if output_space not in {"embedding", "gene", "all"}:
        raise ValueError(
            f"Invalid output_space '{output_space}'. Must be one of: 'embedding', 'gene', 'all'."
        )
    nb_loss_enabled = bool(getattr(model, "nb_loss", cfg.get("model", {}).get("kwargs", {}).get("nb_loss", False)))
    if nb_loss_enabled and output_space == "embedding":
        raise ValueError(
            "model.kwargs.nb_loss=True is incompatible with data.kwargs.output_space='embedding'. "
            "Use output_space='gene' or output_space='all'."
        )
    if nb_loss_enabled and output_space not in {"gene", "all"}:
        raise ValueError(f"nb_loss=True requires output_space in {{'gene', 'all'}}; got {output_space!r}.")

    if not args.quiet:
        info("Model device: %s", device)
        info("Model cell_set_len (max sequence length): %s", cell_set_len)
        info("Batched padded inference: enabled (set_batch_size=%s)", set_batch_size)
        info("Model uses batch encoder: %s", bool(uses_batch_encoder))
        info("Model output space: %s", output_space)
        info("Model nb_loss enabled: %s", nb_loss_enabled)
    emit_event(
        "phase",
        phase="model_loaded",
        message=f"Model loaded on {device}.",
        device=str(device),
        cell_set_len=cell_set_len,
        set_batch_size=set_batch_size,
        batched=True,
        output_space=output_space,
        nb_loss=nb_loss_enabled,
    )

    # -----------------------
    # 3) Load AnnData
    # -----------------------
    ensure_not_cancelled()
    adata = sc.read_h5ad(args.adata)
    emit_event(
        "phase",
        phase="adata_loaded",
        message=f"Loaded AnnData: {args.adata}",
        adata_path=args.adata,
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
    )

    if args.tsv and args.all_perts:
        raise ValueError("--tsv and --all-perts are mutually exclusive. Use one or the other.")

    # optional TSV padding mode - pad with additional perturbation cells
    if args.tsv:
        if not args.quiet:
            info("==> TSV padding mode: loading %s", args.tsv)

        # Initialize RNG for padding (separate from inference RNG for reproducibility)
        pad_rng = np.random.RandomState(args.seed)

        adata = pad_adata_with_tsv(
            adata=adata,
            tsv_path=args.tsv,
            pert_col=args.pert_col,
            control_pert=control_pert,
            rng=pad_rng,
            quiet=args.quiet,
        )

    # optional filter by cell types
    if args.celltype_col and args.celltypes:
        keep_cts = [ct.strip() for ct in args.celltypes.split(",")]
        if cell_type_onehot_map is not None:
            available_cell_types = {str(k) for k in cell_type_onehot_map.keys()}
            missing_in_map = [ct for ct in keep_cts if ct not in available_cell_types]
            if missing_in_map and not args.quiet:
                preview = ", ".join(missing_in_map[:5])
                if len(missing_in_map) > 5:
                    preview += ", ..."
                warn(
                    "%d requested cell types not found in saved mapping (examples: %s).",
                    len(missing_in_map),
                    preview,
                )
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not in adata.obs")
        n0 = adata.n_obs
        adata = adata[adata.obs[args.celltype_col].isin(keep_cts)].copy()
        if not args.quiet:
            info("Filtered to %d cells (from %d) for cell types: %s", adata.n_obs, n0, keep_cts)

    if args.all_perts:
        if args.pert_col not in adata.obs:
            raise KeyError(f"Perturbation column '{args.pert_col}' not found in adata.obs")
        adata.obs = adata.obs.copy()
        adata.obs[args.pert_col] = adata.obs[args.pert_col].astype(str)

    # optionally expand controls to cover every perturbation in the map
    if args.all_perts:
        observed_perts = set(adata.obs[args.pert_col].values)
        missing_perts = [p for p in pert_names_in_map if p not in observed_perts]

        if missing_perts:
            ctrl_mask_all_perts = adata.obs[args.pert_col] == control_pert_str
            if not bool(np.any(ctrl_mask_all_perts)):
                raise ValueError(
                    "--all-perts requested, but no control cells are available to template new perturbations."
                )

            ctrl_template = adata[ctrl_mask_all_perts].copy()
            ctrl_template.obs = ctrl_template.obs.copy()
            ctrl_template.obs[args.pert_col] = ctrl_template.obs[args.pert_col].astype(str)

            if args.cells_per_pert is not None:
                if args.cells_per_pert <= 0:
                    raise ValueError("--cells-per-pert must be a positive integer if provided.")
                if ctrl_template.n_obs > args.cells_per_pert:
                    virtual_rng = np.random.RandomState(args.seed)
                    sampled_idx = virtual_rng.choice(
                        ctrl_template.n_obs, size=args.cells_per_pert, replace=False
                    )
                    ctrl_template = ctrl_template[sampled_idx].copy()
                    ctrl_template.obs = ctrl_template.obs.copy()
                    ctrl_template.obs[args.pert_col] = ctrl_template.obs[args.pert_col].astype(str)
                    if not args.quiet:
                        info(
                            "--all-perts: limiting virtual control template to %d cells per perturbation "
                            "(requested %d).",
                            ctrl_template.n_obs,
                            args.cells_per_pert,
                        )

            virtual_blocks: List["sc.AnnData"] = []
            for pert_name in missing_perts:
                clone = ctrl_template.copy()
                clone.obs = clone.obs.copy()
                clone.obs[args.pert_col] = pert_name
                clone.obs_names = [f"{obs_name}__virt_{pert_name}" for obs_name in clone.obs_names]
                virtual_blocks.append(clone)

            adata = sc.concat([adata, *virtual_blocks], axis=0, join="inner")

            if not args.quiet:
                preview = ", ".join(missing_perts[:5])
                if len(missing_perts) > 5:
                    preview += ", ..."
                info(
                    "Added virtual control copies for %d perturbations (%s). Total cells: %d.",
                    len(missing_perts),
                    preview if preview else "n/a",
                    adata.n_obs,
                )
        elif not args.quiet:
            info("--all-perts requested, but all perturbations already present in AnnData.")

    # select features: embeddings or genes
    if args.embed_key is None:
        X_in = to_dense(adata.X)  # [N, E_in]
        writes_to = (".X", None)  # write predictions to .X
    else:
        if args.embed_key not in adata.obsm:
            raise KeyError(f"Embedding key '{args.embed_key}' not found in adata.obsm")
        X_in = np.asarray(adata.obsm[args.embed_key])  # [N, E_in]
        writes_to = (".obsm", args.embed_key)  # write predictions to obsm[embed_key]

    if not args.quiet:
        info(
            "Using %s as input features: shape %s",
            "adata.X" if args.embed_key is None else f"adata.obsm[{args.embed_key!r}]",
            X_in.shape,
        )

    # pick pert names; ensure they are strings
    if args.pert_col not in adata.obs:
        raise KeyError(f"Perturbation column '{args.pert_col}' not found in adata.obs")
    pert_names_all = adata.obs[args.pert_col].astype(str).values

    # derive batch indices (per-token integers) if needed
    batch_indices_all: Optional[np.ndarray] = None
    if uses_batch_encoder:
        # locate batch column
        batch_col = args.batch_col
        if batch_col is None:
            candidates = ["gem_group", "gemgroup", "batch", "donor", "plate", "experiment", "lane", "batch_id"]
            batch_col = next((c for c in candidates if c in adata.obs), None)
        if batch_col is not None and batch_col in adata.obs:
            raw_labels = adata.obs[batch_col].astype(str).values
            if batch_onehot_map is None:
                raise FileNotFoundError(
                    "Model has a batch encoder, but no batch one-hot map was found at any of: "
                    + ", ".join(batch_onehot_map_candidates)
                    + ". "
                    "Cannot proceed without batch encoding."
                )
            else:
                # Convert labels to indices using saved map
                label_to_idx: Dict[str, int] = {}
                for k, v in batch_onehot_map.items():
                    key = str(k)
                    idx = argmax_index_from_any(v, expected_dim=batch_dim)
                    if idx is not None:
                        label_to_idx[key] = idx
                idxs = np.zeros(len(raw_labels), dtype=np.int64)
                misses = 0
                for i, lab in enumerate(raw_labels):
                    if lab in label_to_idx:
                        idxs[i] = label_to_idx[lab]
                    else:
                        misses += 1
                        idxs[i] = 0  # fallback to zero
                if misses and not args.quiet:
                    warn(
                        "%d / %d batch labels not found in saved mapping; using index 0 as fallback.",
                        misses,
                        len(raw_labels),
                    )
                batch_indices_all = idxs
        else:
            if not args.quiet:
                info("Batch encoder present, but no batch column found; proceeding without batch indices.")
            uses_batch_encoder = False

    # -----------------------
    # 4) Build control template on the fly & simulate ALL cells (controls included)
    # -----------------------
    rng = np.random.RandomState(args.seed)

    # Identify control vs non-control
    ctl_mask = pert_names_all == control_pert_str
    n_controls = int(ctl_mask.sum())
    n_total = adata.n_obs
    n_nonctl = n_total - n_controls
    if not args.quiet:
        info("Cells: total=%d, control=%d, non-control=%d", n_total, n_controls, n_nonctl)

    # Where we will write predictions (initialize with originals; we overwrite all rows, including controls)
    if writes_to[0] == ".X":
        sim_X = X_in.astype(np.float32, copy=True)
        out_target = "X"
    else:
        sim_obsm = X_in.astype(np.float32, copy=True)
        out_target = f"obsm['{writes_to[1]}']"

    store_raw_expression = (args.embed_key is not None and args.embed_key != "X_hvg" and output_space == "gene") or (
        args.embed_key is not None and output_space == "all"
    )
    counts_expected = store_raw_expression or nb_loss_enabled
    counts_out_target: Optional[str] = None
    counts_obsm_key: Optional[str] = None
    sim_counts: Optional[np.ndarray] = None
    counts_written = False

    if counts_expected:
        if output_space == "gene":
            counts_out_target = "obsm['X_hvg']"
            counts_obsm_key = "X_hvg"
        elif output_space == "all":
            counts_out_target = "X"
            if writes_to[0] == ".X":
                sim_counts = sim_X

    # Group labels for set-to-set behavior
    if args.celltype_col and args.celltype_col in adata.obs:
        group_labels = adata.obs[args.celltype_col].astype(str).values
        unique_groups = np.unique(group_labels)
    else:
        group_labels = np.array(["__ALL__"] * n_total)
        unique_groups = np.array(["__ALL__"])

    group_pert_totals: dict[str, int] = {}
    total_perts = 0
    for g in unique_groups:
        g_mask = group_labels == g
        g_total = int(np.unique(pert_names_all[g_mask]).shape[0])
        group_pert_totals[str(g)] = g_total
        total_perts += g_total

    perts_done = 0
    cells_done = 0

    # Control pools (group-specific with fallback to global)
    all_control_indices = np.where(ctl_mask)[0]

    def group_control_indices(group_name: str) -> np.ndarray:
        if group_name == "__ALL__":
            return all_control_indices
        grp_mask = group_labels == group_name
        grp_ctl = np.where(grp_mask & ctl_mask)[0]
        return grp_ctl if len(grp_ctl) > 0 else all_control_indices

    # default pert vector when unmapped label shows up
    control_map_key = pert_name_lookup.get(control_pert_str, control_pert)
    if control_map_key in pert_onehot_map:
        default_pert_vec = pert_onehot_map[control_map_key].float().clone()
    else:
        default_pert_vec = torch.zeros(pert_dim, dtype=torch.float32)
        if pert_dim and pert_dim > 0:
            default_pert_vec[0] = 1.0

    internal_padding_tokens = 0

    def ensure_counts_storage(target_dim: int):
        nonlocal sim_counts
        if sim_counts is not None:
            if sim_counts.shape[1] != target_dim:
                raise ValueError(
                    f"Predicted counts dimension mismatch: expected {sim_counts.shape[1]} but got {target_dim}"
                )
            return

        if output_space == "gene":
            if counts_obsm_key and counts_obsm_key in adata.obsm:
                existing = np.asarray(adata.obsm[counts_obsm_key])
                if existing.shape[1] == target_dim:
                    sim_counts = existing.astype(np.float32, copy=True)
                    return
                if not args.quiet:
                    warn(
                        "Dimension mismatch for existing obsm[%r] (got %d vs predictions %d). "
                        "Reinitializing storage with zeros.",
                        counts_obsm_key,
                        existing.shape[1],
                        target_dim,
                    )
            sim_counts = np.zeros((n_total, target_dim), dtype=np.float32)
            return

        # output_space == "all"
        if writes_to[0] == ".X":
            sim_counts = sim_X
        else:
            sim_counts = np.zeros((n_total, target_dim), dtype=np.float32)

    def write_pred_rows(row_indices: np.ndarray, pred_rows: np.ndarray):
        nonlocal out_target
        if writes_to[0] == ".X":
            if pred_rows.shape[1] != sim_X.shape[1]:
                raise ValueError(
                    f"Prediction dimension mismatch for X: model produced {pred_rows.shape[1]} "
                    f"but input has {sim_X.shape[1]} features. Check model output_space and input data."
                )
            sim_X[row_indices, :] = pred_rows
        else:
            if pred_rows.shape[1] != sim_obsm.shape[1]:
                raise ValueError(
                    f"Prediction dimension mismatch for obsm['{writes_to[1]}']: model produced "
                    f"{pred_rows.shape[1]} but input has {sim_obsm.shape[1]} features. "
                    f"Check model output_space and input data."
                )
            sim_obsm[row_indices, :] = pred_rows

    if not args.quiet:
        info(
            "Running virtual experiment (homogeneous per-perturbation forward passes; "
            "batched fixed-length sets with replacement padding)..."
        )
    emit_event(
        "phase",
        phase="inference_started",
        message="Started batched padded inference.",
        groups_total=int(len(unique_groups)),
        perturbations_total=int(total_perts),
        cells_total=int(n_total),
    )

    model_device = next(model.parameters()).device

    # --- Pseudobulk accumulator setup ---
    accumulator = None
    if getattr(args, "pseudobulk", False):
        from ._pseudobulk import PseudobulkAccumulator, resolve_scale_flags, build_pseudobulk_anndata

        # Determine if outputs are log1p-scaled from training config
        is_log1p = cfg.get("data", {}).get("kwargs", {}).get("is_log1p", True)
        agg_main, agg_gene = resolve_scale_flags(
            metrics_is_log1p=is_log1p,
            use_count_outputs=False,  # infer path doesn't use count outputs
            resolved_exp_counts=False,
            output_space=output_space,
        )
        accumulator = PseudobulkAccumulator(
            output_dim=X_in.shape[1],
            gene_dim=None,
            use_count_outputs=False,
            aggregate_main_in_count_space=agg_main,
            aggregate_gene_in_count_space=agg_gene,
            has_real=False,
            enable_deseq2=getattr(args, "deseq2", False),
            nb_loss_enabled=nb_loss_enabled,
        )
        if not args.quiet:
            info("Pseudobulk mode enabled; aggregating by (cell_type, perturbation).")
            if agg_main:
                info("Pseudobulk scale: detected log1p outputs; accumulating in count space via expm1.")
            if getattr(args, "deseq2", False):
                info("DESeq2 mode enabled; will compute DE on predictions vs control.")

    # Match training precision for inference autocast (prevents CUBLAS errors with bf16-trained models)
    import contextlib

    training_precision = cfg.get("training", {}).get("precision", "32-true")
    if "bf16" in training_precision:
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif "16" in training_precision:
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    with torch.no_grad(), autocast_ctx:
        for group_index, g in enumerate(unique_groups, start=1):
            ensure_not_cancelled()
            grp_idx = np.where(group_labels == g)[0]
            if len(grp_idx) == 0:
                continue

            # control pool for this group (fallback to global if empty)
            grp_ctrl_pool = group_control_indices(g)
            if len(grp_ctrl_pool) == 0:
                if not args.quiet:
                    warn("Group %r: no control cells available anywhere; leaving rows unchanged.", g)
                continue

            emit_event(
                "phase",
                phase="group_started",
                message=f"Started group '{g}'.",
                group=str(g),
                group_index=group_index,
                groups_total=int(len(unique_groups)),
                group_perturbations_total=int(group_pert_totals.get(str(g), 0)),
                perturbations_done=int(perts_done),
                perturbations_total=int(total_perts),
                cells_done=int(cells_done),
                cells_total=int(n_total),
            )

            # --- iterate by perturbation and batch multiple fixed-length sets ---
            grp_perts = np.unique(pert_names_all[grp_idx])
            group_perts_done = 0
            POSTFIX_WIDTH = 30
            pbar = tqdm(
                grp_perts,
                desc=f"Group {g}",
                bar_format="{l_bar}{bar}{r_bar}",
                dynamic_ncols=True,
                disable=args.quiet,
                file=sys.stderr,
            )
            for p in pbar:
                ensure_not_cancelled()
                current_postfix = f"Pert: {p}"
                pbar.set_postfix_str(f"{current_postfix:<{POSTFIX_WIDTH}.{POSTFIX_WIDTH}}")

                idxs = grp_idx[pert_names_all[grp_idx] == p]
                if len(idxs) == 0:
                    continue

                map_key = pert_name_lookup.get(p, p)
                vec = pert_onehot_map.get(map_key, None)
                if vec is None:
                    vec = default_pert_vec
                    logger.warning("Group %r perturbation %r not in mapping; using control fallback one-hot.", g, p)

                sentence_specs = []
                start = 0
                while start < len(idxs):
                    ensure_not_cancelled()
                    end = min(start + cell_set_len, len(idxs))
                    real_rows = idxs[start:end]
                    real_n = len(real_rows)
                    if real_n == 0:
                        break

                    if real_n < cell_set_len:
                        fill_rows = rng.choice(real_rows, size=cell_set_len - real_n, replace=True)
                        set_rows = np.concatenate([real_rows, fill_rows]).astype(np.int64, copy=False)
                        internal_padding_tokens += int(cell_set_len - real_n)
                    else:
                        set_rows = real_rows.astype(np.int64, copy=False)

                    sampled_ctrl_idx = rng.choice(grp_ctrl_pool, size=cell_set_len, replace=True).astype(
                        np.int64, copy=False
                    )

                    if uses_batch_encoder and batch_indices_all is not None:
                        set_batch_idx = batch_indices_all[set_rows].astype(np.int64, copy=False)
                    else:
                        set_batch_idx = None

                    sentence_specs.append(
                        {
                            "real_rows": real_rows.astype(np.int64, copy=False),
                            "real_n": real_n,
                            "ctrl_rows": sampled_ctrl_idx,
                            "set_batch_idx": set_batch_idx,
                        }
                    )

                    should_flush = len(sentence_specs) >= set_batch_size or end >= len(idxs)
                    if should_flush:
                        bsz = len(sentence_specs)
                        ctrl_stack = np.stack([X_in[s["ctrl_rows"], :] for s in sentence_specs], axis=0).astype(
                            np.float32, copy=False
                        )
                        if torch.is_tensor(vec):
                            vec_np = vec.detach().float().cpu().numpy()
                        else:
                            vec_np = np.asarray(vec, dtype=np.float32)
                        if vec_np.ndim != 1:
                            raise ValueError(f"Expected 1D perturbation vector for '{p}', got shape {vec_np.shape}")
                        pert_stack = np.tile(vec_np.reshape(1, 1, -1), (bsz, cell_set_len, 1))

                        if uses_batch_encoder and batch_indices_all is not None:
                            batch_stack = np.stack([s["set_batch_idx"] for s in sentence_specs], axis=0).astype(
                                np.int64, copy=False
                            )
                        else:
                            batch_stack = None

                        batch = prepare_batch(
                            ctrl_basal_np=ctrl_stack,
                            pert_onehots_np=pert_stack,
                            batch_indices_np=batch_stack,
                            pert_names_by_set=[p] * bsz,
                            cell_set_len=cell_set_len,
                            device=model_device,
                        )
                        batch_out = model.predict_step(batch, batch_idx=0, padded=True)

                        if (
                            counts_expected
                            and writes_to[0] == ".X"
                            and ("pert_cell_counts_preds" in batch_out)
                            and (batch_out["pert_cell_counts_preds"] is not None)
                        ):
                            preds_flat = batch_out["pert_cell_counts_preds"].detach().float().cpu().numpy()
                        else:
                            preds_flat = batch_out["preds"].detach().float().cpu().numpy()
                        preds = preds_flat.reshape(bsz, cell_set_len, -1)

                        counts_preds = None
                        if counts_expected and ("pert_cell_counts_preds" in batch_out):
                            counts_tensor = batch_out.get("pert_cell_counts_preds")
                            if counts_tensor is not None:
                                counts_flat = counts_tensor.detach().float().cpu().numpy()
                                counts_preds = counts_flat.reshape(bsz, cell_set_len, -1)

                        if counts_preds is not None:
                            ensure_counts_storage(counts_preds.shape[2])
                            counts_written = True

                        for set_i, spec in enumerate(sentence_specs):
                            real_n_local = int(spec["real_n"])
                            real_rows_local = spec["real_rows"]

                            pred_rows = preds[set_i, :real_n_local, :]
                            if accumulator is not None:
                                # Get context/metadata for these cells
                                cell_cts = group_labels[real_rows_local] if group_labels is not None else ["__ALL__"] * real_n_local
                                cell_perts = [p] * real_n_local
                                cell_contexts = [str(ct) for ct in cell_cts]
                                cell_batches = ["__NA__"] * real_n_local
                                accumulator.accumulate_batch(
                                    batch_size=real_n_local,
                                    context_labels=cell_contexts,
                                    pert_names=cell_perts,
                                    celltypes=[str(ct) for ct in cell_cts],
                                    batch_labels=cell_batches,
                                    pred_np=pred_rows,
                                )
                            else:
                                write_pred_rows(real_rows_local, pred_rows)
                            cells_done += real_n_local

                            if counts_preds is not None and sim_counts is not None:
                                sim_counts[real_rows_local, :] = counts_preds[set_i, :real_n_local, :]

                        sentence_specs = []

                    start = end  # next sentence

                perts_done += 1
                group_perts_done += 1
                emit_event(
                    "progress",
                    phase="inference",
                    message=f"Completed perturbation '{p}' in group '{g}'.",
                    group=str(g),
                    group_index=group_index,
                    groups_total=int(len(unique_groups)),
                    group_perturbations_done=int(group_perts_done),
                    group_perturbations_total=int(group_pert_totals.get(str(g), 0)),
                    perturbations_done=int(perts_done),
                    perturbations_total=int(total_perts),
                    cells_done=int(cells_done),
                    cells_total=int(n_total),
                    internal_padding_tokens=int(internal_padding_tokens),
                )

            emit_event(
                "phase",
                phase="group_completed",
                message=f"Completed group '{g}'.",
                group=str(g),
                group_index=group_index,
                groups_total=int(len(unique_groups)),
                group_perturbations_done=int(group_perts_done),
                group_perturbations_total=int(group_pert_totals.get(str(g), 0)),
                perturbations_done=int(perts_done),
                perturbations_total=int(total_perts),
                cells_done=int(cells_done),
                cells_total=int(n_total),
                internal_padding_tokens=int(internal_padding_tokens),
            )

    # Clip legacy decoder outputs only; NB count outputs remain unclipped.
    if output_space in {"gene", "all"}:
        if nb_loss_enabled:
            if not args.quiet:
                info("nb_loss=True: skipping clipping of simulated outputs.")
        else:
            if out_target == "X":
                clip_array(sim_X)
            elif out_target.startswith("obsm['") and out_target.endswith("']"):
                pred_key = out_target[6:-2]
                if writes_to[0] == ".obsm" and pred_key == writes_to[1]:
                    clip_array(sim_obsm)
                elif pred_key in adata.obsm:
                    clip_array(adata.obsm[pred_key])
            else:
                if writes_to[0] == ".X":
                    clip_array(sim_X)
                else:
                    clip_array(sim_obsm)

            if counts_written and sim_counts is not None:
                clip_array(sim_counts)

    # -----------------------
    # 4b) Pseudobulk finalization
    # -----------------------
    if accumulator is not None:
        group_entries, total_cells = accumulator.finalize()
        if len(group_entries) == 0:
            raise RuntimeError("No pseudobulk groups were generated. Check that input data has perturbation labels.")

        if not args.quiet:
            info("Built %d pseudobulk groups from %d cells.", len(group_entries), total_cells)

        celltype_col = args.celltype_col or "cell_type"
        result = build_pseudobulk_anndata(
            group_entries,
            output_dim=X_in.shape[1],
            gene_dim=None,
            use_count_outputs=False,
            aggregate_main_in_count_space=agg_main,
            aggregate_gene_in_count_space=agg_gene,
            has_real=False,
            pert_col=args.pert_col,
            cell_type_key=celltype_col,
            batch_obs_key="batch",
            embed_key=args.embed_key,
        )

        output_path = args.output or args.adata.replace(".h5ad", "_simulated.h5ad")
        result["adata_pred"].write_h5ad(output_path)

        if not args.quiet:
            info("Wrote pseudobulk predictions (%d groups) to %s", len(group_entries), output_path)

        # --- DESeq2 on predictions ---
        if getattr(args, "deseq2", False):
            try:
                from deseq2_pipeline import run_from_precomputed
            except ImportError:
                logger.warning(
                    "deseq2-pipeline not installed. Skipping DESeq2. "
                    "Install with: pip install deseq2-pipeline"
                )
                run_from_precomputed = None

            if run_from_precomputed is not None:
                import os

                output_dir = os.path.dirname(output_path) or "."
                # Group entries by cell type
                ct_groups: dict[str, dict[str, dict]] = {}
                ct_ctrl: dict[str, dict] = {}
                for entry in group_entries:
                    ct = entry["celltype_name"]
                    pname = entry["pert_name"]
                    if ct not in ct_groups:
                        ct_groups[ct] = {}
                        ct_ctrl[ct] = None
                    if pname == control_pert_str:
                        ct_ctrl[ct] = entry
                    else:
                        ct_groups[ct][pname] = entry

                for ct in sorted(ct_groups.keys()):
                    ctrl_entry = ct_ctrl.get(ct)
                    if ctrl_entry is None:
                        logger.warning("Cell type '%s': no control entry found, skipping DESeq2.", ct)
                        continue
                    pert_entries = ct_groups[ct]
                    if not pert_entries:
                        continue

                    ct_safe = ct.replace("/", "-")
                    pred_pert_counts = {
                        p: np.stack(e["pred_rep_sums"]).astype(np.int64)
                        for p, e in pert_entries.items()
                    }
                    pred_ctrl = np.stack(ctrl_entry["pred_rep_sums"]).astype(np.int64)

                    # Determine gene names
                    _deseq2_dim = X_in.shape[1]
                    _deseq2_var = [f"gene_{i}" for i in range(_deseq2_dim)]

                    pred_outdir = os.path.join(output_dir, f"deseq2_{ct_safe}_pred")
                    if not args.quiet:
                        info("Running DESeq2 for cell type '%s' (pred, %d perts)...", ct, len(pred_pert_counts))
                    try:
                        run_from_precomputed(
                            pred_pert_counts, pred_ctrl, _deseq2_var,
                            outdir=pred_outdir,
                        )
                        if not args.quiet:
                            info("DESeq2 complete for cell type '%s'.", ct)
                    except Exception as exc:
                        logger.warning("DESeq2 failed for cell type '%s': %s", ct, exc)

        emit_event(
            "summary",
            phase="completed",
            message="Pseudobulk inference completed successfully.",
            output_path=output_path,
            n_groups=len(group_entries),
            total_cells=total_cells,
        )
        if not args.quiet:
            logger.info("=== Pseudobulk inference complete ===")
            logger.info("Pseudobulk groups: %d", len(group_entries))
            logger.info("Total cells aggregated: %d", total_cells)
            logger.info("Saved: %s", output_path)
        return

    # -----------------------
    # 5) Persist the updated AnnData
    # -----------------------
    output_path = args.output or args.adata.replace(".h5ad", "_simulated.h5ad")
    output_is_npy = output_path.lower().endswith(".npy")

    if counts_expected and not counts_written and not args.quiet:
        warn(
            "Model configured to produce gene counts, but no predicted counts were returned; counts will not be saved."
        )

    pred_matrix = None
    if writes_to[0] == ".X":
        if out_target == "X":
            adata.X = sim_X
            pred_matrix = sim_X
        elif out_target.startswith("obsm['") and out_target.endswith("']"):
            pred_key = out_target[6:-2]
            pred_matrix = adata.obsm.get(pred_key)
        else:
            pred_matrix = sim_X
    else:
        if out_target == f"obsm['{writes_to[1]}']":
            adata.obsm[writes_to[1]] = sim_obsm
            pred_matrix = sim_obsm
        elif out_target.startswith("obsm['") and out_target.endswith("']"):
            pred_key = out_target[6:-2]
            pred_matrix = adata.obsm.get(pred_key)
        else:
            pred_matrix = sim_obsm

    if counts_written and sim_counts is not None:
        if output_space == "gene":
            key = counts_obsm_key or "X_hvg"
            adata.obsm[key] = sim_counts
        elif output_space == "all":
            adata.X = sim_counts

    if output_is_npy:
        if pred_matrix is None:
            raise ValueError("Predictions matrix is unavailable; cannot write .npy output")
        np.save(output_path, np.asarray(pred_matrix))
    else:
        adata.write_h5ad(output_path)

    emit_event(
        "summary",
        phase="completed",
        message="Inference completed successfully.",
        output_path=output_path,
        output_is_npy=bool(output_is_npy),
        cells_total=int(n_total),
        controls_simulated=int(n_controls),
        treated_simulated=int(n_nonctl),
        perturbations_done=int(perts_done),
        perturbations_total=int(total_perts),
        internal_padding_tokens=int(internal_padding_tokens),
        counts_written=bool(counts_written),
        counts_out_target=counts_out_target,
        out_target=out_target,
    )

    # -----------------------
    # 6) Summary
    # -----------------------
    if not args.quiet:
        logger.info("=== Inference complete ===")
        logger.info("Input cells: %d", n_total)
        logger.info("Controls simulated: %d", n_controls)
        logger.info("Treated simulated: %d", n_nonctl)
        logger.info("Internal padded tokens dropped: %d", internal_padding_tokens)
        if output_is_npy:
            shape_str = " x ".join(str(dim) for dim in pred_matrix.shape) if pred_matrix is not None else "unknown"
            logger.info("Wrote predictions array (shape: %s)", shape_str)
            logger.info("Saved NumPy file: %s", output_path)
        else:
            logger.info("Wrote predictions to adata.%s", out_target)
            logger.info("Saved: %s", output_path)
        if counts_written and counts_out_target:
            logger.info("Saved count predictions to adata.%s", counts_out_target)
