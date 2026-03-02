import argparse as ap


def add_arguments_predict(parser: ap.ArgumentParser):
    """
    CLI for evaluation using cell-eval metrics.
    """

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--toml",
        type=str,
        default=None,
        help="Optional path to a TOML data config to use instead of the training config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="run all metrics, minimal, only de metrics, or only output adatas",
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="If set, only run prediction without evaluation metrics.",
    )

    parser.add_argument(
        "--skip-adatas",
        action="store_true",
        help="If set, skip writing AnnData (.h5ad) outputs and only run metrics/evaluation.",
    )
    parser.add_argument(
        "--skip-de",
        action="store_true",
        help="If set, skip DE computation in cell-eval and only compute AnnData-based metrics.",
    )

    parser.add_argument(
        "--shared-only",
        action="store_true",
        help=("If set, restrict predictions/evaluation to perturbations shared between train and test (train ∩ test)."),
    )

    parser.add_argument(
        "--eval-train-data",
        action="store_true",
        help="If set, evaluate the model on the training data rather than on the test data.",
    )

    parser.add_argument(
        "--pseudobulk",
        action="store_true",
        help=(
            "If set, aggregate predictions in a streaming fashion into running pseudobulks by "
            "(context, perturbation) before cell-eval."
        ),
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help=(
            "Number of perturbations per DE batch. When >0 with --pseudobulk, "
            "computes cell-level DE in batches via temporary disk storage. "
            "Non-DE metrics still use pseudobulk. Default 0 disables."
        ),
    )


# ---------------------------------------------------------------------------
# Extracted helpers
# ---------------------------------------------------------------------------


def _build_results_dir(output_dir: str, checkpoint: str, eval_train_data: bool) -> str:
    """Compute and create the results directory."""
    import os

    prefix = "eval_train_" if eval_train_data else "eval_"
    results_dir = os.path.join(output_dir, prefix + os.path.basename(checkpoint))
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _build_metric_configs(embed_key) -> dict:
    """Build metric_configs dict for MetricsEvaluator.compute()."""
    if embed_key and embed_key != "X_hvg":
        return {
            "discrimination_score": {
                "embed_key": embed_key,
            },
            "pearson_edistance": {
                "embed_key": embed_key,
                "n_jobs": -1,
            },
        }
    return {}


def _filter_shared_only(adata_pred, adata_real, data_module, logger, extra_pairs=None):
    """
    Filter adatas to perturbations shared between train and test.

    extra_pairs is a list of (pred, real) tuples to also filter with the same mask.
    Returns (adata_pred, adata_real, filtered_extra_pairs).
    """
    try:
        shared_perts = data_module.get_shared_perturbations()
        if len(shared_perts) == 0:
            logger.warning("No shared perturbations between train and test; skipping filtering.")
            return adata_pred, adata_real, extra_pairs

        logger.info(
            "Filtering to %d shared perturbations present in train ∩ test.",
            len(shared_perts),
        )
        mask = adata_pred.obs[data_module.pert_col].isin(shared_perts)
        before_n = adata_pred.n_obs
        adata_pred = adata_pred[mask].copy()
        adata_real = adata_real[mask].copy()

        if extra_pairs is not None:
            extra_pairs = [(p[mask].copy(), r[mask].copy()) for p, r in extra_pairs]

        logger.info(
            "Filtered cells: %d -> %d (kept only seen perturbations)",
            before_n,
            adata_pred.n_obs,
        )
        return adata_pred, adata_real, extra_pairs
    except Exception as e:
        logger.warning(
            "Failed to filter by shared perturbations (%s). Proceeding without filter.",
            str(e),
        )
        return adata_pred, adata_real, extra_pairs


def _run_cell_eval(
    adata_real,
    adata_pred,
    data_module,
    results_dir,
    profile,
    pdex_kwargs,
    embed_key,
    skip_de=False,
    write_csv=True,
):
    """
    Run MetricsEvaluator per cell type.

    Returns dict of {celltype: (results_df, agg_df)}.
    """
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    control_pert = data_module.get_control_pert()
    ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=data_module.cell_type_key)
    ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=data_module.cell_type_key)

    assert len(ct_split_real) == len(ct_split_pred), (
        f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"
    )

    metric_configs = _build_metric_configs(embed_key)
    all_results = {}

    for ct in ct_split_real.keys():
        real_ct = ct_split_real[ct]
        pred_ct = ct_split_pred[ct]

        evaluator = MetricsEvaluator(
            adata_pred=pred_ct,
            adata_real=real_ct,
            control_pert=control_pert,
            pert_col=data_module.pert_col,
            outdir=results_dir,
            prefix=ct,
            pdex_kwargs=pdex_kwargs,
            batch_size=2048,
            skip_de=skip_de,
        )
        results_df, agg_df = evaluator.compute(
            profile=profile,
            metric_configs=metric_configs,
            skip_metrics=["pearson_edistance", "clustering_agreement"],
            write_csv=write_csv,
        )
        all_results[ct] = (results_df, agg_df)

    return all_results


def _run_batched_de(
    h5_path,
    results_dir,
    data_module,
    eval_batch_size,
    pdex_kwargs,
    non_de_results,
    shared_perts,
    logger,
):
    """
    Run cell-level DE in batches from temporary h5py storage.

    non_de_results: dict of {celltype: (results_df, agg_df)} from non-DE evaluation.
    shared_perts: set of perturbation names to restrict to, or None for no filtering.
    Writes merged CSVs to results_dir.
    """
    import gc
    import os

    import anndata
    import h5py
    import numpy as np
    import pandas as pd
    import polars as pl
    from cell_eval import MetricsEvaluator

    # locking=False avoids POSIX file-lock issues on NFS / parallel filesystems
    h5f = h5py.File(h5_path, "r", locking=False)
    try:
        all_perts = h5f["pert_name"][:].astype(str)
        all_celltypes = h5f["cell_type"][:].astype(str)
        control = data_module.get_control_pert()

        unique_celltypes = sorted(set(all_celltypes))
        logger.info("Running batched cell-level DE for %d cell types.", len(unique_celltypes))

        for ct in unique_celltypes:
            ct_mask = all_celltypes == ct
            ct_perts = all_perts[ct_mask]
            unique_perts = sorted(set(ct_perts) - {control})

            if shared_perts is not None:
                unique_perts = [p for p in unique_perts if p in shared_perts]

            ct_safe = ct.replace("/", "-")

            if len(unique_perts) == 0:
                logger.info("Cell type '%s': no non-control perturbations, skipping DE.", ct)
                if ct in non_de_results:
                    results_df, agg_df = non_de_results[ct]
                    results_df.write_csv(os.path.join(results_dir, f"{ct_safe}_results.csv"))
                    agg_df.write_csv(os.path.join(results_dir, f"{ct_safe}_agg_results.csv"))
                continue

            n_batches = (len(unique_perts) + eval_batch_size - 1) // eval_batch_size
            all_de_results = []

            for batch_start in range(0, len(unique_perts), eval_batch_size):
                batch_perts = unique_perts[batch_start : batch_start + eval_batch_size]
                keep = set(batch_perts) | {control}
                sel = ct_mask & np.isin(all_perts, list(keep))
                indices = np.sort(np.where(sel)[0])

                batch_pred_X = h5f["X_pred"][indices]
                batch_real_X = h5f["X_real"][indices]
                batch_obs = pd.DataFrame(
                    {
                        data_module.pert_col: all_perts[indices],
                        data_module.cell_type_key: all_celltypes[indices],
                    }
                )

                adata_pred_batch = anndata.AnnData(X=batch_pred_X, obs=batch_obs)
                adata_real_batch = anndata.AnnData(X=batch_real_X, obs=batch_obs)

                evaluator = MetricsEvaluator(
                    adata_pred=adata_pred_batch,
                    adata_real=adata_real_batch,
                    control_pert=control,
                    pert_col=data_module.pert_col,
                    outdir=results_dir,
                    prefix=f"_debatch_{ct_safe}_{batch_start}",
                    pdex_kwargs=pdex_kwargs,
                    batch_size=2048,
                    skip_de=False,
                )
                de_results_df, _ = evaluator.compute(
                    profile="de",
                    write_csv=False,
                )
                all_de_results.append(de_results_df)

                # Clean up intermediate DE CSV files written by MetricsEvaluator
                batch_prefix = f"_debatch_{ct_safe}_{batch_start}"
                for suffix in ("_pred_de.csv", "_real_de.csv", "_results.csv", "_agg_results.csv"):
                    tmp_csv = os.path.join(results_dir, f"{batch_prefix}{suffix}")
                    if os.path.exists(tmp_csv):
                        os.remove(tmp_csv)

                del batch_pred_X, batch_real_X, evaluator, adata_pred_batch, adata_real_batch
                gc.collect()

                logger.info(
                    "Cell type '%s': processed DE batch %d/%d (%d perturbations).",
                    ct,
                    batch_start // eval_batch_size + 1,
                    n_batches,
                    len(batch_perts),
                )

            # Merge DE results with non-DE results for this cell type
            if all_de_results:
                merged_de = pl.concat(all_de_results)

                if ct in non_de_results:
                    non_de_df, non_de_agg = non_de_results[ct]

                    # Find the perturbation join key
                    de_cols = set(merged_de.columns)
                    non_de_cols = set(non_de_df.columns)
                    join_col = "perturbation"
                    if join_col not in de_cols:
                        join_col = data_module.pert_col
                    if join_col not in de_cols or join_col not in non_de_cols:
                        logger.warning(
                            "Cell type '%s': could not find join column for DE merge. "
                            "Writing DE and non-DE results separately.",
                            ct,
                        )
                        non_de_df.write_csv(os.path.join(results_dir, f"{ct_safe}_results.csv"))
                        non_de_agg.write_csv(os.path.join(results_dir, f"{ct_safe}_agg_results.csv"))
                        merged_de.write_csv(os.path.join(results_dir, f"{ct_safe}_de_results.csv"))
                        continue

                    # Drop duplicate columns from DE (except join column)
                    de_unique_cols = [c for c in merged_de.columns if c not in non_de_cols or c == join_col]
                    merged_de_unique = merged_de.select(de_unique_cols)

                    combined = non_de_df.join(merged_de_unique, on=join_col, how="left")
                    combined.write_csv(os.path.join(results_dir, f"{ct_safe}_results.csv"))
                    non_de_agg.write_csv(os.path.join(results_dir, f"{ct_safe}_agg_results.csv"))
                    logger.info(
                        "Cell type '%s': merged %d DE results with non-DE metrics.",
                        ct,
                        len(merged_de),
                    )
                else:
                    merged_de.write_csv(os.path.join(results_dir, f"{ct_safe}_de_results.csv"))
                    logger.info(
                        "Cell type '%s': wrote %d DE results (no non-DE results to merge).",
                        ct,
                        len(merged_de),
                    )
    finally:
        h5f.close()
        if os.path.exists(h5_path):
            os.remove(h5_path)
            logger.info("Removed temporary h5py file: %s", h5_path)


# ---------------------------------------------------------------------------
# Main predict entry point
# ---------------------------------------------------------------------------


def run_tx_predict(args: ap.ArgumentParser):
    import logging
    import os
    import sys

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp
    import torch
    import yaml

    # Cell-eval for metrics computation
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm
    from ._utils import normalize_batch_labels

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.predict_only and args.skip_adatas:
        logger.warning("Both --predict-only and --skip-adatas were set; no prediction artifacts will be written.")
    if args.profile == "anndata" and not args.skip_de:
        logger.warning(
            "--profile anndata does not disable DE computation by itself. "
            "Add --skip-de to skip DE and reduce memory/runtime."
        )
    if args.eval_batch_size > 0 and not args.pseudobulk:
        logger.warning("--eval-batch-size is only supported with --pseudobulk. Ignoring.")
        args.eval_batch_size = 0

    # --- Nested utility functions ---

    def clip_anndata_values(adata: anndata.AnnData, max_value: float, min_value: float = 0.0) -> None:
        """Clip adata.X values in-place to keep cell-eval scale checks happy."""
        if sp.issparse(adata.X):
            # Clip only the stored data to keep sparsity intact.
            if adata.X.data.size:
                np.clip(adata.X.data, min_value, max_value, out=adata.X.data)
                if hasattr(adata.X, "eliminate_zeros"):
                    adata.X.eliminate_zeros()
        else:
            np.clip(adata.X, min_value, max_value, out=adata.X)

    def get_batch_labels(candidates, batch_size: int):
        batch_labels = None
        for candidate in candidates:
            batch_labels = normalize_batch_labels(candidate, batch_size)
            if batch_labels is not None:
                break
        if batch_labels is None:
            batch_labels = ["None"] * batch_size
        return batch_labels

    def resolve_context_labels(batch: dict, batch_size: int, fallback):
        dataset_labels = None
        for key in ("dataset_name", "dataset"):
            if key in batch and batch.get(key) is not None:
                dataset_labels = normalize_batch_labels(batch.get(key), batch_size)
                if dataset_labels is not None:
                    break
        context_labels = normalize_batch_labels(fallback, batch_size) if fallback is not None else None

        if dataset_labels is not None and context_labels is not None:
            combined = [f"{ds}.{ct}" for ds, ct in zip(dataset_labels, context_labels)]
            return combined, "dataset_name+cell_type"
        if context_labels is not None:
            return context_labels, "cell_type"
        return None, None

    def ensure_list(values, batch_size: int):
        if isinstance(values, list):
            return values
        if isinstance(values, tuple):
            return list(values)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(values, np.ndarray):
            if values.ndim == 0:
                return [values.item()] * batch_size
            return values.tolist()
        return [values] * batch_size

    # -----------------------------------------------------------------------
    # 1. Load the config
    # -----------------------------------------------------------------------
    config_path = os.path.join(args.output_dir, "config.yaml")

    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    if args.toml:
        data_section = cfg.get("data")
        if data_section is None or "kwargs" not in data_section:
            raise KeyError("The loaded config does not contain data.kwargs, unable to override toml_config_path.")
        cfg["data"]["kwargs"]["toml_config_path"] = args.toml
        logger.info("Overriding data.kwargs.toml_config_path to %s", args.toml)

    # 2. Find run output directory & load data module
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.torch")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(f"Could not find data module at {data_module_path}?")
    data_module = PerturbationDataModule.load_state(data_module_path)
    if args.toml:
        if not os.path.exists(args.toml):
            raise FileNotFoundError(f"Could not find TOML config file at {args.toml}")
        from cell_load.config import ExperimentConfig

        logger.info("Reloading data module configuration from %s", args.toml)
        data_module.toml_config_path = args.toml
        data_module.config = ExperimentConfig.from_toml(args.toml)
        data_module.config.validate()
        data_module.train_datasets = []
        data_module.val_datasets = []
        data_module.test_datasets = []
        data_module._setup_global_maps()
    data_module.setup(stage="test")
    nb_loss_enabled = bool(cfg.get("model", {}).get("kwargs", {}).get("nb_loss", False))
    _OUTPUT_SPACE_ALIASES = {"hvg": "gene", "transcriptome": "all"}
    output_space = cfg.get("data", {}).get("kwargs", {}).get("output_space", "gene")
    output_space = _OUTPUT_SPACE_ALIASES.get(output_space.strip().lower(), output_space)
    if nb_loss_enabled and output_space == "embedding":
        raise ValueError(
            "model.kwargs.nb_loss=True is incompatible with data.kwargs.output_space='embedding'. "
            "Use output_space='gene' or output_space='all'."
        )
    if nb_loss_enabled and output_space not in {"gene", "all"}:
        raise ValueError(
            f"model.kwargs.nb_loss=True requires data.kwargs.output_space in {{'gene', 'all'}}; got {output_space!r}."
        )
    if nb_loss_enabled:
        resolved_is_log1p = bool(getattr(data_module, "is_log1p", cfg["data"]["kwargs"].get("is_log1p", False)))
        expected_exp_counts = resolved_is_log1p
        current_exp_counts = bool(getattr(data_module, "exp_counts", False))
        if current_exp_counts != expected_exp_counts:
            logger.warning(
                "nb_loss=True requires exp_counts to follow is_log1p. "
                "Resolved is_log1p=%s, overriding exp_counts %s -> %s.",
                resolved_is_log1p,
                current_exp_counts,
                expected_exp_counts,
            )
            data_module.exp_counts = expected_exp_counts
        if data_module.embed_key not in {None, "X_hvg"} and not bool(getattr(data_module, "store_raw_basal", False)):
            logger.warning(
                "nb_loss=True with embed_key=%r and store_raw_basal=False. "
                "NB library-size estimation will fall back to ctrl_cell_emb.",
                data_module.embed_key,
            )
        cfg["data"]["kwargs"]["is_log1p"] = resolved_is_log1p
        cfg["data"]["kwargs"]["exp_counts"] = expected_exp_counts
    resolved_exp_counts = bool(getattr(data_module, "exp_counts", cfg["data"]["kwargs"].get("exp_counts", False)))
    metrics_is_log1p = not (nb_loss_enabled or resolved_exp_counts)
    logger.info(
        "Metrics config: setting pdex is_log1p=%s (nb_loss=%s, exp_counts=%s)",
        metrics_is_log1p,
        nb_loss_enabled,
        resolved_exp_counts,
    )
    logger.info("Loaded data module from %s", data_module_path)

    # Seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # -----------------------------------------------------------------------
    # 3. Load the trained model
    # -----------------------------------------------------------------------
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # Determine model class and load
    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    # Import the correct model class
    if model_class_name.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel

    elif model_class_name.lower() in ["globalsimplesum", "perturb_mean"]:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_class_name.lower() in ["celltypemean", "context_mean"]:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_class_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    elif model_class_name.lower() == "pseudobulk":
        from ...tx.models.pseudobulk import PseudobulkPerturbationModel

        ModelClass = PseudobulkPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    model = ModelClass.load_from_checkpoint(checkpoint_path, weights_only=False, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    # -----------------------------------------------------------------------
    # 4. Run inference on test set
    # -----------------------------------------------------------------------
    data_module.setup(stage="test")
    if args.eval_train_data:
        test_loader = data_module.train_dataloader(test=True)
    else:
        test_loader = data_module.test_dataloader()

    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    num_cells = test_loader.batch_sampler.tot_num
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims["gene_dim"]
    hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions on test set using manual loop...")
    device = next(model.parameters()).device

    cfg_batch_col = cfg.get("data", {}).get("kwargs", {}).get("batch_col", None)
    batch_obs_key = cfg_batch_col or data_module.batch_col
    if batch_obs_key is None:
        batch_obs_key = "batch"

    store_raw_expression = (
        data_module.embed_key is not None
        and data_module.embed_key != "X_hvg"
        and cfg["data"]["kwargs"]["output_space"] == "gene"
    ) or (data_module.embed_key is not None and cfg["data"]["kwargs"]["output_space"] == "all")
    use_count_outputs = store_raw_expression or nb_loss_enabled
    if nb_loss_enabled and not store_raw_expression:
        logger.info(
            "nb_loss=True: forcing prediction artifacts to use NB count outputs even though "
            "store_raw_expression would otherwise be disabled."
        )

    results_dir = _build_results_dir(args.output_dir, args.checkpoint, args.eval_train_data)
    h5_path = None  # Set if h5py temp file is created for batched DE

    # -----------------------------------------------------------------------
    # 4a. Pseudobulk inference path
    # -----------------------------------------------------------------------
    if args.pseudobulk:
        logger.info("Pseudobulk enabled; aggregating by (context, perturbation).")

        pseudo_x_dim = None
        if use_count_outputs:
            if output_space == "gene":
                pseudo_x_dim = hvg_dim
            elif output_space == "all":
                pseudo_x_dim = gene_dim
            else:
                raise ValueError(f"Unsupported output_space for pseudobulk: {output_space}")

        # If outputs are log1p-scaled expression, aggregate in count space and convert
        # back to log1p only for eval matrices.
        aggregate_main_in_count_space = bool(
            (not use_count_outputs) and metrics_is_log1p and output_space != "embedding"
        )
        aggregate_gene_in_count_space = bool(use_count_outputs and metrics_is_log1p)
        if aggregate_main_in_count_space or aggregate_gene_in_count_space:
            logger.info(
                "Pseudobulk scale handling: detected log1p outputs; accumulating sums in count space via expm1."
            )
        else:
            logger.info("Pseudobulk scale handling: using direct summation (no expm1 conversion before aggregation).")

        # --- h5py setup for batched cell-level DE ---
        write_h5 = args.eval_batch_size > 0 and not args.predict_only and not args.skip_de
        h5f_writer = None
        h5_idx = 0

        if write_h5:
            import h5py

            # Determine gene dimension for h5py storage
            if use_count_outputs:
                h5_gene_dim = pseudo_x_dim
            elif output_space != "embedding":
                h5_gene_dim = output_dim
            else:
                logger.warning(
                    "--eval-batch-size with output_space='embedding' and no gene-level outputs: "
                    "cell-level DE requires gene expression data. Disabling batched DE."
                )
                write_h5 = False

        if write_h5:
            h5_path = os.path.join(results_dir, "_tmp_celllevel.h5")
            h5f_writer = h5py.File(h5_path, "w", locking=False)
            str_dt = h5py.string_dtype()
            chunk_rows = min(1024, num_cells)
            h5f_writer.create_dataset(
                "X_pred", shape=(num_cells, h5_gene_dim), dtype="float32",
                chunks=(chunk_rows, h5_gene_dim), compression="lzf",
            )
            h5f_writer.create_dataset(
                "X_real", shape=(num_cells, h5_gene_dim), dtype="float32",
                chunks=(chunk_rows, h5_gene_dim), compression="lzf",
            )
            h5f_writer.create_dataset("pert_name", shape=(num_cells,), dtype=str_dt)
            h5f_writer.create_dataset("cell_type", shape=(num_cells,), dtype=str_dt)
            h5f_writer.create_dataset("batch", shape=(num_cells,), dtype=str_dt)
            logger.info(
                "Created temporary h5py file for batched DE: %s (cells=%d, genes=%d)",
                h5_path, num_cells, h5_gene_dim,
            )

        pb_groups: dict[tuple[str, str], dict] = {}
        context_mode = None
        total_cells_seen = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch", file=sys.stderr)):
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                batch_preds = model.predict_step(batch, batch_idx, padded=False)

                batch_size = batch_preds["preds"].shape[0]
                total_cells_seen += batch_size

                batch_labels = get_batch_labels(
                    (
                        batch.get("batch_name"),
                        batch_preds.get("batch_name"),
                        batch_preds.get("batch"),
                    ),
                    batch_size,
                )

                pert_names = [str(x) for x in ensure_list(batch_preds.get("pert_name"), batch_size)]
                celltypes = [str(x) for x in ensure_list(batch_preds.get("celltype_name"), batch_size)]
                if len(pert_names) != batch_size or len(celltypes) != batch_size:
                    raise ValueError("Mismatch between batch size and pert/celltype metadata lengths.")

                context_labels, detected_mode = resolve_context_labels(batch, batch_size, celltypes)
                if context_labels is None:
                    raise ValueError(
                        "pseudobulk requires dataset_name + cell_type (preferred) or cell_type alone. "
                        f"Check data.kwargs.cell_type_key (current: {data_module.cell_type_key})."
                    )
                context_labels = [str(x) for x in context_labels]
                if context_mode is None:
                    context_mode = detected_mode
                    logger.info("Pseudobulk context source: %s", context_mode)
                elif detected_mode != context_mode:
                    raise ValueError(
                        f"Inconsistent context source during prediction: saw '{detected_mode}' after '{context_mode}'."
                    )

                batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
                batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)

                if aggregate_main_in_count_space:
                    batch_pred_pb_np = np.expm1(batch_pred_np.astype(np.float64, copy=False))
                    batch_real_pb_np = np.expm1(batch_real_np.astype(np.float64, copy=False))
                    np.clip(batch_pred_pb_np, a_min=0.0, a_max=None, out=batch_pred_pb_np)
                    np.clip(batch_real_pb_np, a_min=0.0, a_max=None, out=batch_real_pb_np)
                else:
                    batch_pred_pb_np = batch_pred_np
                    batch_real_pb_np = batch_real_np

                batch_real_gene_np = None
                batch_gene_pred_np = None
                batch_real_gene_pb_np = None
                batch_gene_pred_pb_np = None
                if use_count_outputs:
                    batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                    batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                    if aggregate_gene_in_count_space:
                        batch_real_gene_pb_np = np.expm1(batch_real_gene_np.astype(np.float64, copy=False))
                        batch_gene_pred_pb_np = np.expm1(batch_gene_pred_np.astype(np.float64, copy=False))
                        np.clip(batch_real_gene_pb_np, a_min=0.0, a_max=None, out=batch_real_gene_pb_np)
                        np.clip(batch_gene_pred_pb_np, a_min=0.0, a_max=None, out=batch_gene_pred_pb_np)
                    else:
                        batch_real_gene_pb_np = batch_real_gene_np
                        batch_gene_pred_pb_np = batch_gene_pred_np

                # --- Write cell-level data to h5py for batched DE ---
                if write_h5 and h5f_writer is not None:
                    if use_count_outputs:
                        h5f_writer["X_pred"][h5_idx : h5_idx + batch_size] = batch_gene_pred_np
                        h5f_writer["X_real"][h5_idx : h5_idx + batch_size] = batch_real_gene_np
                    else:
                        h5f_writer["X_pred"][h5_idx : h5_idx + batch_size] = batch_pred_np
                        h5f_writer["X_real"][h5_idx : h5_idx + batch_size] = batch_real_np
                    h5f_writer["pert_name"][h5_idx : h5_idx + batch_size] = pert_names
                    h5f_writer["cell_type"][h5_idx : h5_idx + batch_size] = celltypes
                    h5f_writer["batch"][h5_idx : h5_idx + batch_size] = batch_labels
                    h5_idx += batch_size

                # --- Accumulate pseudobulk sums ---
                group_to_indices: dict[tuple[str, str], list[int]] = {}
                for idx in range(batch_size):
                    key = (context_labels[idx], pert_names[idx])
                    group_to_indices.setdefault(key, []).append(idx)

                for (context_label, pert_name), idxs in group_to_indices.items():
                    idx_arr = np.asarray(idxs, dtype=np.int64)
                    first_idx = int(idx_arr[0])
                    current_celltype = celltypes[first_idx]
                    current_batch = str(batch_labels[first_idx])

                    entry = pb_groups.get((context_label, pert_name))
                    if entry is None:
                        entry = {
                            "context": context_label,
                            "pert_name": pert_name,
                            "celltype_name": current_celltype,
                            "batch_name": current_batch,
                            "count": 0,
                            "pred_sum": np.zeros(output_dim, dtype=np.float64),
                            "real_sum": np.zeros(output_dim, dtype=np.float64),
                            "x_hvg_sum": np.zeros(pseudo_x_dim, dtype=np.float64) if use_count_outputs else None,
                            "counts_pred_sum": np.zeros(pseudo_x_dim, dtype=np.float64) if use_count_outputs else None,
                        }
                        pb_groups[(context_label, pert_name)] = entry
                    elif entry["celltype_name"] != current_celltype:
                        raise ValueError(
                            f"Inconsistent cell type for context/pert pair ({context_label}, {pert_name}): "
                            f"saw '{current_celltype}' after '{entry['celltype_name']}'."
                        )

                    entry["count"] += int(idx_arr.size)
                    entry["pred_sum"] += batch_pred_pb_np[idx_arr].sum(axis=0, dtype=np.float64)
                    entry["real_sum"] += batch_real_pb_np[idx_arr].sum(axis=0, dtype=np.float64)
                    if use_count_outputs:
                        entry["x_hvg_sum"] += batch_real_gene_pb_np[idx_arr].sum(axis=0, dtype=np.float64)
                        entry["counts_pred_sum"] += batch_gene_pred_pb_np[idx_arr].sum(axis=0, dtype=np.float64)

        # Close h5py writer
        if h5f_writer is not None:
            h5f_writer.flush()
            h5f_writer.close()
            h5f_writer = None
            logger.info("Closed h5py writer (%d cells written).", h5_idx)

        if len(pb_groups) == 0:
            logger.warning("No pseudobulk groups were generated. Exiting.")
            if h5_path and os.path.exists(h5_path):
                os.remove(h5_path)
            sys.exit(0)

        logger.info(
            "Built %d pseudobulk groups from %d cells.",
            len(pb_groups),
            total_cells_seen,
        )

        group_entries = list(pb_groups.values())
        group_entries.sort(key=lambda x: (str(x["celltype_name"]), str(x["context"]), str(x["pert_name"])))
        n_groups = len(group_entries)

        # Keep both views:
        # - sum matrices for persisted pseudobulk outputs
        # - eval matrices for cell-eval (log1p(sum) when inputs are log1p, mean otherwise)
        pred_bulk_sum = np.empty((n_groups, output_dim), dtype=np.float32)
        real_bulk_sum = np.empty((n_groups, output_dim), dtype=np.float32)
        pred_bulk_eval = np.empty((n_groups, output_dim), dtype=np.float32)
        real_bulk_eval = np.empty((n_groups, output_dim), dtype=np.float32)
        pred_x_sum = np.empty((n_groups, pseudo_x_dim), dtype=np.float32) if use_count_outputs else None
        real_x_sum = np.empty((n_groups, pseudo_x_dim), dtype=np.float32) if use_count_outputs else None
        pred_x_eval = np.empty((n_groups, pseudo_x_dim), dtype=np.float32) if use_count_outputs else None
        real_x_eval = np.empty((n_groups, pseudo_x_dim), dtype=np.float32) if use_count_outputs else None

        reserved_obs_keys = {
            str(data_module.pert_col),
            str(data_module.cell_type_key),
            str(batch_obs_key),
        }
        if data_module.batch_col:
            reserved_obs_keys.add(str(data_module.batch_col))

        pseudobulk_context_key = "pseudobulk_context"
        while pseudobulk_context_key in reserved_obs_keys:
            pseudobulk_context_key = f"_{pseudobulk_context_key}"
        pseudobulk_n_cells_key = "pseudobulk_n_cells"
        while pseudobulk_n_cells_key in reserved_obs_keys or pseudobulk_n_cells_key == pseudobulk_context_key:
            pseudobulk_n_cells_key = f"_{pseudobulk_n_cells_key}"

        obs_dict = {
            data_module.pert_col: [],
            data_module.cell_type_key: [],
            batch_obs_key: [],
            pseudobulk_context_key: [],
            pseudobulk_n_cells_key: [],
        }
        if data_module.batch_col and data_module.batch_col != batch_obs_key:
            obs_dict[data_module.batch_col] = []

        for idx, entry in enumerate(group_entries):
            count = int(entry["count"])
            denom = float(count)
            pred_bulk_sum[idx, :] = entry["pred_sum"].astype(np.float32, copy=False)
            real_bulk_sum[idx, :] = entry["real_sum"].astype(np.float32, copy=False)
            if aggregate_main_in_count_space:
                pred_bulk_eval[idx, :] = np.log1p(entry["pred_sum"]).astype(np.float32, copy=False)
                real_bulk_eval[idx, :] = np.log1p(entry["real_sum"]).astype(np.float32, copy=False)
            else:
                pred_bulk_eval[idx, :] = (entry["pred_sum"] / denom).astype(np.float32, copy=False)
                real_bulk_eval[idx, :] = (entry["real_sum"] / denom).astype(np.float32, copy=False)
            if use_count_outputs:
                pred_x_sum[idx, :] = entry["counts_pred_sum"].astype(np.float32, copy=False)
                real_x_sum[idx, :] = entry["x_hvg_sum"].astype(np.float32, copy=False)
                if aggregate_gene_in_count_space:
                    pred_x_eval[idx, :] = np.log1p(entry["counts_pred_sum"]).astype(np.float32, copy=False)
                    real_x_eval[idx, :] = np.log1p(entry["x_hvg_sum"]).astype(np.float32, copy=False)
                else:
                    pred_x_eval[idx, :] = (entry["counts_pred_sum"] / denom).astype(np.float32, copy=False)
                    real_x_eval[idx, :] = (entry["x_hvg_sum"] / denom).astype(np.float32, copy=False)

            obs_dict[data_module.pert_col].append(entry["pert_name"])
            obs_dict[data_module.cell_type_key].append(entry["celltype_name"])
            obs_dict[batch_obs_key].append(entry["batch_name"])
            obs_dict[pseudobulk_context_key].append(entry["context"])
            obs_dict[pseudobulk_n_cells_key].append(count)
            if data_module.batch_col and data_module.batch_col != batch_obs_key:
                obs_dict[data_module.batch_col].append(entry["batch_name"])

        obs = pd.DataFrame(obs_dict)
        if use_count_outputs:
            # Persisted outputs: summed pseudobulks.
            adata_pred = anndata.AnnData(X=pred_x_sum, obs=obs)
            adata_real = anndata.AnnData(X=real_x_sum, obs=obs)
            if data_module.embed_key is not None:
                adata_pred.obsm[data_module.embed_key] = pred_bulk_sum
                adata_real.obsm[data_module.embed_key] = real_bulk_sum

            # Metric inputs: log1p(sum) pseudobulks for log1p outputs, mean otherwise.
            adata_pred_eval = anndata.AnnData(X=pred_x_eval, obs=obs.copy())
            adata_real_eval = anndata.AnnData(X=real_x_eval, obs=obs.copy())
            if data_module.embed_key is not None:
                adata_pred_eval.obsm[data_module.embed_key] = pred_bulk_eval
                adata_real_eval.obsm[data_module.embed_key] = real_bulk_eval
        else:
            # Persisted outputs: summed pseudobulks.
            adata_pred = anndata.AnnData(X=pred_bulk_sum, obs=obs)
            adata_real = anndata.AnnData(X=real_bulk_sum, obs=obs)
            # Metric inputs: log1p(sum) pseudobulks for log1p outputs, mean otherwise.
            adata_pred_eval = anndata.AnnData(X=pred_bulk_eval, obs=obs.copy())
            adata_real_eval = anndata.AnnData(X=real_bulk_eval, obs=obs.copy())

        persist_mode = (
            "sum(expm1(log1p(x)))" if (aggregate_main_in_count_space or aggregate_gene_in_count_space) else "sum(x)"
        )
        eval_mode = (
            "log1p(sum(expm1(log1p(x))))"
            if (aggregate_main_in_count_space or aggregate_gene_in_count_space)
            else "mean(x)"
        )
        pseudobulk_meta = {
            "persisted_aggregation": persist_mode,
            "eval_aggregation": eval_mode,
            "n_cells_obs_column": pseudobulk_n_cells_key,
        }
        adata_pred.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)
        adata_real.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)
        adata_pred_eval.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)
        adata_real_eval.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)

    # -------------------------------------------------------------------
    # 4b. Cell-level inference path
    # -------------------------------------------------------------------
    else:
        final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
        final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

        final_X_hvg = None
        final_pert_cell_counts_preds = None
        if use_count_outputs:
            # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
            if output_space == "gene":
                final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, hvg_dim), dtype=np.float32)
            if output_space == "all":
                final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, gene_dim), dtype=np.float32)

        current_idx = 0

        # Initialize aggregation variables directly
        all_pert_names = []
        all_celltypes = []
        all_gem_groups = []
        all_pert_barcodes = []
        all_ctrl_barcodes = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(test_loader, desc="Predicting", unit="batch", file=sys.stderr)
            ):
                # Move each tensor in the batch to the model's device
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

                # Get predictions
                batch_preds = model.predict_step(batch, batch_idx, padded=False)

                # Extract metadata and data directly from batch_preds
                # Handle pert_name
                if isinstance(batch_preds["pert_name"], list):
                    all_pert_names.extend(batch_preds["pert_name"])
                else:
                    all_pert_names.append(batch_preds["pert_name"])

                if "pert_cell_barcode" in batch_preds:
                    if isinstance(batch_preds["pert_cell_barcode"], list):
                        all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.extend(batch_preds["ctrl_cell_barcode"])
                    else:
                        all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.append(batch_preds["ctrl_cell_barcode"])

                # Handle celltype_name
                if isinstance(batch_preds["celltype_name"], list):
                    all_celltypes.extend(batch_preds["celltype_name"])
                else:
                    all_celltypes.append(batch_preds["celltype_name"])

                batch_size = batch_preds["preds"].shape[0]

                # Handle gem_group - prefer human-readable batch names when available
                batch_labels = get_batch_labels(
                    (
                        batch.get("batch_name"),
                        batch_preds.get("batch_name"),
                        batch_preds.get("batch"),
                    ),
                    batch_size,
                )
                all_gem_groups.extend(batch_labels)

                batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
                batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)
                final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
                final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
                current_idx += batch_size

                # Handle X_hvg for HVG space ground truth
                if final_X_hvg is not None:
                    batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                    final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

                # Handle decoded gene predictions if available
                if final_pert_cell_counts_preds is not None:
                    batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                    final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

        logger.info("Creating anndatas from predictions from manual loop...")

        # Build pandas DataFrame for obs and var
        logger.info("Resolved batch obs key: %s", batch_obs_key)
        df_dict = {
            data_module.pert_col: all_pert_names,
            data_module.cell_type_key: all_celltypes,
            batch_obs_key: all_gem_groups,
        }
        if data_module.batch_col and data_module.batch_col != batch_obs_key:
            logger.info("Adding explicit batch column to output obs: %s", data_module.batch_col)
            df_dict[data_module.batch_col] = all_gem_groups

        if len(all_pert_barcodes) > 0:
            df_dict["pert_cell_barcode"] = all_pert_barcodes
            df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

        obs = pd.DataFrame(df_dict)

        if final_X_hvg is not None:
            # Create adata for predictions - using the decoded gene expression values
            adata_pred = anndata.AnnData(X=final_pert_cell_counts_preds, obs=obs)
            # Create adata for real - using the true gene expression values
            adata_real = anndata.AnnData(X=final_X_hvg, obs=obs)

            # add the embedding predictions
            if data_module.embed_key is not None:
                adata_pred.obsm[data_module.embed_key] = final_preds
                adata_real.obsm[data_module.embed_key] = final_reals
                logger.info(f"Added predicted embeddings to adata.obsm['{data_module.embed_key}']")
        else:
            # Create adata for predictions - model was trained on gene expression space already
            adata_pred = anndata.AnnData(X=final_preds, obs=obs)
            # Create adata for real - using the true gene expression values
            adata_real = anndata.AnnData(X=final_reals, obs=obs)

        # Cell-level: eval adatas are the same as save adatas
        adata_pred_eval = adata_pred
        adata_real_eval = adata_real

    # ===================================================================
    # 5. Shared post-processing: clip, filter, save, evaluate
    # ===================================================================

    # --- Clip ---
    if nb_loss_enabled:
        logger.info(
            "nb_loss=True in run config; keeping outputs unchanged and skipping metric clipping."
        )
    else:
        clip_anndata_values(adata_pred_eval, max_value=14.0)
        clip_anndata_values(adata_real_eval, max_value=14.0)
        logger.info("Clipped eval data X values to [0.0, 14.0] for cell-eval metrics.")

    # --- Filter shared_only ---
    shared_perts_set = None
    if args.shared_only:
        if args.pseudobulk:
            adata_pred, adata_real, extra = _filter_shared_only(
                adata_pred, adata_real, data_module, logger,
                extra_pairs=[(adata_pred_eval, adata_real_eval)],
            )
            if extra is not None:
                adata_pred_eval, adata_real_eval = extra[0]
            # Capture shared_perts for batched DE filtering
            try:
                shared_perts_set = set(data_module.get_shared_perturbations())
            except Exception:
                pass
        else:
            adata_pred, adata_real, _ = _filter_shared_only(
                adata_pred, adata_real, data_module, logger,
            )
            adata_pred_eval = adata_pred
            adata_real_eval = adata_real

    # --- Save AnnDatas ---
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")
    if args.skip_adatas:
        logger.info("Skipping AnnData writes (--skip-adatas).")
    else:
        adata_pred.write_h5ad(adata_pred_path)
        adata_real.write_h5ad(adata_real_path)
        logger.info(f"Saved adata_pred to {adata_pred_path}")
        logger.info(f"Saved adata_real to {adata_real_path}")

    # --- Evaluate ---
    if not args.predict_only:
        logger.info("Computing metrics using cell-eval...")

        pdex_kwargs = dict(exp_post_agg=True, is_log1p=metrics_is_log1p)

        if args.pseudobulk and args.eval_batch_size > 0 and not args.skip_de and h5_path is not None:
            # Batched DE mode:
            # 1. Non-DE metrics from pseudobulk (skip_de=True, no CSV yet)
            logger.info("Batched DE mode: computing non-DE metrics from pseudobulk first...")
            non_de_results = _run_cell_eval(
                adata_real_eval, adata_pred_eval, data_module, results_dir,
                args.profile, pdex_kwargs, data_module.embed_key,
                skip_de=True, write_csv=False,
            )
            # 2. Batched cell-level DE from h5py
            logger.info("Running batched cell-level DE with batch_size=%d...", args.eval_batch_size)
            _run_batched_de(
                h5_path, results_dir, data_module, args.eval_batch_size,
                pdex_kwargs, non_de_results, shared_perts_set, logger,
            )
            h5_path = None  # Already cleaned up by _run_batched_de
        else:
            # Standard evaluation (all metrics together)
            _run_cell_eval(
                adata_real_eval, adata_pred_eval, data_module, results_dir,
                args.profile, pdex_kwargs, data_module.embed_key,
                skip_de=args.skip_de,
            )

    # Clean up h5py temp file if it wasn't used (e.g., predict_only or skip_de)
    if h5_path is not None and os.path.exists(h5_path):
        os.remove(h5_path)
        logger.info("Cleaned up unused temporary h5py file: %s", h5_path)
