"""Core logic for H5AD normalization and transformation for TX training.

Ported from arc-bench normalize_transform with additions for STATE
(HVG selection, progress callbacks, cancellation support).
"""

import logging
from pathlib import Path
from typing import Any, Callable

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from .discovery import compute_output_path, discover_h5ad_files_with_exclusions, force_release_memory
from .schemas import PreprocessTrainConfig, PreprocessTrainResult, TransformStats

logger = logging.getLogger(__name__)

NDArrayFloat = np.ndarray[Any, np.dtype[np.floating[Any]]]

CANONICAL_PERTURBATION_COL = "perturbation"
CANONICAL_CONTROL_LABEL = "control"


class PreprocessCancelledError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Sparse matrix utilities
# ---------------------------------------------------------------------------

def ensure_csr(adata: ad.AnnData) -> None:
    """Ensure sparse X is stored as CSR."""
    X = adata.X
    if sp.issparse(X) and not sp.isspmatrix_csr(X):
        adata.X = X.tocsr()


# ---------------------------------------------------------------------------
# Log1p reversal
# ---------------------------------------------------------------------------

def apply_expm1_if_needed(adata: ad.AnnData, already_log1p: bool) -> None:
    """Undo log1p in X when requested, keeping sparse matrices sparse."""
    if not already_log1p:
        return
    X = adata.X
    if X is None:
        raise ValueError("AnnData.X is None")
    if sp.issparse(X):
        X_csr = X.tocsr(copy=True)
        X_csr.data = np.expm1(X_csr.data)
        adata.X = X_csr
    else:
        adata.X = np.expm1(np.asarray(X))


# ---------------------------------------------------------------------------
# Gene set alignment
# ---------------------------------------------------------------------------

def load_gene_set(gene_set_path: Path) -> list[str]:
    """Load ordered gene names from a .npy file."""
    gene_names = np.load(gene_set_path, allow_pickle=True)
    if gene_names.ndim != 1:
        gene_names = gene_names.ravel()
    if gene_names.dtype.kind in {"S", "O"}:
        gene_names = gene_names.astype(str)
    gene_list = gene_names.tolist()
    if not gene_list:
        raise ValueError(f"No genes found in {gene_set_path}")
    return gene_list


def _normalize_gene_label(value: object) -> str:
    """Normalize gene labels to strings, decoding bytes when needed."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    return str(value)


def select_gene_labels(
    adata: ad.AnnData,
    gene_set: list[str],
) -> tuple[list[str], str, int]:
    """Select the gene label source with the highest overlap to gene_set."""
    gene_set_set = set(gene_set)
    best_name = "var_names"
    best_labels = [_normalize_gene_label(v) for v in adata.var_names]
    best_overlap = len(set(best_labels) & gene_set_set)

    for col in adata.var.columns:
        values = adata.var[col].tolist()
        labels = [_normalize_gene_label(v) for v in values]
        overlap = len(set(labels) & gene_set_set)
        if overlap > best_overlap:
            best_name = col
            best_labels = labels
            best_overlap = overlap

    if best_overlap == 0:
        raise ValueError("No overlap between gene_set and any var column or var_names.")
    return best_labels, best_name, best_overlap


def align_to_gene_set(
    adata: ad.AnnData,
    gene_set: list[str],
) -> tuple[ad.AnnData, str, int]:
    """Align X/var to a provided gene set in the exact order.

    Missing genes are added as all-zero columns to match the target ordering.
    """
    source_genes, source_name, overlap = select_gene_labels(adata, gene_set)

    gene_to_idx = {g: i for i, g in enumerate(gene_set)}
    target_positions = np.fromiter(
        (gene_to_idx.get(g, -1) for g in source_genes),
        dtype=np.int64,
        count=adata.n_vars,
    )

    X_raw = adata.X
    if X_raw is None:
        raise ValueError("AnnData.X is None")
    if sp.issparse(X_raw):
        X_csr = X_raw.tocsr()
    else:
        X_csr = sp.csr_matrix(np.asarray(X_raw))

    n_obs = adata.n_obs
    n_target = len(gene_set)

    if X_csr.nnz == 0:
        new_X = sp.csr_matrix((n_obs, n_target), dtype=X_csr.dtype)
    else:
        indices = X_csr.indices
        data = X_csr.data
        indptr = X_csr.indptr

        mapped = target_positions[indices]
        keep = mapped >= 0

        if not keep.any():
            new_X = sp.csr_matrix((n_obs, n_target), dtype=X_csr.dtype)
        else:
            keep_counts = np.add.reduceat(keep.astype(np.int64), indptr[:-1])
            new_indptr = np.empty_like(indptr)
            new_indptr[0] = 0
            np.cumsum(keep_counts, out=new_indptr[1:])

            new_data = data[keep]
            new_indices = mapped[keep]

            new_X = sp.csr_matrix(
                (new_data, new_indices, new_indptr),
                shape=(n_obs, n_target),
            )
            new_X.sort_indices()
            new_X.sum_duplicates()

    # Reindex var to ensure exact ordering and fill boolean columns.
    var = adata.var.copy()
    var.index = source_genes
    if not var.index.is_unique:
        var = var[~var.index.duplicated(keep="first")]
    bool_cols = var.select_dtypes(include=["bool"]).columns.tolist()
    var = var.reindex(gene_set)
    if bool_cols:
        for col in bool_cols:
            var[col] = var[col].fillna(False).astype(bool)

    aligned = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=var)
    if adata.uns:
        aligned.uns = adata.uns.copy()
    if adata.obsm:
        aligned.obsm = adata.obsm.copy()
    if adata.obsp:
        aligned.obsp = adata.obsp.copy()

    return aligned, source_name, overlap


# ---------------------------------------------------------------------------
# Perturbation / context / batch standardization
# ---------------------------------------------------------------------------

def standardize_perturbation_fields(
    adata: ad.AnnData,
    perturbation_col: str,
    control_perturbation: str,
) -> tuple[str, str]:
    """Ensure perturbation/control naming matches canonical expectations."""
    if CANONICAL_PERTURBATION_COL in adata.obs.columns:
        canonical_col = CANONICAL_PERTURBATION_COL
    elif perturbation_col in adata.obs.columns:
        adata.obs[CANONICAL_PERTURBATION_COL] = adata.obs[perturbation_col].astype(str)
        canonical_col = CANONICAL_PERTURBATION_COL
    else:
        return perturbation_col, control_perturbation

    adata.obs[canonical_col] = adata.obs[canonical_col].astype(str)
    if control_perturbation != CANONICAL_CONTROL_LABEL:
        adata.obs[canonical_col] = adata.obs[canonical_col].replace(
            {control_perturbation: CANONICAL_CONTROL_LABEL}
        )
    adata.obs[canonical_col] = adata.obs[canonical_col].astype("category")
    return canonical_col, CANONICAL_CONTROL_LABEL


def apply_context_and_batch(
    adata: ad.AnnData,
    context_col: str | None,
    batch_col: str | None,
) -> None:
    """Populate canonical context/batch columns from provided obs fields."""
    obs = adata.obs
    if batch_col is not None and context_col is None:
        raise ValueError("context_col is required when batch_col is provided")
    if context_col is not None:
        if context_col not in obs.columns:
            raise KeyError(f"Column '{context_col}' not found in adata.obs")
        obs["context"] = obs[context_col].astype(str)
    if batch_col is not None:
        if batch_col not in obs.columns:
            raise KeyError(f"Column '{batch_col}' not found in adata.obs")
        obs["batch_col"] = obs[batch_col].astype(str) + "_" + obs["context"].astype(str)
    if "context" in obs.columns:
        obs["context"] = obs["context"].astype("category")
    if "batch_col" in obs.columns:
        obs["batch_col"] = obs["batch_col"].astype("category")


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_adata_by_columns(adata: ad.AnnData, columns: list[str]) -> ad.AnnData:
    """Sort cells by obs columns, returning a reordered copy when needed."""
    if not columns:
        return adata
    if adata.obs_names.has_duplicates:
        adata = adata.copy()
        adata.obs_names_make_unique()
    missing = [col for col in columns if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"Columns not found in adata.obs: {missing}")
    sort_df = adata.obs.loc[:, columns].copy()
    for col in columns:
        sort_df[col] = sort_df[col].astype(str)
    order = sort_df.sort_values(by=columns, kind="mergesort").index
    if order.equals(adata.obs_names):
        return adata
    try:
        return adata[order].copy()
    except (TypeError, ValueError):
        # Work around AnnData/SciPy sparse indexing failures when reordering
        # pairwise matrices in obsp for some large datasets.
        if not adata.obsp:
            raise

        order_pos = adata.obs_names.get_indexer(order)
        if np.any(order_pos < 0):
            raise ValueError("Failed to map sorted obs index to positional indices")

        obsp_items = {key: value for key, value in adata.obsp.items()}
        for key in list(adata.obsp.keys()):
            del adata.obsp[key]

        try:
            sorted_adata = adata[order_pos].copy()
        finally:
            for key, value in obsp_items.items():
                adata.obsp[key] = value

        for key, value in obsp_items.items():
            sorted_adata.obsp[key] = value[order_pos][:, order_pos]
        return sorted_adata


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def downsample_counts(
    adata: ad.AnnData,
    frac: float,
    seed: int,
) -> ad.AnnData:
    """Downsample raw counts using binomial sampling."""
    if frac >= 1.0:
        return adata.copy()

    rng = np.random.default_rng(seed)
    X_raw = adata.X
    if X_raw is None:
        raise ValueError("AnnData.X is None, cannot downsample")

    result = adata.copy()

    if sp.issparse(X_raw):
        X_sparse = result.X.tocsr()
        X_sparse.data = rng.binomial(X_sparse.data.astype(np.int64), frac).astype(np.float32)
        X_sparse.eliminate_zeros()
        result.X = X_sparse
    else:
        X = np.asarray(X_raw)
        X_int = X.astype(np.int64)
        result.X = rng.binomial(X_int, frac).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Knockdown efficiency / log deviation
# ---------------------------------------------------------------------------

def compute_control_baseline(
    adata: ad.AnnData,
    perturbation_col: str,
    control_perturbation: str,
) -> NDArrayFloat:
    """Compute mean expression of control cells per gene."""
    if perturbation_col not in adata.obs.columns:
        raise KeyError(f"Column '{perturbation_col}' not found in adata.obs")

    control_mask = adata.obs[perturbation_col] == control_perturbation
    n_control = control_mask.sum()

    if n_control == 0:
        raise ValueError(
            f"No cells found with {perturbation_col}='{control_perturbation}'. "
            f"Available values: {sorted(adata.obs[perturbation_col].unique()[:10])}"
        )

    control_adata = adata[control_mask]
    X_raw = control_adata.X
    if X_raw is None:
        raise ValueError("AnnData.X is None")

    if sp.issparse(X_raw):
        X_sparse = X_raw.tocsr()
        result: NDArrayFloat = np.asarray(X_sparse.mean(axis=0)).ravel().astype(np.float32)
    else:
        result = np.mean(np.asarray(X_raw), axis=0).astype(np.float32)
    return result


def compute_knockdown_efficiency(
    adata: ad.AnnData,
    control_baseline: NDArrayFloat,
    perturbation_col: str,
    control_perturbation: str,
    eps: float = 1e-8,
) -> NDArrayFloat:
    """Compute knockdown efficiency for each cell's target gene.

    KD = 1 - (x_target / (mu_control_target + eps))
    Computed BEFORE log1p. Control cells get NaN.
    """
    n_cells = adata.n_obs
    efficiency = np.full(n_cells, np.nan, dtype=np.float32)

    X_raw = adata.X
    if X_raw is None:
        raise ValueError("AnnData.X is None")

    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    perturbations = adata.obs[perturbation_col].values
    unique_perts = np.unique(perturbations)

    for target_gene in unique_perts:
        if target_gene == control_perturbation or target_gene not in gene_to_idx:
            continue

        gene_idx = gene_to_idx[target_gene]
        pert_mask = perturbations == target_gene

        expr_slice = X_raw[pert_mask, gene_idx]
        if sp.issparse(expr_slice):
            expr = np.asarray(expr_slice.toarray()).ravel()
        else:
            expr = np.asarray(expr_slice).ravel()

        mu_control = control_baseline[gene_idx]
        efficiency[pert_mask] = 1.0 - (expr / (mu_control + eps))

    return efficiency


def compute_log_deviation(
    adata: ad.AnnData,
    control_baseline_log: NDArrayFloat,
    perturbation_col: str,
    control_perturbation: str,
) -> NDArrayFloat:
    """Compute log fold change for each cell's target gene.

    FC = log1p(x_target) - log1p(mu_control_target)
    Computed AFTER log1p. Control cells get NaN.
    """
    n_cells = adata.n_obs
    log_fc = np.full(n_cells, np.nan, dtype=np.float32)

    X_raw = adata.X
    if X_raw is None:
        raise ValueError("AnnData.X is None")

    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    perturbations = adata.obs[perturbation_col].values
    unique_perts = np.unique(perturbations)

    for target_gene in unique_perts:
        if target_gene == control_perturbation or target_gene not in gene_to_idx:
            continue

        gene_idx = gene_to_idx[target_gene]
        pert_mask = perturbations == target_gene

        expr_slice = X_raw[pert_mask, gene_idx]
        if sp.issparse(expr_slice):
            expr_log = np.asarray(expr_slice.toarray()).ravel()
        else:
            expr_log = np.asarray(expr_slice).ravel()

        log_fc[pert_mask] = expr_log - control_baseline_log[gene_idx]

    return log_fc


# ---------------------------------------------------------------------------
# Per-file processing pipeline
# ---------------------------------------------------------------------------

def normalize_log_transform_single(
    input_path: Path,
    output_path: Path,
    config: PreprocessTrainConfig,
    file_seed: int,
    gene_set: list[str] | None,
) -> TransformStats | None:
    """Process a single H5AD file: normalize, transform, add metrics.

    Pipeline:
    1. Load H5AD
    2. Optional gene alignment to gene_set
    3. Standardize perturbation fields
    4. Apply context/batch columns
    5. Sort cells by specified columns
    6. Optional expm1 (undo log1p)
    7. Optional downsampling
    8. Normalize total counts
    9. Compute knockdown efficiency (BEFORE log1p)
    10. Apply log1p transformation
    11. Compute log deviation (AFTER log1p)
    12. Optional HVG selection
    13. Write output
    """
    try:
        adata = ad.read_h5ad(input_path)
    except Exception as e:
        logger.warning(f"Failed to read {input_path.name} - {e}")
        return None

    logger.info(f"Loaded {input_path.name} with {adata.n_obs} cells, {adata.n_vars} genes")

    # Gene alignment
    if gene_set is not None:
        adata, source_name, overlap = align_to_gene_set(adata, gene_set)
        logger.info(
            f"Aligned genes to gene_set ({adata.n_vars} genes), source={source_name}, "
            f"overlap={overlap}"
        )
        ensure_csr(adata)

    # Standardize perturbation fields
    perturbation_col, control_label = standardize_perturbation_fields(
        adata, config.perturbation_col, config.control_perturbation
    )

    # Context/batch columns
    apply_context_and_batch(adata, config.context_col, config.batch_col)

    # Sort
    if config.sort_by:
        adata = sort_adata_by_columns(adata, config.sort_by)
        logger.info(f"Sorted cells by {config.sort_by}")

    # Count control and perturbed cells
    if perturbation_col in adata.obs.columns:
        control_mask = adata.obs[perturbation_col] == control_label
        n_control = int(control_mask.sum())
        n_perturbed = adata.n_obs - n_control
    else:
        logger.warning(f"Column '{perturbation_col}' not found, skipping metrics")
        n_control = 0
        n_perturbed = adata.n_obs

    # Undo log1p if needed
    if config.already_log1p:
        apply_expm1_if_needed(adata, config.already_log1p)
        logger.info("Applied expm1 to restore counts (already_log1p=true)")
        ensure_csr(adata)

    # Downsampling
    if config.downsample_frac < 1.0:
        adata = downsample_counts(adata, config.downsample_frac, file_seed)
        logger.info(f"Downsampled to {config.downsample_frac:.1%} of counts")
        ensure_csr(adata)

    # Normalize total counts
    if config.target_sum is None:
        sc.pp.normalize_total(adata, target_sum=None)
        logger.info("Normalized to median library size (scanpy default)")
    else:
        sc.pp.normalize_total(adata, target_sum=config.target_sum)
        logger.info(f"Normalized to target_sum={config.target_sum}")
    ensure_csr(adata)

    # Compute knockdown efficiency (BEFORE log1p)
    control_baseline = None
    if config.add_pert_efficiency and perturbation_col in adata.obs.columns:
        try:
            control_baseline = compute_control_baseline(adata, perturbation_col, control_label)
            efficiency = compute_knockdown_efficiency(
                adata,
                control_baseline,
                perturbation_col,
                control_label,
                config.eps,
            )
            adata.obs[config.efficiency_key] = efficiency
            logger.info(f"Computed {config.efficiency_key}")
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not compute efficiency - {e}")
            control_baseline = None

    # Log1p transformation
    sc.pp.log1p(adata)
    logger.info("Applied log1p transformation")
    ensure_csr(adata)

    # Compute log deviation (AFTER log1p)
    if config.add_pert_efficiency and perturbation_col in adata.obs.columns:
        if control_baseline is not None:
            control_baseline_log = np.log1p(control_baseline)
            log_fc = compute_log_deviation(
                adata,
                control_baseline_log,
                perturbation_col,
                control_label,
            )
            adata.obs[config.target_fc_key] = log_fc
            logger.info(f"Computed {config.target_fc_key}")

    # Optional HVG selection (STATE-specific)
    if config.num_hvgs is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=config.num_hvgs)
        adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()
        logger.info(f"Selected {config.num_hvgs} HVGs, stored in obsm['X_hvg']")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression=None)
    logger.info(f"Wrote {output_path.name}")

    return TransformStats(
        input_path=input_path,
        output_path=output_path,
        cells_total=adata.n_obs,
        genes_total=adata.n_vars,
        control_cells=n_control,
        perturbed_cells=n_perturbed,
    )


# ---------------------------------------------------------------------------
# Multi-file orchestrator
# ---------------------------------------------------------------------------

def normalize_transform_files(
    config: PreprocessTrainConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PreprocessTrainResult:
    """Normalize and transform H5AD files for TX training.

    Processes each file through the full pipeline:
    1. Optional gene alignment to gene_set
    2. Standardize perturbation fields and context/batch columns
    3. Sort cells by specified columns
    4. Optional expm1 (undo log1p)
    5. Optional binomial downsampling
    6. Normalize total counts
    7. Compute knockdown efficiency (before log1p)
    8. Apply log1p transformation
    9. Compute log fold change (after log1p)
    10. Optional HVG selection
    11. Write transformed file
    """
    gene_set = load_gene_set(config.gene_set) if config.gene_set else None

    # Resolve input files
    if config.input_paths:
        h5ad_files = [Path(p).expanduser().resolve() for p in config.input_paths]
        excluded_files: list[Path] = []
    elif config.input_pattern:
        h5ad_files, excluded_files = discover_h5ad_files_with_exclusions(
            config.input_pattern, config.exclude_patterns, verbose=True
        )
    else:
        raise ValueError("Either input_paths or input_pattern must be set")

    # Dry run: just report discovery
    if config.dry_run:
        logger.info(f"[DRY RUN] Discovered {len(h5ad_files)} files, {len(excluded_files)} excluded")
        for f in h5ad_files:
            output_path = compute_output_path(f, config.output_dir)
            logger.info(f"  {f} -> {output_path}")
        if progress_callback:
            progress_callback({
                "kind": "progress",
                "phase": "dry_run_complete",
                "files_discovered": len(h5ad_files),
                "files_excluded": len(excluded_files),
                "message": f"[DRY RUN] Discovered {len(h5ad_files)} input files.",
            })
        return PreprocessTrainResult(
            files_processed=0,
            files_skipped=0,
            total_cells=0,
            file_stats=[],
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    file_stats: list[TransformStats] = []
    skipped_count = 0
    total_cells = 0

    if progress_callback:
        progress_callback({
            "kind": "progress",
            "phase": "starting",
            "files_total": len(h5ad_files),
            "message": f"Starting preprocessing of {len(h5ad_files)} files.",
        })

    for idx, h5ad_path in enumerate(h5ad_files):
        # Check for cancellation
        if cancel_check is not None and cancel_check():
            raise PreprocessCancelledError(
                f"Preprocessing cancelled after {idx}/{len(h5ad_files)} files."
            )

        if progress_callback:
            progress_callback({
                "kind": "progress",
                "phase": "processing",
                "files_done": idx,
                "files_total": len(h5ad_files),
                "current_file": str(h5ad_path.name),
                "message": f"Processing file {idx + 1}/{len(h5ad_files)}: {h5ad_path.name}",
            })

        output_path = compute_output_path(h5ad_path, config.output_dir)

        # Skip if output already exists (unless overwrite is enabled)
        if output_path.exists() and not config.overwrite:
            logger.info(f"Skipping {h5ad_path.name} - output already exists")
            skipped_count += 1
            continue

        file_seed = config.seed + idx
        stats = normalize_log_transform_single(
            h5ad_path, output_path, config, file_seed, gene_set
        )

        if stats is None:
            skipped_count += 1
        else:
            file_stats.append(stats)
            total_cells += stats.cells_total

        force_release_memory()

    logger.info(
        f"Preprocessing complete: {len(file_stats)} files processed, "
        f"{skipped_count} skipped, {total_cells:,} total cells"
    )

    return PreprocessTrainResult(
        files_processed=len(file_stats),
        files_skipped=skipped_count,
        total_cells=total_cells,
        file_stats=file_stats,
    )
