"""Shared pseudobulk accumulation logic for predict and infer paths."""

from __future__ import annotations


def _to_deseq2_counts_np(x, from_log1p: bool):
    """Convert array-like values to non-negative rounded counts for DESeq2."""
    import numpy as np

    arr = np.asarray(x, dtype=np.float64)
    if from_log1p:
        arr = np.expm1(arr)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    return np.round(arr)


def resolve_scale_flags(
    *,
    metrics_is_log1p: bool,
    use_count_outputs: bool,
    resolved_exp_counts: bool,
    output_space: str,
    nb_loss_enabled: bool = False,
) -> tuple[bool, bool]:
    """Determine whether pseudobulk accumulation should use expm1 before summing.

    Returns
    -------
    aggregate_main_in_count_space : bool
        True when the main (embedding-space) outputs are log1p-scaled and must be
        converted to count space before summation.
    aggregate_gene_in_count_space : bool
        True when the gene-space outputs are log1p-scaled and must be converted to
        count space before summation.
    """
    aggregate_main_in_count_space = bool(
        (not use_count_outputs) and metrics_is_log1p and output_space != "embedding"
    )
    aggregate_gene_in_count_space = bool(
        use_count_outputs and metrics_is_log1p and (not resolved_exp_counts)
    )
    return aggregate_main_in_count_space, aggregate_gene_in_count_space


class PseudobulkAccumulator:
    """Streaming pseudobulk accumulator for (context, perturbation) groups.

    When ``has_real=False`` (inference path), skips all real-related accumulation.
    When ``enable_deseq2=False``, skips replicate sums.
    """

    def __init__(
        self,
        output_dim: int,
        gene_dim: int | None,
        use_count_outputs: bool,
        aggregate_main_in_count_space: bool,
        aggregate_gene_in_count_space: bool,
        has_real: bool = True,
        enable_deseq2: bool = True,
        deseq2_n_reps: int = 2,
        nb_loss_enabled: bool = False,
        resolved_exp_counts: bool = False,
    ):
        self._output_dim = output_dim
        self._gene_dim = gene_dim
        self._use_count_outputs = use_count_outputs
        self._agg_main_count = aggregate_main_in_count_space
        self._agg_gene_count = aggregate_gene_in_count_space
        self._has_real = has_real
        self._enable_deseq2 = enable_deseq2
        self._deseq2_n_reps = deseq2_n_reps
        self._nb_loss_enabled = nb_loss_enabled
        self._resolved_exp_counts = resolved_exp_counts
        self._groups: dict[tuple[str, str], dict] = {}
        self._total_cells = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def accumulate_batch(
        self,
        *,
        batch_size: int,
        context_labels: list[str],
        pert_names: list[str],
        celltypes: list[str],
        batch_labels: list[str],
        pred_np: "np.ndarray",
        real_np: "np.ndarray | None" = None,
        gene_pred_np: "np.ndarray | None" = None,
        gene_real_np: "np.ndarray | None" = None,
    ) -> None:
        """Accumulate a batch of predictions into pseudobulk groups.

        Parameters
        ----------
        batch_size:
            Number of cells in this batch.
        context_labels:
            Per-cell context string (e.g. ``dataset_name::cell_type``).
        pert_names:
            Per-cell perturbation label.
        celltypes:
            Per-cell cell-type name.
        batch_labels:
            Per-cell batch/gem_group label.
        pred_np:
            Predicted embedding/expression array, shape ``(batch_size, output_dim)``.
        real_np:
            Ground-truth embedding/expression array (required when ``has_real=True``).
        gene_pred_np:
            Predicted gene-space array when ``use_count_outputs=True``.
        gene_real_np:
            Ground-truth gene-space array when ``use_count_outputs=True`` and
            ``has_real=True``.
        """
        import numpy as np

        self._total_cells += batch_size

        # ---- Convert to count space for summation if needed ----
        if self._agg_main_count:
            batch_pred_pb = np.expm1(pred_np.astype(np.float64, copy=False))
            np.clip(batch_pred_pb, a_min=0.0, a_max=None, out=batch_pred_pb)
            if self._has_real:
                batch_real_pb = np.expm1(real_np.astype(np.float64, copy=False))
                np.clip(batch_real_pb, a_min=0.0, a_max=None, out=batch_real_pb)
            else:
                batch_real_pb = None
        else:
            batch_pred_pb = pred_np
            batch_real_pb = real_np if self._has_real else None

        batch_gene_pred_pb = None
        batch_gene_real_pb = None
        if self._use_count_outputs:
            if self._agg_gene_count:
                # NB predictions are already in count space; non-NB decoder outputs may be log1p.
                if self._nb_loss_enabled:
                    batch_gene_pred_pb = gene_pred_np.astype(np.float64, copy=False)
                else:
                    batch_gene_pred_pb = np.expm1(gene_pred_np.astype(np.float64, copy=False))
                np.clip(batch_gene_pred_pb, a_min=0.0, a_max=None, out=batch_gene_pred_pb)
                if self._has_real:
                    # Real values are log1p-scaled when exp_counts=False; convert to count space.
                    batch_gene_real_pb = np.expm1(gene_real_np.astype(np.float64, copy=False))
                    np.clip(batch_gene_real_pb, a_min=0.0, a_max=None, out=batch_gene_real_pb)
            else:
                batch_gene_pred_pb = gene_pred_np
                if self._has_real:
                    batch_gene_real_pb = gene_real_np

        # ---- Group indices by (context, pert) ----
        group_to_indices: dict[tuple[str, str], list[int]] = {}
        for idx in range(batch_size):
            key = (context_labels[idx], pert_names[idx])
            group_to_indices.setdefault(key, []).append(idx)

        # ---- Accumulate per-group ----
        for (context_label, pert_name), idxs in group_to_indices.items():
            idx_arr = np.asarray(idxs, dtype=np.int64)
            first_idx = int(idx_arr[0])
            current_celltype = celltypes[first_idx]
            current_batch = str(batch_labels[first_idx])

            entry = self._groups.get((context_label, pert_name))
            if entry is None:
                entry = self._make_entry(context_label, pert_name, current_celltype, current_batch)
                self._groups[(context_label, pert_name)] = entry
            elif entry["celltype_name"] != current_celltype:
                raise ValueError(
                    f"Inconsistent cell type for context/pert pair ({context_label}, {pert_name}): "
                    f"saw '{current_celltype}' after '{entry['celltype_name']}'."
                )

            n_cells_this = int(idx_arr.size)
            entry["count"] += n_cells_this

            # Main embedding/expression sums
            entry["pred_sum"] += batch_pred_pb[idx_arr].sum(axis=0, dtype=np.float64)
            if self._has_real:
                entry["real_sum"] += batch_real_pb[idx_arr].sum(axis=0, dtype=np.float64)

            # Main log-space sums (for eval mean when aggregating in count space)
            if entry.get("main_pred_logsum") is not None:
                entry["main_pred_logsum"] += pred_np[idx_arr].sum(axis=0, dtype=np.float64)
                if self._has_real:
                    entry["main_real_logsum"] += real_np[idx_arr].sum(axis=0, dtype=np.float64)

            # Gene-space sums
            if self._use_count_outputs:
                entry["counts_pred_sum"] += batch_gene_pred_pb[idx_arr].sum(axis=0, dtype=np.float64)
                if self._has_real:
                    entry["x_hvg_sum"] += batch_gene_real_pb[idx_arr].sum(axis=0, dtype=np.float64)
                if entry.get("gene_pred_logsum") is not None:
                    # Accumulate in log1p space for eval (mean of log1p values)
                    entry["gene_pred_logsum"] += gene_pred_np[idx_arr].sum(axis=0, dtype=np.float64)
                    if self._has_real:
                        entry["gene_real_logsum"] += gene_real_np[idx_arr].sum(axis=0, dtype=np.float64)

            # ---- Per-replicate accumulation for DESeq2 ----
            if self._enable_deseq2:
                self._accumulate_deseq2_reps(
                    entry,
                    idx_arr,
                    n_cells_this,
                    pred_np=pred_np,
                    real_np=real_np,
                    gene_pred_np=gene_pred_np,
                    gene_real_np=gene_real_np,
                )

    def finalize(self) -> tuple[list[dict], int]:
        """Return sorted group entries and total cells seen."""
        group_entries = list(self._groups.values())
        group_entries.sort(key=lambda x: (str(x["celltype_name"]), str(x["context"]), str(x["pert_name"])))
        return group_entries, self._total_cells

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_entry(
        self,
        context_label: str,
        pert_name: str,
        celltype: str,
        batch_name: str,
    ) -> dict:
        """Create a fresh accumulation entry for a (context, pert) group."""
        import numpy as np

        _need_gene_logsum = self._use_count_outputs and self._agg_gene_count
        _need_main_logsum = self._agg_main_count
        _deseq2_gene_dim = self._gene_dim if self._use_count_outputs else self._output_dim

        entry: dict = {
            "context": context_label,
            "pert_name": pert_name,
            "celltype_name": celltype,
            "batch_name": batch_name,
            "count": 0,
            "pred_sum": np.zeros(self._output_dim, dtype=np.float64),
            # Gene-space accumulators (only when use_count_outputs)
            "counts_pred_sum": np.zeros(self._gene_dim, dtype=np.float64) if self._use_count_outputs else None,
            # Log-space accumulators for eval means
            "gene_pred_logsum": np.zeros(self._gene_dim, dtype=np.float64) if _need_gene_logsum else None,
            "main_pred_logsum": np.zeros(self._output_dim, dtype=np.float64) if _need_main_logsum else None,
        }

        # Real-side accumulators (predict path only)
        if self._has_real:
            entry["real_sum"] = np.zeros(self._output_dim, dtype=np.float64)
            entry["x_hvg_sum"] = np.zeros(self._gene_dim, dtype=np.float64) if self._use_count_outputs else None
            entry["gene_real_logsum"] = np.zeros(self._gene_dim, dtype=np.float64) if _need_gene_logsum else None
            entry["main_real_logsum"] = np.zeros(self._output_dim, dtype=np.float64) if _need_main_logsum else None

        # Per-replicate count sums for DESeq2 (integer counts)
        if self._enable_deseq2:
            entry["pred_rep_sums"] = [np.zeros(_deseq2_gene_dim, dtype=np.float64) for _ in range(self._deseq2_n_reps)]
            entry["rep_counts"] = [0] * self._deseq2_n_reps
            if self._has_real:
                entry["real_rep_sums"] = [
                    np.zeros(_deseq2_gene_dim, dtype=np.float64) for _ in range(self._deseq2_n_reps)
                ]

        return entry

    def _accumulate_deseq2_reps(
        self,
        entry: dict,
        idx_arr: "np.ndarray",
        n_cells_this: int,
        *,
        pred_np: "np.ndarray",
        real_np: "np.ndarray | None",
        gene_pred_np: "np.ndarray | None",
        gene_real_np: "np.ndarray | None",
    ) -> None:
        """Accumulate per-replicate DESeq2 integer counts via round-robin."""
        import numpy as np

        if self._use_count_outputs:
            # Prediction scale: NB outputs are count-space; non-NB may be log1p.
            _pred_from_log1p = (not self._nb_loss_enabled) and self._agg_gene_count
            _pred_count = _to_deseq2_counts_np(gene_pred_np[idx_arr], from_log1p=_pred_from_log1p)

            if self._has_real:
                # Real scale: exp_counts=True -> already count-space; exp_counts=False -> expm1.
                _real_from_log1p = not self._resolved_exp_counts
                _real_count = _to_deseq2_counts_np(gene_real_np[idx_arr], from_log1p=_real_from_log1p)
        else:
            _pred_count = _to_deseq2_counts_np(pred_np[idx_arr], from_log1p=True)
            if self._has_real:
                _real_count = _to_deseq2_counts_np(real_np[idx_arr], from_log1p=True)

        # Round-robin assign cells to replicates
        prev_count = entry["count"] - n_cells_this
        rep_assignments = np.arange(prev_count, prev_count + n_cells_this) % self._deseq2_n_reps
        for rep in range(self._deseq2_n_reps):
            mask = rep_assignments == rep
            if mask.any():
                entry["pred_rep_sums"][rep] += _pred_count[mask].sum(axis=0)
                entry["rep_counts"][rep] += int(mask.sum())
                if self._has_real:
                    entry["real_rep_sums"][rep] += _real_count[mask].sum(axis=0)


def build_pseudobulk_anndata(
    group_entries: list[dict],
    *,
    output_dim: int,
    gene_dim: int | None,
    use_count_outputs: bool,
    aggregate_main_in_count_space: bool,
    aggregate_gene_in_count_space: bool,
    has_real: bool = True,
    pert_col: str,
    cell_type_key: str,
    batch_obs_key: str,
    batch_col: str | None = None,
    embed_key: str | None = None,
    gene_var_names: list[str] | None = None,
) -> dict:
    """Build AnnData objects from finalized pseudobulk group entries.

    Parameters
    ----------
    group_entries:
        Sorted list of group dicts from ``PseudobulkAccumulator.finalize()``.
    output_dim:
        Dimensionality of the main (embedding) output.
    gene_dim:
        Dimensionality of the gene-space output (required when ``use_count_outputs``).
    use_count_outputs:
        Whether gene-space count outputs were collected.
    aggregate_main_in_count_space:
        Whether main sums were accumulated in count space (via expm1).
    aggregate_gene_in_count_space:
        Whether gene sums were accumulated in count space (via expm1).
    has_real:
        Whether real (ground-truth) arrays are present.
    pert_col:
        obs column name for perturbation labels.
    cell_type_key:
        obs column name for cell-type labels.
    batch_obs_key:
        obs column name for batch labels (always written).
    batch_col:
        Original batch column name from training config. If different from
        ``batch_obs_key``, a duplicate column is added.
    embed_key:
        If not None, embedding-space sums are stored in ``.obsm[embed_key]``
        instead of (or alongside) ``.X``.
    gene_var_names:
        Gene names to use for ``.var`` index when ``use_count_outputs=True``.

    Returns
    -------
    dict with keys:
        ``adata_pred``, ``adata_real``, ``adata_pred_eval``, ``adata_real_eval``,
        ``pseudobulk_meta``.  When ``has_real=False``, ``adata_real`` and
        ``adata_real_eval`` are ``None``.
    """
    import anndata
    import numpy as np
    import pandas as pd

    n_groups = len(group_entries)

    # ---- Allocate sum and eval matrices ----
    pred_bulk_sum = np.empty((n_groups, output_dim), dtype=np.float32)
    pred_bulk_eval = np.empty((n_groups, output_dim), dtype=np.float32)
    real_bulk_sum = np.empty((n_groups, output_dim), dtype=np.float32) if has_real else None
    real_bulk_eval = np.empty((n_groups, output_dim), dtype=np.float32) if has_real else None

    pred_x_sum = np.empty((n_groups, gene_dim), dtype=np.float32) if use_count_outputs else None
    pred_x_eval = np.empty((n_groups, gene_dim), dtype=np.float32) if use_count_outputs else None
    real_x_sum = np.empty((n_groups, gene_dim), dtype=np.float32) if (use_count_outputs and has_real) else None
    real_x_eval = np.empty((n_groups, gene_dim), dtype=np.float32) if (use_count_outputs and has_real) else None

    # ---- Collision-safe obs column names ----
    reserved_obs_keys = {pert_col, cell_type_key, batch_obs_key}
    if batch_col:
        reserved_obs_keys.add(batch_col)

    pseudobulk_context_key = "pseudobulk_context"
    while pseudobulk_context_key in reserved_obs_keys:
        pseudobulk_context_key = f"_{pseudobulk_context_key}"
    pseudobulk_n_cells_key = "pseudobulk_n_cells"
    while pseudobulk_n_cells_key in reserved_obs_keys or pseudobulk_n_cells_key == pseudobulk_context_key:
        pseudobulk_n_cells_key = f"_{pseudobulk_n_cells_key}"

    # ---- Build obs dict ----
    obs_dict: dict[str, list] = {
        pert_col: [],
        cell_type_key: [],
        batch_obs_key: [],
        pseudobulk_context_key: [],
        pseudobulk_n_cells_key: [],
    }
    if batch_col and batch_col != batch_obs_key:
        obs_dict[batch_col] = []

    # ---- Fill matrices and obs from group entries ----
    for idx, entry in enumerate(group_entries):
        count = int(entry["count"])
        denom = float(count)

        # Main embedding sums
        pred_bulk_sum[idx, :] = entry["pred_sum"].astype(np.float32, copy=False)
        if has_real:
            real_bulk_sum[idx, :] = entry["real_sum"].astype(np.float32, copy=False)

        # Main eval: mean(log1p(x)) when aggregating in count space, else mean(x)
        if aggregate_main_in_count_space:
            pred_bulk_eval[idx, :] = (entry["main_pred_logsum"] / denom).astype(np.float32, copy=False)
            if has_real:
                real_bulk_eval[idx, :] = (entry["main_real_logsum"] / denom).astype(np.float32, copy=False)
        else:
            pred_bulk_eval[idx, :] = (entry["pred_sum"] / denom).astype(np.float32, copy=False)
            if has_real:
                real_bulk_eval[idx, :] = (entry["real_sum"] / denom).astype(np.float32, copy=False)

        # Gene-space sums and eval
        if use_count_outputs:
            pred_x_sum[idx, :] = entry["counts_pred_sum"].astype(np.float32, copy=False)
            if has_real:
                real_x_sum[idx, :] = entry["x_hvg_sum"].astype(np.float32, copy=False)

            if aggregate_gene_in_count_space:
                pred_x_eval[idx, :] = (entry["gene_pred_logsum"] / denom).astype(np.float32, copy=False)
                if has_real:
                    real_x_eval[idx, :] = (entry["gene_real_logsum"] / denom).astype(np.float32, copy=False)
            else:
                pred_x_eval[idx, :] = (entry["counts_pred_sum"] / denom).astype(np.float32, copy=False)
                if has_real:
                    real_x_eval[idx, :] = (entry["x_hvg_sum"] / denom).astype(np.float32, copy=False)

        # obs metadata
        obs_dict[pert_col].append(entry["pert_name"])
        obs_dict[cell_type_key].append(entry["celltype_name"])
        obs_dict[batch_obs_key].append(entry["batch_name"])
        obs_dict[pseudobulk_context_key].append(entry["context"])
        obs_dict[pseudobulk_n_cells_key].append(count)
        if batch_col and batch_col != batch_obs_key:
            obs_dict[batch_col].append(entry["batch_name"])

    obs = pd.DataFrame(obs_dict)
    gene_var_df = pd.DataFrame(index=gene_var_names) if gene_var_names is not None else None

    # ---- Build AnnData objects ----
    if use_count_outputs:
        _var = gene_var_df if (gene_var_df is not None and len(gene_var_df) == pred_x_sum.shape[1]) else None
        # Persisted outputs: summed pseudobulks.
        adata_pred = anndata.AnnData(X=pred_x_sum, obs=obs, var=_var)
        if embed_key is not None:
            adata_pred.obsm[embed_key] = pred_bulk_sum

        adata_real = None
        if has_real:
            adata_real = anndata.AnnData(X=real_x_sum, obs=obs, var=_var)
            if embed_key is not None:
                adata_real.obsm[embed_key] = real_bulk_sum

        # Metric inputs: mean(log1p) pseudobulks for log1p outputs, mean otherwise.
        adata_pred_eval = anndata.AnnData(X=pred_x_eval, obs=obs.copy(), var=_var)
        if embed_key is not None:
            adata_pred_eval.obsm[embed_key] = pred_bulk_eval

        adata_real_eval = None
        if has_real:
            adata_real_eval = anndata.AnnData(X=real_x_eval, obs=obs.copy(), var=_var)
            if embed_key is not None:
                adata_real_eval.obsm[embed_key] = real_bulk_eval
    else:
        # Persisted outputs: summed pseudobulks.
        adata_pred = anndata.AnnData(X=pred_bulk_sum, obs=obs)
        adata_real = anndata.AnnData(X=real_bulk_sum, obs=obs) if has_real else None
        # Metric inputs: mean(log1p) pseudobulks for log1p outputs, mean otherwise.
        adata_pred_eval = anndata.AnnData(X=pred_bulk_eval, obs=obs.copy())
        adata_real_eval = anndata.AnnData(X=real_bulk_eval, obs=obs.copy()) if has_real else None

    # ---- Aggregation metadata ----
    persist_mode = (
        "sum(expm1(log1p(x)))" if (aggregate_main_in_count_space or aggregate_gene_in_count_space) else "sum(x)"
    )
    eval_mode = (
        "mean(log1p(x))" if (aggregate_main_in_count_space or aggregate_gene_in_count_space) else "mean(x)"
    )
    pseudobulk_meta = {
        "persisted_aggregation": persist_mode,
        "eval_aggregation": eval_mode,
        "n_cells_obs_column": pseudobulk_n_cells_key,
    }
    adata_pred.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)
    adata_pred_eval.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)
    if adata_real is not None:
        adata_real.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)
    if adata_real_eval is not None:
        adata_real_eval.uns["pseudobulk_aggregation"] = dict(pseudobulk_meta)

    return {
        "adata_pred": adata_pred,
        "adata_real": adata_real,
        "adata_pred_eval": adata_pred_eval,
        "adata_real_eval": adata_real_eval,
        "pseudobulk_meta": pseudobulk_meta,
    }
