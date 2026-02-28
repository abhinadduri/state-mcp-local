"""CLI entry point for TX preprocess_train subcommand."""

import argparse as ap


def add_arguments_preprocess_train(parser: ap.ArgumentParser):
    """Add arguments for the preprocess_train subcommand."""
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--adata",
        type=str,
        help="Path to single input H5AD file",
    )
    input_group.add_argument(
        "--input-pattern",
        type=str,
        help="Glob pattern for input H5AD files (e.g., '/data/**/*.h5ad')",
    )

    parser.add_argument(
        "--exclude-patterns",
        type=str,
        nargs="*",
        default=[],
        help="Glob patterns to exclude from input_pattern discovery",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write transformed files",
    )

    # Normalization
    parser.add_argument("--target-sum", type=float, default=None, help="Target count for normalization")
    parser.add_argument("--already-log1p", action="store_true", help="Apply expm1 before normalization")

    # Column mapping
    parser.add_argument(
        "--perturbation-col", type=str, default="target_gene", help="obs column for perturbation identifiers"
    )
    parser.add_argument("--control-perturbation", type=str, default="non-targeting", help="Control perturbation label")
    parser.add_argument("--context-col", type=str, default=None, help="obs column to copy into obs['context']")
    parser.add_argument(
        "--batch-col", type=str, default=None, help="obs column to combine with context into obs['batch_col']"
    )

    # Ordering
    parser.add_argument(
        "--sort-by",
        type=str,
        nargs="*",
        default=[],
        help="Sort cells by these obs columns (e.g., --sort-by context perturbation). "
        "Sorting produces contiguous groups required by use_consecutive_loading=True in TX training.",
    )

    # Gene alignment
    parser.add_argument("--gene-set", type=str, default=None, help="Path to .npy file with ordered gene names")

    # Efficiency
    parser.add_argument("--no-pert-efficiency", action="store_true", help="Skip knockdown efficiency computation")
    parser.add_argument("--efficiency-key", type=str, default="KnockDownEfficiency")
    parser.add_argument("--target-fc-key", type=str, default="KnockDownGeneFC")
    parser.add_argument("--eps", type=float, default=1e-8)

    # Downsampling
    parser.add_argument("--downsample-frac", type=float, default=1.0, help="Fraction of counts to retain")

    # HVG (STATE-specific)
    parser.add_argument("--num-hvgs", type=int, default=None, help="Number of highly variable genes to select")

    # Control
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Show file discovery without processing")


def run_tx_preprocess_train(args: ap.Namespace):
    """Run the full preprocessing pipeline from CLI args."""
    import logging
    from pathlib import Path

    from state.tx.preprocess import PreprocessTrainConfig, normalize_transform_files

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_paths = [args.adata] if args.adata else []
    config = PreprocessTrainConfig(
        input_paths=input_paths,
        input_pattern=getattr(args, "input_pattern", None),
        exclude_patterns=getattr(args, "exclude_patterns", []),
        output_dir=Path(args.output_dir),
        target_sum=getattr(args, "target_sum", None),
        already_log1p=getattr(args, "already_log1p", False),
        perturbation_col=getattr(args, "perturbation_col", "target_gene"),
        control_perturbation=getattr(args, "control_perturbation", "non-targeting"),
        context_col=getattr(args, "context_col", None),
        batch_col=getattr(args, "batch_col", None),
        sort_by=getattr(args, "sort_by", []),
        gene_set=Path(args.gene_set) if getattr(args, "gene_set", None) else None,
        add_pert_efficiency=not getattr(args, "no_pert_efficiency", False),
        efficiency_key=getattr(args, "efficiency_key", "KnockDownEfficiency"),
        target_fc_key=getattr(args, "target_fc_key", "KnockDownGeneFC"),
        eps=getattr(args, "eps", 1e-8),
        downsample_frac=getattr(args, "downsample_frac", 1.0),
        num_hvgs=getattr(args, "num_hvgs", None),
        seed=getattr(args, "seed", 42),
        overwrite=getattr(args, "overwrite", False),
        dry_run=getattr(args, "dry_run", False),
    )

    result = normalize_transform_files(config)
    print(
        f"\nProcessed {result.files_processed} files, {result.files_skipped} skipped, {result.total_cells:,} total cells."
    )
