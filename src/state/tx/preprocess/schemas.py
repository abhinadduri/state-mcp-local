"""Pydantic schemas for TX training data preprocessing."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class PreprocessTrainConfig(BaseModel):
    """Configuration for normalizing and transforming H5AD files for TX training."""

    # --- Input specification ---
    input_paths: list[str] = Field(
        default_factory=list,
        description="Explicit list of H5AD file paths (takes precedence over input_pattern)",
    )
    input_pattern: str | None = Field(
        default=None,
        description="Glob pattern for input H5AD files (e.g., '/data/**/*.h5ad')",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns to exclude from input_pattern discovery",
    )
    output_dir: Path = Field(..., description="Directory to write transformed files")

    # --- Normalization ---
    target_sum: float | None = Field(
        default=None,
        description="Target count for normalize_total (None = scanpy default median)",
    )
    already_log1p: bool = Field(
        default=False,
        description="If True, apply expm1 to restore raw counts before re-normalizing",
    )

    # --- Column mapping ---
    perturbation_col: str = Field(
        default="target_gene",
        description="obs column for perturbation identifiers",
    )
    control_perturbation: str = Field(
        default="non-targeting",
        description="Control perturbation label for baseline computation",
    )
    context_col: str | None = Field(
        default=None,
        description="obs column to copy into obs['context']",
    )
    batch_col: str | None = Field(
        default=None,
        description="obs column to combine with context into obs['batch_col']",
    )

    # --- Ordering ---
    sort_by: list[str] = Field(
        default_factory=list,
        description="Sort cells by these obs columns (mergesort, e.g., ['context','perturbation'])",
    )

    # --- Gene alignment ---
    gene_set: Path | None = Field(
        default=None,
        description="Path to .npy file with ordered gene names for X/var alignment",
    )

    # --- Knockdown efficiency ---
    add_pert_efficiency: bool = Field(
        default=True,
        description="Compute and add knockdown efficiency + log deviation to obs",
    )
    efficiency_key: str = Field(
        default="KnockDownEfficiency",
        description="Key to store knockdown efficiency values in obs",
    )
    target_fc_key: str = Field(
        default="KnockDownGeneFC",
        description="Key to store log fold change values in obs",
    )
    eps: float = Field(
        default=1e-8,
        gt=0,
        description="Small epsilon for numerical stability in efficiency computation",
    )

    # --- Downsampling ---
    downsample_frac: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Fraction of counts to retain via binomial downsampling (1.0 = no downsample)",
    )

    # --- HVG selection (STATE-specific, optional post-processing) ---
    num_hvgs: int | None = Field(
        default=None,
        description="If set, select N highly variable genes and store in obsm['X_hvg']",
    )

    # --- Control ---
    seed: int = Field(default=42, description="Random seed for reproducibility")
    overwrite: bool = Field(
        default=False,
        description="If True, overwrite existing output files instead of skipping",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, show file discovery without processing",
    )

    @model_validator(mode="after")
    def validate_input_source(self) -> "PreprocessTrainConfig":
        if not self.input_paths and not self.input_pattern:
            raise ValueError("Either input_paths or input_pattern must be provided")
        return self

    @model_validator(mode="after")
    def validate_keys_different(self) -> "PreprocessTrainConfig":
        if self.efficiency_key == self.target_fc_key:
            raise ValueError(
                f"efficiency_key and target_fc_key must be different, "
                f"got '{self.efficiency_key}' for both"
            )
        return self

    @field_validator("target_sum")
    @classmethod
    def validate_target_sum(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("target_sum must be > 0 when provided")
        return value


class TransformStats(BaseModel):
    """Statistics for a single transformed H5AD file."""

    input_path: Path
    output_path: Path
    cells_total: int
    genes_total: int
    control_cells: int
    perturbed_cells: int


class PreprocessTrainResult(BaseModel):
    """Result of the preprocessing operation."""

    files_processed: int
    files_skipped: int
    total_cells: int
    file_stats: list[TransformStats]
