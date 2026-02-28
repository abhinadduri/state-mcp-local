from __future__ import annotations

import os
import shutil
import subprocess
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .mcp_utils import (
    infer_tx_inference_defaults,
    inspect_adata_schema as inspect_adata_schema_file,
    inspect_emb_checkpoint as inspect_emb_checkpoint_file,
    inspect_model_folder,
    inspect_tx_split_sources as inspect_tx_split_sources_file,
    inspect_tx_split_toml as inspect_tx_split_toml_file,
    inspect_tx_checkpoint as inspect_tx_checkpoint_file,
    normalize_and_validate_emb_inference_dir,
    normalize_and_validate_run_dir,
    plan_tx_split_toml as plan_tx_split_toml_file,
    resolve_emb_checkpoint_path,
    resolve_and_validate_model_folder,
)

from ._jobs import (
    InferenceJob,
    PreprocessJob,
    TrainJob,
    _CANCEL_GRACE_SECONDS,
    _FALLBACK_SESSION_KEY,
    _JOBS,
    _JOBS_LOCK,
    _PREPROCESS_JOBS_LOCK,
    _SESSION_SERVER_STATE_LOCK,
    _TERMINAL_JOB_STATUSES,
    _TRAIN_JOBS_LOCK,
    _append_job_log,
    _append_preprocess_job_log,
    _append_train_job_log,
    _get_job_locked,
    _get_preprocess_job_locked,
    _get_session_jobs_locked,
    _get_session_preprocess_jobs_locked,
    _get_session_server_state_locked,
    _get_session_train_idempotency_locked,
    _get_session_train_jobs_locked,
    _get_train_job_locked,
    _get_worker_mp_context,
    _touch_job,
    _touch_preprocess_job,
    _touch_train_job,
    _utc_now_iso,
)

from ._slurm import (
    _build_emb_transform_cli_args,
    _build_tx_infer_cli_args,
    _resolve_backend_mode,
    _submit_slurm_job,
    _submit_tx_train_slurm_job,
)

from ._workers import (
    _run_emb_inference_job_worker,
    _run_inference_job_worker,
    _run_preprocess_job_worker,
    _run_tx_train_job_worker,
)

from ._gpu import (
    find_free_devices,
    is_nvidia_smi_available,
    query_gpu_devices,
)

from ._sync import (
    _process_alive,
    _recommend_poll_interval_seconds,
    _recommend_preprocess_poll_interval_seconds,
    _recommend_train_poll_interval_seconds,
    _release_job_runtime_resources,
    _release_preprocess_job_runtime_resources,
    _release_train_job_runtime_resources,
    _sync_job_state_locked,
    _sync_preprocess_job_state_locked,
    _sync_train_job_state_locked,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time dependency guidance
    raise ImportError(
        "MCP support requires the `mcp` package. Install it in this environment to run `state.mcp`."
    ) from exc


mcp = FastMCP("state")


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _get_current_session_key() -> str:
    """
    Derive a stable session partition key for the current MCP request.

    Priority:
    1) explicit streamable-http session id header
    2) MCP client id (if provided by client metadata)
    3) in-process session object identity
    4) process-global fallback (non-request contexts)
    """
    try:
        ctx = mcp.get_context()
    except Exception:
        return _FALLBACK_SESSION_KEY

    request = None
    try:
        request = ctx.request_context.request
    except Exception:
        request = None

    if request is not None:
        try:
            headers = request.headers
            if headers is not None:
                session_id = headers.get("mcp-session-id")
                if isinstance(session_id, str) and session_id.strip():
                    return f"mcp:{session_id.strip()}"
        except Exception:
            pass

    try:
        client_id = ctx.client_id
    except Exception:
        client_id = None
    if isinstance(client_id, str) and client_id.strip():
        return f"client:{client_id.strip()}"

    try:
        session_obj = ctx.request_context.session
    except Exception:
        session_obj = None
    if session_obj is not None:
        return f"sessionobj:{id(session_obj)}"

    return _FALLBACK_SESSION_KEY


def _has_wandb_credentials() -> tuple[bool, str]:
    api_key = os.getenv("WANDB_API_KEY")
    if isinstance(api_key, str) and api_key.strip():
        return True, "Detected WANDB_API_KEY in environment."

    netrc_path = Path.home() / ".netrc"
    if netrc_path.is_file():
        try:
            content = netrc_path.read_text(encoding="utf-8", errors="ignore")
            if "api.wandb.ai" in content:
                return True, "Detected wandb credentials in ~/.netrc."
        except Exception:
            pass

    wandb_settings = Path.home() / ".config" / "wandb" / "settings"
    if wandb_settings.is_file():
        return True, "Detected wandb settings at ~/.config/wandb/settings."

    return False, "No wandb credentials detected in environment or standard local config."


def _normalize_wandb_mode(use_wandb: str) -> str:
    mode = str(use_wandb or "auto").strip().lower()
    if mode not in {"auto", "true", "false"}:
        raise ValueError("`use_wandb` must be one of: 'auto', 'true', 'false'.")
    return mode


def _resolve_cuda_devices_for_local(
    cuda_devices: str | None,
    backend_mode: str,
) -> tuple[str | None, bool]:
    """Resolve CUDA_VISIBLE_DEVICES for a local backend job.

    Returns ``(resolved_cuda_devices, auto_assigned)``.
    """
    if backend_mode != "local":
        return None, False
    if cuda_devices is not None:
        return str(cuda_devices).strip(), False
    if is_nvidia_smi_available():
        free = find_free_devices(1)
        if free:
            return str(free[0]), True
    return None, False


def _resolve_tx_inference_request(
    *,
    adata_path: str,
    output_path: str | None,
    model_folder: str | None,
    default_model_folder: str | None,
    checkpoint_path: str | None,
    perturbation_column: str | None,
    embedding_key: str | None,
    cell_type_column: str | None,
    include_cell_types: list[str] | None,
    batch_column: str | None,
    control_perturbation: str | None,
    seed: int,
    max_set_len: int | None,
    padding_tsv_path: str | None,
    simulate_all_perturbations: bool,
    virtual_cells_per_perturbation: int | None,
    min_cells_per_perturbation: int | None,
    max_cells_per_perturbation: int | None,
    batched: bool,
    set_batch_size: int | None,
    quiet: bool,
) -> tuple[Namespace, dict[str, Any]]:
    if not adata_path or not adata_path.strip():
        raise ValueError("`adata_path` must be a non-empty path to an .h5ad file.")
    adata_resolved = str(Path(adata_path).expanduser().resolve())
    if not Path(adata_resolved).is_file():
        raise FileNotFoundError(f"AnnData file not found: {adata_resolved}")
    adata_size_bytes = int(Path(adata_resolved).stat().st_size)

    resolved_model_folder = resolve_and_validate_model_folder(model_folder, default_model_folder)
    defaults = infer_tx_inference_defaults(resolved_model_folder)

    resolved_checkpoint = (
        str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else defaults.get("checkpoint_path")
    )
    if resolved_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint file found in {resolved_model_folder}/checkpoints and no `checkpoint_path` was provided."
        )
    if not Path(resolved_checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved_checkpoint}")

    resolved_set_batch_size = set_batch_size
    if resolved_set_batch_size is None:
        default_set_batch = defaults.get("set_batch_size")
        if isinstance(default_set_batch, int) and default_set_batch > 0:
            resolved_set_batch_size = default_set_batch
    if resolved_set_batch_size is not None and resolved_set_batch_size <= 0:
        raise ValueError("`set_batch_size` must be a positive integer when provided.")

    resolved_pert_col = perturbation_column or defaults.get("perturbation_column") or "drugname_drugconc"
    resolved_embed_key = embedding_key if embedding_key is not None else defaults.get("embedding_key")
    resolved_celltypes_csv = (
        ",".join(ct.strip() for ct in include_cell_types if ct and ct.strip()) if include_cell_types else None
    )
    resolved_output = (
        str(Path(output_path).expanduser().resolve()) if output_path else adata_resolved.replace(".h5ad", "_simulated.h5ad")
    )

    infer_args = Namespace(
        checkpoint=resolved_checkpoint,
        adata=adata_resolved,
        embed_key=resolved_embed_key,
        pert_col=resolved_pert_col,
        output=resolved_output,
        model_dir=resolved_model_folder,
        celltype_col=cell_type_column or defaults.get("cell_type_column"),
        celltypes=resolved_celltypes_csv,
        batch_col=batch_column or defaults.get("batch_column"),
        control_pert=control_perturbation or defaults.get("control_perturbation"),
        seed=seed,
        max_set_len=max_set_len,
        quiet=quiet,
        tsv=padding_tsv_path,
        all_perts=simulate_all_perturbations,
        virtual_cells_per_pert=virtual_cells_per_perturbation,
        min_cells=min_cells_per_perturbation,
        max_cells=max_cells_per_perturbation,
        batched=batched,
        set_batch_size=resolved_set_batch_size,
    )

    resolved_payload = {
        "model_folder": resolved_model_folder,
        "adata_path": adata_resolved,
        "adata_size_bytes": adata_size_bytes,
        "output_path": resolved_output,
        "checkpoint_path": resolved_checkpoint,
        "resolved_args": {
            "perturbation_column": resolved_pert_col,
            "embedding_key": resolved_embed_key,
            "cell_type_column": infer_args.celltype_col,
            "include_cell_types": include_cell_types,
            "batch_column": infer_args.batch_col,
            "control_perturbation": infer_args.control_pert,
            "seed": seed,
            "max_set_len": max_set_len,
            "padding_tsv_path": padding_tsv_path,
            "simulate_all_perturbations": simulate_all_perturbations,
            "virtual_cells_per_perturbation": virtual_cells_per_perturbation,
            "min_cells_per_perturbation": min_cells_per_perturbation,
            "max_cells_per_perturbation": max_cells_per_perturbation,
            "batched": batched,
            "set_batch_size": resolved_set_batch_size,
            "quiet": quiet,
        },
    }

    return infer_args, resolved_payload


def _resolve_emb_inference_request(
    *,
    input_adata_path: str,
    output_path: str | None,
    checkpoint_path: str | None,
    model_folder: str | None,
    config_path: str | None,
    embedding_key: str,
    protein_embeddings_path: str | None,
    batch_size: int | None,
) -> dict[str, Any]:
    if not input_adata_path or not input_adata_path.strip():
        raise ValueError("`input_adata_path` must be a non-empty path to an .h5ad file.")
    input_adata_resolved = str(Path(input_adata_path).expanduser().resolve())
    if not Path(input_adata_resolved).is_file():
        raise FileNotFoundError(f"Input AnnData file not found: {input_adata_resolved}")
    input_adata_size_bytes = int(Path(input_adata_resolved).stat().st_size)

    resolved_checkpoint = resolve_emb_checkpoint_path(
        checkpoint_path=checkpoint_path,
        model_folder=model_folder,
        prefer_ckpt=False,
    )
    resolved_model_folder = str(Path(resolved_checkpoint).parent)

    resolved_config_path: str | None = None
    if config_path is not None:
        resolved_config_path = str(Path(config_path).expanduser().resolve())
        if not Path(resolved_config_path).is_file():
            raise FileNotFoundError(f"Config override file not found: {resolved_config_path}")

    resolved_protein_embeddings_path: str | None = None
    if protein_embeddings_path is not None:
        resolved_protein_embeddings_path = str(Path(protein_embeddings_path).expanduser().resolve())
        if not Path(resolved_protein_embeddings_path).is_file():
            raise FileNotFoundError(f"Protein embeddings file not found: {resolved_protein_embeddings_path}")

    if not embedding_key or not embedding_key.strip():
        raise ValueError("`embedding_key` must be a non-empty string.")

    if batch_size is not None and batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer when provided.")

    resolved_output_path: str
    if output_path is not None:
        resolved_output_path = str(Path(output_path).expanduser().resolve())
    elif input_adata_resolved.lower().endswith(".h5ad"):
        resolved_output_path = input_adata_resolved[:-5] + "_embedded.h5ad"
    else:
        resolved_output_path = input_adata_resolved + "_embedded.h5ad"

    output_mode: str
    if resolved_output_path.lower().endswith(".npy"):
        output_mode = "npy"
    elif resolved_output_path.lower().endswith(".h5ad"):
        output_mode = "h5ad"
    else:
        raise ValueError("`output_path` must end with `.h5ad` or `.npy` when provided.")

    output_adata_path_for_encode: str | None = None
    if resolved_output_path.lower().endswith(".h5ad"):
        output_adata_path_for_encode = resolved_output_path

    return {
        "checkpoint_type": "emb",
        "input_adata_path": input_adata_resolved,
        "input_adata_size_bytes": input_adata_size_bytes,
        "checkpoint_path": resolved_checkpoint,
        "model_folder": resolved_model_folder,
        "config_path": resolved_config_path,
        "protein_embeddings_path": resolved_protein_embeddings_path,
        "embedding_key": embedding_key.strip(),
        "batch_size": batch_size,
        "output_path": resolved_output_path,
        "output_adata_path_for_encode": output_adata_path_for_encode,
        "output_mode": output_mode,
    }


def _normalized_override_key(override: str) -> str | None:
    if "=" not in override:
        return None
    key, _ = override.split("=", 1)
    return key.strip().lstrip("+~")


def _hydra_format_value(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value)
    if text == "":
        return '""'
    safe_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:-")
    if all(char in safe_chars for char in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Model preset registry

_PRESET_ALIASES: dict[str, str] = {
    "cmean": "context_mean",
    "pmean": "perturb_mean",
}

_OUTPUT_SPACE_ALIASES: dict[str, str] = {"hvg": "gene", "transcriptome": "all"}

_KNOWN_PRESETS: set[str] = {
    "state", "state_sm", "state_lg",
    "context_mean", "perturb_mean",
    "embedsum", "globalsimplesum", "pseudobulk", "decoder_only", "pertsets",
}

# Key model kwargs defaults per preset (from configs/model/*.yaml)
_PRESET_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "state":        {"hidden_dim": 384,  "cell_set_len": 64,  "batch_encoder": False, "nb_loss": False},
    "state_sm":     {"hidden_dim": 672,  "cell_set_len": 128, "batch_encoder": False, "nb_loss": False},
    "state_lg":     {"hidden_dim": 1488, "cell_set_len": 512, "batch_encoder": False, "nb_loss": False},
    "context_mean": {"hidden_dim": 512,  "cell_set_len": 512, "batch_encoder": False, "nb_loss": False},
    "perturb_mean": {"hidden_dim": 512,  "cell_set_len": 512, "batch_encoder": False, "nb_loss": False},
}

# Hydra training/data defaults (from configs/training/default.yaml and data/perturbation.yaml)
_DEFAULT_TRAINING_PARAMS: dict[str, Any] = {
    "max_steps": 100000,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "val_freq": 2000,
    "ckpt_every_n_steps": 2000,
}

_DEFAULT_DATA_PARAMS: dict[str, Any] = {
    "use_consecutive_loading": False,
    "num_workers": 12,
}


def _infer_tx_training_contract(embed_key: str | None, output_space: str) -> dict[str, Any]:
    if output_space == "embedding":
        return {
            "main_loss_space": "embedding",
            "decoder_enabled": False,
            "checkpoint_monitor_metric": "val/embedding_loss",
        }

    if output_space == "gene" and embed_key in {None, "X_hvg"}:
        return {
            "main_loss_space": "expression",
            "decoder_enabled": False,
            "checkpoint_monitor_metric": "val/expression_loss",
        }

    return {
        "main_loss_space": "embedding",
        "decoder_enabled": True,
        "checkpoint_monitor_metric": "val/expression_loss",
    }


def _build_tx_train_plan(
    *,
    toml_config_path: str | None,
    output_dir: str | None,
    name: str | None,
    embed_key: str | None,
    output_space: str | None,
    perturbation_column: str | None,
    cell_type_column: str | None,
    batch_column: str | None,
    control_perturbation: str | None,
    model_preset: str,
    max_steps: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    val_freq: int | None,
    seed: int,
    use_wandb: str,
    backend: str,
    backend_profile: str | None,
    slurm_partition: str | None,
    slurm_gpus: int | None,
    slurm_cpus_per_task: int | None,
    slurm_mem: str | None,
    slurm_time: str | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_tags: list[str] | None,
    hidden_dim: int | None,
    cell_set_len: int | None,
    nb_loss: bool | None,
    batch_encoder: bool | None,
    ckpt_every_n_steps: int | None,
    num_workers: int | None,
    use_consecutive_loading: bool | None,
    extra_overrides: list[str] | None,
) -> dict[str, Any]:
    missing_fields: list[str] = []
    validation_errors: list[str] = []
    validation_warnings: list[str] = []
    suggestions: dict[str, Any] = {}

    mode_wandb = "auto"
    try:
        mode_wandb = _normalize_wandb_mode(use_wandb)
    except Exception as exc:
        validation_errors.append(str(exc))

    backend_mode = "local"
    backend_reason = "No backend resolved."
    try:
        backend_mode, backend_reason = _resolve_backend_mode(backend)
    except Exception as exc:
        validation_errors.append(str(exc))

    if max_steps is not None and max_steps <= 0:
        validation_errors.append("`max_steps` must be a positive integer when provided.")
    if batch_size is not None and batch_size <= 0:
        validation_errors.append("`batch_size` must be a positive integer when provided.")
    if val_freq is not None and val_freq <= 0:
        validation_errors.append("`val_freq` must be a positive integer when provided.")
    if learning_rate is not None and learning_rate <= 0:
        validation_errors.append("`learning_rate` must be > 0 when provided.")
    if seed < 0:
        validation_errors.append("`seed` must be >= 0.")

    model_preset_clean = str(model_preset or "").strip()
    model_preset_clean = _PRESET_ALIASES.get(model_preset_clean, model_preset_clean)
    if not model_preset_clean:
        validation_errors.append("`model_preset` must be a non-empty model config name.")
    elif model_preset_clean not in _KNOWN_PRESETS:
        validation_warnings.append(
            f"Model preset {model_preset_clean!r} is not a recognised preset "
            f"(known: {', '.join(sorted(_KNOWN_PRESETS))}). "
            "Hydra will raise an error at launch if the corresponding config file does not exist."
        )

    if slurm_gpus is not None and slurm_gpus <= 0:
        validation_errors.append("`slurm_gpus` must be > 0 when provided.")
    if slurm_cpus_per_task is not None and slurm_cpus_per_task <= 0:
        validation_errors.append("`slurm_cpus_per_task` must be > 0 when provided.")
    if hidden_dim is not None and hidden_dim <= 0:
        validation_errors.append("`hidden_dim` must be > 0 when provided.")
    if cell_set_len is not None and cell_set_len <= 0:
        validation_errors.append("`cell_set_len` must be > 0 when provided.")
    if ckpt_every_n_steps is not None and ckpt_every_n_steps <= 0:
        validation_errors.append("`ckpt_every_n_steps` must be > 0 when provided.")
    if num_workers is not None and num_workers < 0:
        validation_errors.append("`num_workers` must be >= 0 when provided.")

    resolved_num_workers = num_workers
    if resolved_num_workers is None and slurm_cpus_per_task is not None:
        resolved_num_workers = slurm_cpus_per_task

    cleaned_extra_overrides: list[str] = []
    if extra_overrides is not None:
        for raw in extra_overrides:
            text = str(raw).strip()
            if not text:
                continue
            cleaned_extra_overrides.append(text)

    resolved_toml_path: str | None = None
    toml_inspection: dict[str, Any] | None = None
    sample_schema: dict[str, Any] | None = None
    if toml_config_path is None or not str(toml_config_path).strip():
        missing_fields.append("toml_config_path")
    else:
        resolved_toml_path = str(Path(toml_config_path).expanduser().resolve())
        try:
            toml_inspection = inspect_tx_split_toml_file(
                toml_path=resolved_toml_path,
                sample_files_per_dataset=1,
                inspect_sample_schemas=True,
            )
        except Exception as exc:
            validation_errors.append(f"Failed to inspect TOML config: {type(exc).__name__}: {exc}")
        else:
            for err in toml_inspection.get("errors", []):
                validation_errors.append(f"TOML validation: {err}")
            for warn in toml_inspection.get("warnings", []):
                validation_warnings.append(f"TOML warning: {warn}")

            datasets = toml_inspection.get("datasets", [])
            sample_files: list[str] = []
            if isinstance(datasets, list):
                for item in datasets:
                    if not isinstance(item, dict):
                        continue
                    dataset_sample_files = item.get("sample_files", [])
                    if isinstance(dataset_sample_files, list):
                        for sample_file in dataset_sample_files:
                            if isinstance(sample_file, str) and sample_file:
                                sample_files.append(sample_file)
            if sample_files:
                try:
                    sample_schema = inspect_adata_schema_file(sample_files[0], max_top_values=10)
                    suggestions["sample_adata_path"] = sample_files[0]
                except Exception as exc:
                    validation_warnings.append(
                        f"Failed to inspect sample AnnData schema ({sample_files[0]}): {type(exc).__name__}: {exc}"
                    )

    if output_dir is None or not str(output_dir).strip():
        missing_fields.append("output_dir")
        resolved_output_dir = None
    else:
        resolved_output_dir = str(Path(output_dir).expanduser().resolve())

    if name is None or not str(name).strip():
        missing_fields.append("name")
        resolved_name = None
    else:
        resolved_name = str(name).strip()

    candidate_columns = {}
    if isinstance(sample_schema, dict):
        candidate_columns = sample_schema.get("candidate_columns", {})
    if isinstance(candidate_columns, dict):
        suggestions["candidate_columns"] = candidate_columns

    def _pick_default(candidates: Any, fallback: str) -> str:
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return fallback

    resolved_perturbation_column = (
        perturbation_column.strip() if isinstance(perturbation_column, str) and perturbation_column.strip() else None
    )
    if resolved_perturbation_column is None:
        resolved_perturbation_column = _pick_default(
            candidate_columns.get("perturbation") if isinstance(candidate_columns, dict) else None,
            "gene",
        )

    resolved_cell_type_column = (
        cell_type_column.strip() if isinstance(cell_type_column, str) and cell_type_column.strip() else None
    )
    if resolved_cell_type_column is None:
        resolved_cell_type_column = _pick_default(
            candidate_columns.get("cell_type") if isinstance(candidate_columns, dict) else None,
            "cell_type",
        )

    resolved_batch_column = batch_column.strip() if isinstance(batch_column, str) and batch_column.strip() else None
    if resolved_batch_column is None:
        resolved_batch_column = _pick_default(
            candidate_columns.get("batch") if isinstance(candidate_columns, dict) else None,
            "gem_group",
        )

    resolved_embed_key = embed_key.strip() if isinstance(embed_key, str) and embed_key.strip() else None
    if isinstance(resolved_embed_key, str) and resolved_embed_key.lower() in {"null", "none"}:
        resolved_embed_key = None

    resolved_output_space = output_space.strip().lower() if isinstance(output_space, str) and output_space.strip() else None
    if resolved_output_space is not None:
        resolved_output_space = _OUTPUT_SPACE_ALIASES.get(resolved_output_space, resolved_output_space)
    if resolved_output_space is None:
        resolved_output_space = "gene" if resolved_embed_key in {None, "X_hvg"} else "all"
    if resolved_output_space not in {"gene", "all", "embedding"}:
        validation_errors.append(
            f"`output_space` must be one of 'gene', 'all', 'embedding' (aliases: 'hvg' -> 'gene', 'transcriptome' -> 'all'); got {resolved_output_space!r}."
        )
        resolved_output_space = "gene"

    resolved_control_pert = (
        control_perturbation.strip()
        if isinstance(control_perturbation, str) and control_perturbation.strip()
        else None
    )
    if resolved_control_pert is None:
        if resolved_perturbation_column == "drugname_drugconc":
            resolved_control_pert = "[('DMSO_TF', 0.0, 'uM')]"
        else:
            resolved_control_pert = "non-targeting"

    # --- Column validation against sample schema ---
    if sample_schema is not None:
        obs_column_names: set[str] = set()
        obsm_keys: set[str] = set()
        for _col_info in sample_schema.get("obs_columns", []):
            if isinstance(_col_info, dict) and isinstance(_col_info.get("name"), str):
                obs_column_names.add(_col_info["name"])
        for _obsm_info in sample_schema.get("obsm", []):
            if isinstance(_obsm_info, dict) and isinstance(_obsm_info.get("key"), str):
                obsm_keys.add(_obsm_info["key"])

        suggestions["available_obs_columns"] = sorted(obs_column_names)
        suggestions["available_obsm_keys"] = sorted(obsm_keys)

        _pert_user = isinstance(perturbation_column, str) and bool(perturbation_column.strip())
        _ct_user = isinstance(cell_type_column, str) and bool(cell_type_column.strip())
        _batch_user = isinstance(batch_column, str) and bool(batch_column.strip())
        _ctrl_user = isinstance(control_perturbation, str) and bool(control_perturbation.strip())

        for _param, _val, _user in [
            ("perturbation_column", resolved_perturbation_column, _pert_user),
            ("cell_type_column", resolved_cell_type_column, _ct_user),
            ("batch_column", resolved_batch_column, _batch_user),
        ]:
            if _val not in obs_column_names:
                _avail = ", ".join(sorted(obs_column_names)[:20])
                if _user:
                    validation_warnings.append(
                        f"`{_param}` = {_val!r} not found in sample data obs columns. "
                        f"Available: [{_avail}]"
                    )
                else:
                    missing_fields.append(_param)
                    suggestions[f"{_param}_reason"] = (
                        f"Auto-default {_val!r} not found in sample data. "
                        f"Available obs columns: [{_avail}]"
                    )

        if resolved_embed_key is not None and resolved_embed_key not in obsm_keys:
            _avail_obsm = ", ".join(sorted(obsm_keys)[:20])
            validation_warnings.append(
                f"`embed_key` = {resolved_embed_key!r} not found in sample data obsm keys. "
                f"Available: [{_avail_obsm}]"
            )

        # Validate control_perturbation against perturbation column values
        if resolved_perturbation_column in obs_column_names:
            _pert_top_values: set[str] = set()
            for _col_info in sample_schema.get("obs_columns", []):
                if isinstance(_col_info, dict) and _col_info.get("name") == resolved_perturbation_column:
                    for _tv in _col_info.get("top_values", []):
                        if isinstance(_tv, dict) and isinstance(_tv.get("value"), str):
                            _pert_top_values.add(_tv["value"])
                    break
            _ctrl_candidates = sample_schema.get("control_label_candidates", [])
            _known_ctrl = {
                c["value"] for c in _ctrl_candidates
                if isinstance(c, dict) and isinstance(c.get("value"), str)
            }
            if resolved_control_pert not in _pert_top_values and resolved_control_pert not in _known_ctrl:
                _suggestion_vals = [
                    c.get("value") for c in _ctrl_candidates[:5]
                    if isinstance(c, dict)
                ]
                if _ctrl_user:
                    validation_warnings.append(
                        f"`control_perturbation` = {resolved_control_pert!r} not found among "
                        f"top perturbation values or known control labels. "
                        f"Candidates: {_suggestion_vals}"
                    )
                else:
                    missing_fields.append("control_perturbation")
                    suggestions["control_perturbation_reason"] = (
                        f"Auto-default {resolved_control_pert!r} not found among "
                        f"top perturbation values or known control labels. "
                        f"Candidates: {_suggestion_vals}"
                    )

    wandb_enabled: bool
    wandb_reason: str
    if mode_wandb == "true":
        wandb_enabled = True
        wandb_reason = "Explicitly enabled by request."
    elif mode_wandb == "false":
        wandb_enabled = False
        wandb_reason = "Explicitly disabled by request."
    else:
        has_credentials, credentials_reason = _has_wandb_credentials()
        wandb_enabled = has_credentials
        wandb_reason = credentials_reason

    training_contract = _infer_tx_training_contract(resolved_embed_key, resolved_output_space)

    run_dir: str | None = None
    if resolved_output_dir and resolved_name:
        run_dir = str((Path(resolved_output_dir) / resolved_name).resolve())

    curated_overrides: list[str] = []
    curated_keys: set[str] = set()

    def _add_override(key: str, value: Any) -> None:
        curated_keys.add(key)
        curated_overrides.append(f"{key}={_hydra_format_value(value)}")

    if resolved_toml_path and resolved_output_dir and resolved_name and not validation_errors:
        _add_override("data.kwargs.toml_config_path", resolved_toml_path)
        _add_override("data.kwargs.embed_key", resolved_embed_key)
        _add_override("data.kwargs.output_space", resolved_output_space)
        _add_override("data.kwargs.pert_col", resolved_perturbation_column)
        _add_override("data.kwargs.cell_type_key", resolved_cell_type_column)
        _add_override("data.kwargs.batch_col", resolved_batch_column)
        if resolved_perturbation_column != "drugname_drugconc":
            _add_override("data.kwargs.control_pert", resolved_control_pert)
        _add_override("training.train_seed", seed)
        if max_steps is not None:
            _add_override("training.max_steps", max_steps)
        if batch_size is not None:
            _add_override("training.batch_size", batch_size)
        if learning_rate is not None:
            _add_override("training.lr", learning_rate)
        if val_freq is not None:
            _add_override("training.val_freq", val_freq)
        _add_override("model", model_preset_clean)
        _add_override("output_dir", resolved_output_dir)
        _add_override("name", resolved_name)
        _add_override("use_wandb", "true" if wandb_enabled else "false")

        # W&B overrides
        if wandb_project is not None:
            _add_override("wandb.project", wandb_project)
        if wandb_entity is not None:
            _add_override("wandb.entity", wandb_entity)
        if wandb_tags is not None:
            _add_override("wandb.tags", "[" + ",".join(wandb_tags) + "]")

        # Model architecture overrides
        if hidden_dim is not None:
            _add_override("model.kwargs.hidden_dim", hidden_dim)
        if cell_set_len is not None:
            _add_override("model.kwargs.cell_set_len", cell_set_len)
        if nb_loss is not None:
            _add_override("model.kwargs.nb_loss", str(nb_loss).lower())
        if batch_encoder is not None:
            _add_override("model.kwargs.batch_encoder", str(batch_encoder).lower())

        # Training overrides
        if ckpt_every_n_steps is not None:
            _add_override("training.ckpt_every_n_steps", ckpt_every_n_steps)

        # Data overrides
        if resolved_num_workers is not None:
            _add_override("data.kwargs.num_workers", resolved_num_workers)
        if use_consecutive_loading is not None:
            _add_override("data.kwargs.use_consecutive_loading", str(use_consecutive_loading).lower())

    for extra in cleaned_extra_overrides:
        extra_key = _normalized_override_key(extra)
        if extra_key is not None and extra_key in curated_keys:
            validation_warnings.append(
                f"Extra override {extra!r} overrides curated key {extra_key!r}; extra override takes precedence."
            )

    all_overrides = list(curated_overrides) + list(cleaned_extra_overrides)

    if validation_errors:
        status = "invalid"
    elif missing_fields:
        status = "needs_input"
    else:
        status = "ready"

    preset_defaults = _PRESET_MODEL_DEFAULTS.get(model_preset_clean, {})

    submission_hints: list[str] = []
    if max_steps is None:
        submission_hints.append(
            f"max_steps not set — Hydra default is {_DEFAULT_TRAINING_PARAMS['max_steps']}. "
            "Review before production runs."
        )
    if batch_size is None:
        submission_hints.append(
            f"batch_size not set — Hydra default is {_DEFAULT_TRAINING_PARAMS['batch_size']}. "
            "For large datasets consider increasing (e.g. 64–128)."
        )
    if learning_rate is None:
        submission_hints.append(
            f"learning_rate not set — Hydra default is {_DEFAULT_TRAINING_PARAMS['learning_rate']}."
        )
    if val_freq is None:
        submission_hints.append(
            f"val_freq not set — Hydra default is {_DEFAULT_TRAINING_PARAMS['val_freq']} steps."
        )
    if use_consecutive_loading is None:
        if resolved_output_space == "all":
            submission_hints.append(
                "use_consecutive_loading not set — defaults to False. "
                "RECOMMENDED for output_space='all' (full transcriptome): set True "
                "for significantly faster I/O. Requires data to be sorted by condition "
                "(training will fail early with a clear error if not)."
            )
        else:
            submission_hints.append(
                "use_consecutive_loading not set — defaults to False (random per-epoch loading). "
                "Set True when training on large concatenated h5ad files for more efficient sequential I/O."
            )
    if batch_encoder is None and preset_defaults.get("batch_encoder") is False:
        submission_hints.append(
            f"batch_encoder not set — '{model_preset_clean}' preset defaults to False. "
            "Set True to condition the model on batch labels."
        )
    if backend_mode == "slurm":
        if slurm_gpus is None:
            submission_hints.append(
                "slurm_gpus not set — a default of 1 GPU is applied at submission. "
                "Set explicitly to request more GPUs for larger models."
            )
        if slurm_partition is None:
            submission_hints.append(
                "slurm_partition not set — the job will use the cluster's default partition. "
                "Set to target a specific partition (e.g. 'gpu', 'preemptible')."
            )
        if slurm_time is None:
            submission_hints.append(
                "slurm_time not set — the job will use the partition's default time limit. "
                "Set to a specific limit (e.g. '2-00:00:00' for 2 days) for long training runs."
            )
        if slurm_mem is None:
            submission_hints.append(
                "slurm_mem not set — the job will use the partition's default memory allocation. "
                "Set explicitly (e.g. '256G') for large datasets."
            )
        if slurm_cpus_per_task is None:
            submission_hints.append(
                "slurm_cpus_per_task not set — also controls num_workers for data loading "
                "when num_workers is not set explicitly."
            )
    if wandb_enabled:
        wandb_detail_hints: list[str] = []
        if wandb_project is None:
            wandb_detail_hints.append("wandb_project")
        if wandb_entity is None:
            wandb_detail_hints.append("wandb_entity")
        if not wandb_tags:
            wandb_detail_hints.append("wandb_tags")
        if wandb_detail_hints:
            submission_hints.append(
                f"W&B logging is enabled ({wandb_reason.rstrip('.')}), but "
                f"{', '.join(wandb_detail_hints)} not set. Consider specifying these "
                "for better experiment organization and tracking."
            )
    if backend_mode == "local" and is_nvidia_smi_available():
        free = find_free_devices(1)
        if free:
            submission_hints.append(
                f"Local GPU detected. Least-loaded device is currently GPU {free[0]}. "
                "A free GPU will be auto-assigned via cuda_devices if not set explicitly."
            )
        else:
            submission_hints.append(
                "Local GPUs detected but none are currently free (all above utilization/memory "
                "thresholds). Consider waiting or explicitly setting cuda_devices."
            )

    return {
        "status": status,
        "missing_fields": sorted(set(missing_fields)),
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "suggestions": suggestions,
        "resolved": {
            "run_dir": run_dir,
            "toml_config_path": resolved_toml_path,
            "hydra_overrides": all_overrides,
            "training_contract_summary": training_contract,
            "wandb_resolution": {
                "enabled": wandb_enabled,
                "mode": mode_wandb,
                "reason": wandb_reason,
            },
            "backend_resolution": {
                "backend": backend_mode,
                "profile": backend_profile,
                "reason": backend_reason,
            },
            "resolved_columns": {
                "perturbation_column": resolved_perturbation_column,
                "cell_type_column": resolved_cell_type_column,
                "batch_column": resolved_batch_column,
                "control_perturbation": resolved_control_pert,
            },
            "resolved_slurm": {
                "partition": slurm_partition,
                "gpus": slurm_gpus,
                "gpus_effective": slurm_gpus if slurm_gpus is not None else (1 if backend_mode == "slurm" else None),
                "cpus_per_task": slurm_cpus_per_task,
                "mem": slurm_mem,
                "time": slurm_time,
            },
            "resolved_wandb": {
                "project": wandb_project,
                "entity": wandb_entity,
                "tags": wandb_tags,
            },
            "resolved_model": {
                "model_preset": model_preset_clean,
                "embed_key": resolved_embed_key,
                "output_space": resolved_output_space,
                "hidden_dim": hidden_dim,
                "cell_set_len": cell_set_len,
                "nb_loss": nb_loss,
                "batch_encoder": batch_encoder,
            },
            "resolved_training": {
                "max_steps": max_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "val_freq": val_freq,
                "seed": seed,
                "ckpt_every_n_steps": ckpt_every_n_steps,
            },
            "resolved_data": {
                "num_workers": resolved_num_workers,
                "use_consecutive_loading": use_consecutive_loading,
            },
        },
        "preset_defaults": preset_defaults,
        "submission_hints": submission_hints,
        "toml_inspection": toml_inspection,
    }


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

@mcp.tool()
def set_tx_model_folder(model_folder: str) -> dict[str, str]:
    """
    Set the default STATE TX model run folder for this MCP server session.

    The folder must contain STATE run artifacts:
    `config.yaml`, `data_module.{torch,pt,pkl}`, `pert_onehot_map.pt`, `var_dims.pkl`,
    batch/cell-type one-hot maps (`.torch`, `.pt`, or `.pkl`), and
    `checkpoints/` with at least one file.
    """
    resolved = normalize_and_validate_run_dir(model_folder)
    session_key = _get_current_session_key()
    with _SESSION_SERVER_STATE_LOCK:
        state = _get_session_server_state_locked(session_key)
        state["tx_model_folder"] = resolved
    return {"status": "ok", "tx_model_folder": resolved}


@mcp.tool()
def get_tx_model_folder() -> dict[str, str | None]:
    """
    Return the current default STATE TX model run folder for this MCP server session.
    """
    session_key = _get_current_session_key()
    with _SESSION_SERVER_STATE_LOCK:
        state = _get_session_server_state_locked(session_key)
        tx_model_folder = state["tx_model_folder"]
    return {"tx_model_folder": tx_model_folder}


@mcp.tool()
def clear_tx_model_folder() -> dict[str, str]:
    """
    Clear the current default STATE TX model run folder for this MCP server session.
    """
    session_key = _get_current_session_key()
    with _SESSION_SERVER_STATE_LOCK:
        state = _get_session_server_state_locked(session_key)
        state["tx_model_folder"] = None
    return {"status": "cleared"}


@mcp.tool()
def query_gpus() -> dict[str, Any]:
    """
    Query local NVIDIA GPU devices and return per-device status information.

    Returns device index, name, memory (total/used/free in MB), GPU utilization
    percentage, temperature, and PIDs of processes running on each device.

    Useful for checking GPU availability before launching local training or
    inference jobs with the `cuda_devices` parameter. Returns an empty device
    list if nvidia-smi is not available.
    """
    available = is_nvidia_smi_available()
    devices = query_gpu_devices() if available else []
    return {
        "nvidia_smi_available": available,
        "device_count": len(devices),
        "devices": devices,
    }


@mcp.tool()
def inspect_tx_checkpoint(
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
) -> dict[str, Any]:
    """
    Inspect a STATE TX checkpoint.

    If neither `checkpoint_path` nor `model_folder` is provided,
    uses the current TX default set via `set_tx_model_folder`.
    """
    session_key = _get_current_session_key()
    with _SESSION_SERVER_STATE_LOCK:
        default_model_folder = _get_session_server_state_locked(session_key)["tx_model_folder"]
    return inspect_tx_checkpoint_file(
        checkpoint_path=checkpoint_path,
        model_folder=model_folder,
        default_model_folder=default_model_folder,
    )


@mcp.tool()
def inspect_folder(
    model_folder: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """
    Inspect either a TX run folder/checkpoint or an EMB checkpoint.

    Routing behavior:
    - if `checkpoint_path` is provided, route by inferred checkpoint kind.
    - else if `model_folder` is a TX run directory, run TX folder inspection.
    - else if `model_folder` is an EMB checkpoint directory, run EMB checkpoint inspection.
    - else if `model_folder` is omitted, use TX default from `set_tx_model_folder`.
    """
    if checkpoint_path is not None and checkpoint_path.strip():
        resolved_checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        checkpoint_obj = Path(resolved_checkpoint)
        if not checkpoint_obj.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {resolved_checkpoint}")
        if resolved_checkpoint.endswith(".safetensors"):
            return inspect_emb_checkpoint_file(checkpoint_path=resolved_checkpoint)

        # Heuristic: TX checkpoints are commonly nested under `<run>/checkpoints/*.ckpt`.
        if checkpoint_obj.parent.name == "checkpoints":
            candidate_tx_run_dir = str(checkpoint_obj.parent.parent)
            try:
                normalize_and_validate_run_dir(candidate_tx_run_dir)
            except Exception:
                pass
            else:
                return inspect_tx_checkpoint_file(
                    checkpoint_path=resolved_checkpoint,
                    model_folder=candidate_tx_run_dir,
                )

        tx_error: Exception | None = None
        try:
            tx_result = inspect_tx_checkpoint_file(checkpoint_path=resolved_checkpoint)
        except Exception as exc:
            tx_error = exc
        else:
            kind = str(tx_result.get("inferred_checkpoint_kind") or "unknown")
            if kind == "state-transition":
                return tx_result
            if kind == "state-embedding":
                try:
                    return inspect_emb_checkpoint_file(checkpoint_path=resolved_checkpoint)
                except Exception:
                    # Some embedding .ckpt files require unavailable training modules
                    # to deserialize; prefer folder-level inspection fallback.
                    pass

        try:
            return inspect_emb_checkpoint_file(checkpoint_path=resolved_checkpoint)
        except Exception as emb_error:
            try:
                return inspect_emb_checkpoint_file(model_folder=str(checkpoint_obj.parent))
            except Exception as emb_folder_error:
                tx_msg = str(tx_error) if tx_error is not None else "unknown TX inspection failure"
                raise ValueError(
                    "Unable to inspect checkpoint as TX or EMB. "
                    f"TX error: {tx_msg}. EMB error: {emb_error}. "
                    f"EMB folder fallback error: {emb_folder_error}"
                ) from emb_error

    session_key = _get_current_session_key()
    with _SESSION_SERVER_STATE_LOCK:
        default_model_folder = _get_session_server_state_locked(session_key)["tx_model_folder"]

    candidate = model_folder or default_model_folder
    if candidate is None:
        raise ValueError(
            "No path provided. Pass `model_folder`/`checkpoint_path`, or set a TX default using `set_tx_model_folder`."
        )

    tx_error: Exception | None = None
    emb_error: Exception | None = None
    resolved_tx: str | None = None
    resolved_emb: str | None = None

    try:
        resolved_tx = resolve_and_validate_model_folder(candidate, None)
    except Exception as exc:
        tx_error = exc

    try:
        resolved_emb = normalize_and_validate_emb_inference_dir(candidate)
    except Exception as exc:
        emb_error = exc

    if resolved_tx is not None and resolved_emb is None:
        return inspect_model_folder(resolved_tx)
    if resolved_emb is not None and resolved_tx is None:
        return inspect_emb_checkpoint_file(model_folder=resolved_emb)
    if resolved_tx is not None and resolved_emb is not None:
        tx_result = inspect_model_folder(resolved_tx)
        tx_result.setdefault("warnings", [])
        tx_result["warnings"].append(
            "Path also matched an EMB checkpoint directory shape. "
            "Defaulting to TX run-folder inspection."
        )
        return tx_result

    tx_msg = str(tx_error) if tx_error is not None else "unknown TX validation failure"
    emb_msg = str(emb_error) if emb_error is not None else "unknown EMB validation failure"
    raise ValueError(
        "Path is neither a valid TX run directory nor a valid EMB checkpoint directory. "
        f"TX validation error: {tx_msg}. EMB validation error: {emb_msg}"
    )


@mcp.tool()
def inspect_emb_checkpoint(
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
) -> dict[str, Any]:
    """
    Inspect a STATE embedding checkpoint (single-file artifact), distinct from TX run-folder inspection.

    Provide either:
    - `checkpoint_path` to inspect a specific `.ckpt`/`.safetensors` file, or
    - `model_folder` to auto-select an embedding checkpoint from that directory.
    If both are provided, `checkpoint_path` takes precedence.
    """
    return inspect_emb_checkpoint_file(
        checkpoint_path=checkpoint_path,
        model_folder=model_folder,
    )


@mcp.tool()
def inspect_adata_schema(adata_path: str, max_top_values: int = 25) -> dict[str, Any]:
    """
    Inspect an AnnData file and return schema metadata useful for guided ST training setup.
    """
    return inspect_adata_schema_file(
        adata_path=adata_path,
        max_top_values=max_top_values,
    )


@mcp.tool()
def inspect_tx_split_toml(
    toml_path: str,
    sample_files_per_dataset: int = 1,
    inspect_sample_schemas: bool = True,
) -> dict[str, Any]:
    """
    Inspect a TX split TOML file and summarize datasets, split sections, and sample schema compatibility.
    """
    return inspect_tx_split_toml_file(
        toml_path=toml_path,
        sample_files_per_dataset=sample_files_per_dataset,
        inspect_sample_schemas=inspect_sample_schemas,
    )


@mcp.tool()
def inspect_tx_split_sources(
    dataset_paths: dict[str, str],
    cell_type_column: str | None = None,
    perturbation_column: str | None = None,
    control_perturbation: str | None = None,
    max_top_perturbations: int = 10,
    max_contexts: int = 200,
) -> dict[str, Any]:
    """
    Inspect dataset paths for split authoring and return context-level perturbation summaries.

    This is intended as a guided precursor to TOML generation.
    """
    return inspect_tx_split_sources_file(
        dataset_paths=dataset_paths,
        cell_type_column=cell_type_column,
        perturbation_column=perturbation_column,
        control_perturbation=control_perturbation,
        max_top_perturbations=max_top_perturbations,
        max_contexts=max_contexts,
    )


@mcp.tool()
def plan_tx_split_toml(
    dataset_paths: dict[str, str],
    training_datasets: list[str] | None = None,
    cell_type_column: str | None = None,
    perturbation_column: str | None = None,
    control_perturbation: str | None = None,
    random_holdout_contexts: list[str] | None = None,
    random_test_fraction: float = 0.7,
    random_val_fraction: float = 0.0,
    random_seed: int = 0,
    zeroshot_contexts: dict[str, str] | None = None,
    fewshot_overrides: dict[str, dict[str, list[str]]] | None = None,
    output_path: str | None = None,
    overwrite: bool = False,
    include_toml: bool = False,
    preview_lines: int = 120,
    max_contexts_in_summary: int = 200,
) -> dict[str, Any]:
    """
    Build a TX split TOML from dataset paths and split directives.

    Supports random fewshot holdouts per context, plus zeroshot and manual fewshot overrides.
    """
    return plan_tx_split_toml_file(
        dataset_paths=dataset_paths,
        training_datasets=training_datasets,
        cell_type_column=cell_type_column,
        perturbation_column=perturbation_column,
        control_perturbation=control_perturbation,
        random_holdout_contexts=random_holdout_contexts,
        random_test_fraction=random_test_fraction,
        random_val_fraction=random_val_fraction,
        random_seed=random_seed,
        zeroshot_contexts=zeroshot_contexts,
        fewshot_overrides=fewshot_overrides,
        output_path=output_path,
        overwrite=overwrite,
        include_toml=include_toml,
        preview_lines=preview_lines,
        max_contexts_in_summary=max_contexts_in_summary,
    )


@mcp.tool()
def plan_tx_train(
    toml_config_path: str | None = None,
    output_dir: str | None = None,
    name: str | None = None,
    embed_key: str | None = None,
    output_space: str | None = None,
    perturbation_column: str | None = None,
    cell_type_column: str | None = None,
    batch_column: str | None = None,
    control_perturbation: str | None = None,
    model_preset: str = "state",
    max_steps: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    val_freq: int | None = None,
    seed: int = 42,
    use_wandb: str = "auto",
    backend: str = "auto",
    backend_profile: str | None = None,
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
    hidden_dim: int | None = None,
    cell_set_len: int | None = None,
    nb_loss: bool | None = None,
    batch_encoder: bool | None = None,
    ckpt_every_n_steps: int | None = None,
    num_workers: int | None = None,
    use_consecutive_loading: bool | None = None,
    extra_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build and validate a curated TX training plan without launching training.

    Returns `status`:
    - `needs_input`: required fields are missing — supply them and call again
    - `invalid`: values failed validation — check `validation_errors`
    - `ready`: launch-ready plan with resolved Hydra overrides

    The response also includes:
    - `preset_defaults`: key model kwargs from the chosen preset YAML (hidden_dim,
      cell_set_len, batch_encoder, nb_loss) — review before submitting.
    - `submission_hints`: list of plain-language nudges for parameters that were
      left at Hydra defaults and are commonly worth reviewing before a production run.
      Includes GPU allocation reminders for Slurm backends and W&B organization
      details (project, entity, tags) when W&B credentials are detected.

    **Required fields** (status stays `needs_input` until all are set):
    - `toml_config_path`: path to the TX split TOML produced by `plan_tx_split_toml`
    - `output_dir`: parent directory for the run folder
    - `name`: run name (a subdirectory `output_dir/name` is created)

    **Data / column parameters:**
    - `embed_key`: obsm key for pre-computed embeddings (e.g. `"X_state"`).
      Set to `"null"` or omit to train without embeddings.
    - `output_space`: `"gene"` (HVG expression only, default when no embed_key),
      `"all"` (full transcriptome / all genes, default when embed_key is set),
      or `"embedding"` (embedding-space only).
      Aliases: `"hvg"` -> `"gene"`, `"transcriptome"` -> `"all"`.
    - `perturbation_column`: obs column for perturbation labels (default auto-detected, fallback `"gene"`)
    - `cell_type_column`: obs column for cell-type labels (default auto-detected, fallback `"cell_type"`)
    - `batch_column`: obs column for batch labels (default auto-detected, fallback `"gem_group"`)
    - `control_perturbation`: label for control/unperturbed cells (default `"non-targeting"`)
    - `use_consecutive_loading`: if True, load h5ad files sequentially rather than randomly
      across epochs — more efficient I/O for large concatenated files (default False).

    **Model presets** (`model_preset`):
    - `"state"` (default): hidden_dim=384, cell_set_len=64, 8 transformer layers, 12 heads
    - `"state_sm"`: hidden_dim=672, cell_set_len=128, 4 transformer layers, 8 heads
    - `"state_lg"`: hidden_dim=1488, cell_set_len=512, 6 transformer layers, 12 heads
    - `"context_mean"` (alias `"cmean"`): hidden_dim=512, cell_set_len=512, GPT-2 backbone
    - `"perturb_mean"` (alias `"pmean"`): hidden_dim=512, cell_set_len=512, GPT-2 backbone
    Preset defaults are returned in `preset_defaults` so you can confirm them before launch.

    **Model architecture overrides** (override preset defaults):
    - `hidden_dim`: transformer hidden dimension (must be > 0)
    - `cell_set_len`: number of cells per set/context (must be > 0)
    - `nb_loss`: use negative binomial loss (default false)
    - `batch_encoder`: enable batch encoder to condition on batch labels (default false)

    **Training parameters:**
    - `max_steps`: maximum training steps (Hydra default: 100000)
    - `batch_size`: training batch size (Hydra default: 16; consider 64–128 for large datasets)
    - `learning_rate`: learning rate (Hydra default: 1e-4)
    - `val_freq`: validation frequency in steps (Hydra default: 2000)
    - `seed`: random seed (default 42)
    - `ckpt_every_n_steps`: checkpoint save frequency in steps (must be > 0)

    **Data parameters:**
    - `num_workers`: dataloader worker count (Hydra default: 12; must be >= 0).
      Auto-set from `slurm_cpus_per_task` if not provided explicitly.
    - `use_consecutive_loading`: see Data / column parameters above.

    **W&B parameters** (use these dedicated params — do NOT set via `extra_overrides`):
    - `use_wandb`: `"auto"` (detect credentials), `"true"`, or `"false"`
    - `wandb_project`: W&B project name (Hydra key `wandb.project`)
    - `wandb_entity`: W&B entity/team name (Hydra key `wandb.entity`)
    - `wandb_tags`: list of W&B tags (Hydra key `wandb.tags`)

    **Backend / Slurm parameters:**
    - `backend`: `"auto"` (detect sbatch), `"slurm"`, or `"local"`
    - `backend_profile`: slurm profile name, partition, or raw sbatch args
    - `slurm_partition`: slurm partition(s), e.g. `"standard,preemptible"` (maps to `--partition`)
    - `slurm_gpus`: number of GPUs (maps to `--gres=gpu:N`)
    - `slurm_cpus_per_task`: CPUs per task (maps to `--cpus-per-task`)
    - `slurm_mem`: memory request, e.g. `"256G"` (maps to `--mem`)
    - `slurm_time`: time limit, e.g. `"2-00:00:00"` (maps to `--time`)

    **Advanced:**
    - `extra_overrides`: raw Hydra overrides appended after all curated ones.
      Use only for parameters that have no dedicated first-class argument above.
      A warning is emitted if an extra override duplicates a curated key.
    """
    return _build_tx_train_plan(
        toml_config_path=toml_config_path,
        output_dir=output_dir,
        name=name,
        embed_key=embed_key,
        output_space=output_space,
        perturbation_column=perturbation_column,
        cell_type_column=cell_type_column,
        batch_column=batch_column,
        control_perturbation=control_perturbation,
        model_preset=model_preset,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_freq=val_freq,
        seed=seed,
        use_wandb=use_wandb,
        backend=backend,
        backend_profile=backend_profile,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=wandb_tags,
        hidden_dim=hidden_dim,
        cell_set_len=cell_set_len,
        nb_loss=nb_loss,
        batch_encoder=batch_encoder,
        ckpt_every_n_steps=ckpt_every_n_steps,
        num_workers=num_workers,
        use_consecutive_loading=use_consecutive_loading,
        extra_overrides=extra_overrides,
    )


@mcp.tool()
def run_tx_train(
    toml_config_path: str | None = None,
    output_dir: str | None = None,
    name: str | None = None,
    embed_key: str | None = None,
    output_space: str | None = None,
    perturbation_column: str | None = None,
    cell_type_column: str | None = None,
    batch_column: str | None = None,
    control_perturbation: str | None = None,
    model_preset: str = "state",
    max_steps: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    val_freq: int | None = None,
    seed: int = 42,
    use_wandb: str = "auto",
    backend: str = "auto",
    backend_profile: str | None = None,
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
    hidden_dim: int | None = None,
    cell_set_len: int | None = None,
    nb_loss: bool | None = None,
    batch_encoder: bool | None = None,
    ckpt_every_n_steps: int | None = None,
    num_workers: int | None = None,
    use_consecutive_loading: bool | None = None,
    extra_overrides: list[str] | None = None,
    idempotency_key: str | None = None,
    cuda_devices: str | None = None,
) -> dict[str, Any]:
    """
    Start TX training asynchronously. Legacy alias for `start_tx_train`.
    """
    return start_tx_train(
        toml_config_path=toml_config_path,
        output_dir=output_dir,
        name=name,
        embed_key=embed_key,
        output_space=output_space,
        perturbation_column=perturbation_column,
        cell_type_column=cell_type_column,
        batch_column=batch_column,
        control_perturbation=control_perturbation,
        model_preset=model_preset,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_freq=val_freq,
        seed=seed,
        use_wandb=use_wandb,
        backend=backend,
        backend_profile=backend_profile,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=wandb_tags,
        hidden_dim=hidden_dim,
        cell_set_len=cell_set_len,
        nb_loss=nb_loss,
        batch_encoder=batch_encoder,
        ckpt_every_n_steps=ckpt_every_n_steps,
        num_workers=num_workers,
        use_consecutive_loading=use_consecutive_loading,
        extra_overrides=extra_overrides,
        idempotency_key=idempotency_key,
        cuda_devices=cuda_devices,
    )


@mcp.tool()
def start_tx_train(
    toml_config_path: str | None = None,
    output_dir: str | None = None,
    name: str | None = None,
    embed_key: str | None = None,
    output_space: str | None = None,
    perturbation_column: str | None = None,
    cell_type_column: str | None = None,
    batch_column: str | None = None,
    control_perturbation: str | None = None,
    model_preset: str = "state",
    max_steps: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    val_freq: int | None = None,
    seed: int = 42,
    use_wandb: str = "auto",
    backend: str = "auto",
    backend_profile: str | None = None,
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
    hidden_dim: int | None = None,
    cell_set_len: int | None = None,
    nb_loss: bool | None = None,
    batch_encoder: bool | None = None,
    ckpt_every_n_steps: int | None = None,
    num_workers: int | None = None,
    use_consecutive_loading: bool | None = None,
    extra_overrides: list[str] | None = None,
    idempotency_key: str | None = None,
    cuda_devices: str | None = None,
) -> dict[str, Any]:
    """
    Start TX training in the background and return a `job_id` immediately.

    Recommended workflow:
    1) call `plan_tx_train` until status is `ready`
    2) review `submission_hints` and `preset_defaults` in the plan response
    3) call `start_tx_train`
    4) poll via `get_tx_train_status` and `get_tx_train_logs`
    5) optionally cancel via `cancel_tx_train`
    """
    plan = _build_tx_train_plan(
        toml_config_path=toml_config_path,
        output_dir=output_dir,
        name=name,
        embed_key=embed_key,
        output_space=output_space,
        perturbation_column=perturbation_column,
        cell_type_column=cell_type_column,
        batch_column=batch_column,
        control_perturbation=control_perturbation,
        model_preset=model_preset,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_freq=val_freq,
        seed=seed,
        use_wandb=use_wandb,
        backend=backend,
        backend_profile=backend_profile,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_tags=wandb_tags,
        hidden_dim=hidden_dim,
        cell_set_len=cell_set_len,
        nb_loss=nb_loss,
        batch_encoder=batch_encoder,
        ckpt_every_n_steps=ckpt_every_n_steps,
        num_workers=num_workers,
        use_consecutive_loading=use_consecutive_loading,
        extra_overrides=extra_overrides,
    )

    if str(plan.get("status")) != "ready":
        return {
            "status": "not_started",
            "reason": "plan_not_ready",
            "plan": plan,
        }

    resolved = plan.get("resolved")
    if not isinstance(resolved, dict):
        raise RuntimeError("Training plan is missing resolved launch information.")

    hydra_overrides = resolved.get("hydra_overrides")
    if not isinstance(hydra_overrides, list) or not hydra_overrides:
        raise RuntimeError("Training plan did not produce launch overrides.")
    resolved_overrides = [str(item) for item in hydra_overrides]

    backend_info = resolved.get("backend_resolution")
    if not isinstance(backend_info, dict):
        backend_info = {}
    backend_mode = str(backend_info.get("backend") or "local").strip().lower()
    profile_value = backend_profile
    if profile_value is None:
        backend_profile_from_plan = backend_info.get("profile")
        if isinstance(backend_profile_from_plan, str):
            profile_value = backend_profile_from_plan

    resolved_run_dir = resolved.get("run_dir")
    run_dir = str(resolved_run_dir) if isinstance(resolved_run_dir, str) and resolved_run_dir.strip() else None

    cleaned_idempotency_key = None
    if isinstance(idempotency_key, str) and idempotency_key.strip():
        cleaned_idempotency_key = idempotency_key.strip()
    session_key = _get_current_session_key()

    with _TRAIN_JOBS_LOCK:
        train_jobs_by_id = _get_session_train_jobs_locked(session_key)
        idempotency_by_key = _get_session_train_idempotency_locked(session_key)
        if cleaned_idempotency_key is not None:
            existing_job_id = idempotency_by_key.get(cleaned_idempotency_key)
            if isinstance(existing_job_id, str):
                existing_job = train_jobs_by_id.get(existing_job_id)
                if isinstance(existing_job, TrainJob):
                    _sync_train_job_state_locked(existing_job)
                    if existing_job.status not in _TERMINAL_JOB_STATUSES:
                        return {
                            "status": "already_started",
                            "job_id": existing_job.job_id,
                            "backend": existing_job.backend,
                            "scheduler_job_id": existing_job.scheduler_job_id,
                            "worker_pid": existing_job.worker_pid,
                            "worker_log_path": existing_job.worker_log_path,
                            "worker_error_log_path": existing_job.worker_error_log_path,
                            "recommended_poll_interval_seconds": _recommend_train_poll_interval_seconds(existing_job),
                        }

    job_id = uuid4().hex
    job = TrainJob(
        job_id=job_id,
        status="queued",
        created_at=_utc_now_iso(),
        run_dir=run_dir,
        backend=backend_mode,
        backend_profile=profile_value,
        hydra_overrides=resolved_overrides,
        resolved_plan=resolved,
        idempotency_key=cleaned_idempotency_key,
    )
    _append_train_job_log(job, "[queued] Training job queued.")

    with _TRAIN_JOBS_LOCK:
        train_jobs_by_id = _get_session_train_jobs_locked(session_key)
        train_jobs_by_id[job_id] = job
        if cleaned_idempotency_key is not None:
            idempotency_by_key = _get_session_train_idempotency_locked(session_key)
            idempotency_by_key[cleaned_idempotency_key] = job_id

    if backend_mode == "slurm":
        resolved_slurm = resolved.get("resolved_slurm") or {}
        try:
            submission = _submit_tx_train_slurm_job(
                job_id=job_id,
                run_dir=run_dir,
                hydra_overrides=resolved_overrides,
                backend_profile=profile_value,
                slurm_partition=resolved_slurm.get("partition"),
                slurm_gpus=resolved_slurm.get("gpus"),
                slurm_cpus_per_task=resolved_slurm.get("cpus_per_task"),
                slurm_mem=resolved_slurm.get("mem"),
                slurm_time=resolved_slurm.get("time"),
            )
        except Exception as exc:
            with _TRAIN_JOBS_LOCK:
                current = _get_session_train_jobs_locked(session_key).get(job_id)
                if current is not None:
                    current.status = "failed"
                    current.error = f"Failed to submit slurm job: {type(exc).__name__}: {exc}"
                    current.finished_at = _utc_now_iso()
                    _append_train_job_log(current, f"[failed] {current.error}")
            raise RuntimeError(f"Unable to submit tx train slurm job: {type(exc).__name__}: {exc}") from exc

        with _TRAIN_JOBS_LOCK:
            current = _get_session_train_jobs_locked(session_key).get(job_id)
            if current is not None:
                current.scheduler_job_id = str(submission["scheduler_job_id"])
                current.worker_log_path = str(submission["worker_log_path"])
                current.worker_error_log_path = str(submission["worker_error_log_path"])
                current.progress = {
                    "phase": "submitted",
                    "message": "Submitted to slurm.",
                    "scheduler_job_id": current.scheduler_job_id,
                    "slurm_profile_reason": submission.get("profile_reason"),
                }
                _append_train_job_log(current, f"[submitted] Slurm job id={current.scheduler_job_id}.")
                _sync_train_job_state_locked(current)
                poll_interval = _recommend_train_poll_interval_seconds(current)
            else:
                poll_interval = 5.0

        return {
            "status": "started",
            "job_id": job_id,
            "backend": "slurm",
            "scheduler_job_id": submission["scheduler_job_id"],
            "worker_log_path": submission["worker_log_path"],
            "worker_error_log_path": submission["worker_error_log_path"],
            "run_dir": run_dir,
            "resolved": resolved,
            "recommended_initial_poll_interval_seconds": poll_interval,
        }

    resolved_cuda_devices, cuda_auto_assigned = _resolve_cuda_devices_for_local(cuda_devices, backend_mode)

    mp_ctx = _get_worker_mp_context()
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    cancel_flag_path = str((Path("/tmp") / f"state_tx_train_cancel_{job_id}.flag").resolve())
    if run_dir is not None:
        worker_log_path = str((Path(run_dir) / f"mcp_tx_train_worker_{job_id}.log").resolve())
    else:
        worker_log_path = str((Path("/tmp") / f"state_tx_train_worker_{job_id}.log").resolve())
    Path(cancel_flag_path).unlink(missing_ok=True)
    process = mp_ctx.Process(
        target=_run_tx_train_job_worker,
        args=(resolved_overrides, cancel_flag_path, child_conn, worker_log_path, resolved_cuda_devices),
        daemon=False,
        name=f"state_tx_train_{job_id[:8]}",
    )

    with _TRAIN_JOBS_LOCK:
        current = _get_session_train_jobs_locked(session_key).get(job_id)
        if current is not None:
            current.event_conn = parent_conn
            current.cancel_flag_path = cancel_flag_path
            current.worker_log_path = worker_log_path
            current.process = process

    try:
        process.start()
        try:
            child_conn.close()
        except Exception:
            pass
    except Exception as exc:
        with _TRAIN_JOBS_LOCK:
            current = _get_session_train_jobs_locked(session_key).get(job_id)
            if current is not None:
                current.status = "failed"
                current.error = f"Failed to launch worker process: {type(exc).__name__}: {exc}"
                current.finished_at = _utc_now_iso()
                _append_train_job_log(current, f"[failed] {current.error}")
                _release_train_job_runtime_resources(current)
        try:
            child_conn.close()
        except Exception:
            pass
        raise RuntimeError(f"Unable to start tx train worker process: {type(exc).__name__}: {exc}") from exc

    with _TRAIN_JOBS_LOCK:
        current = _get_session_train_jobs_locked(session_key).get(job_id)
        if current is not None:
            current.worker_pid = process.pid
            _sync_train_job_state_locked(current)
            poll_interval = _recommend_train_poll_interval_seconds(current)
        else:
            poll_interval = 5.0

    return {
        "status": "started",
        "job_id": job_id,
        "backend": "local",
        "run_dir": run_dir,
        "resolved": resolved,
        "worker_pid": process.pid,
        "worker_log_path": worker_log_path,
        "cuda_devices": resolved_cuda_devices,
        "cuda_auto_assigned": cuda_auto_assigned,
        "recommended_initial_poll_interval_seconds": poll_interval,
    }


@mcp.tool()
def get_tx_train_status(job_id: str) -> dict[str, Any]:
    """
    Return status, progress, and adaptive polling hints for a background tx training job.
    """
    session_key = _get_current_session_key()
    with _TRAIN_JOBS_LOCK:
        train_jobs_by_id = _get_session_train_jobs_locked(session_key)
        job = _get_train_job_locked(job_id, train_jobs_by_id)
        _sync_train_job_state_locked(job)
        progress = dict(job.progress)
        seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
        recommended_poll = _recommend_train_poll_interval_seconds(job)
        recommended_log_poll = _recommend_train_poll_interval_seconds(job, for_logs=True)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "backend": job.backend,
            "backend_profile": job.backend_profile,
            "scheduler_job_id": job.scheduler_job_id,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "last_update_at": job.last_update_at,
            "seconds_since_last_update": round(seconds_since_update, 3),
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "error": job.error,
            "run_dir": job.run_dir,
            "hydra_overrides": list(job.hydra_overrides),
            "resolved_plan": dict(job.resolved_plan),
            "progress": progress,
            "last_event": dict(job.last_event) if isinstance(job.last_event, dict) else None,
            "log_line_count": len(job.logs),
            "worker_pid": job.worker_pid,
            "worker_exit_code": job.worker_exit_code,
            "worker_log_path": job.worker_log_path,
            "worker_error_log_path": job.worker_error_log_path,
            "idempotency_key": job.idempotency_key,
            "recommended_poll_interval_seconds": recommended_poll,
            "recommended_log_poll_interval_seconds": recommended_log_poll,
        }


@mcp.tool()
def get_tx_train_logs(job_id: str, from_line: int = 0, max_lines: int = 200) -> dict[str, Any]:
    """
    Return buffered logs for a background tx training job plus a polling hint.
    """
    if from_line < 0:
        raise ValueError("`from_line` must be >= 0.")
    if max_lines <= 0:
        raise ValueError("`max_lines` must be > 0.")

    session_key = _get_current_session_key()
    with _TRAIN_JOBS_LOCK:
        train_jobs_by_id = _get_session_train_jobs_locked(session_key)
        job = _get_train_job_locked(job_id, train_jobs_by_id)
        _sync_train_job_state_locked(job)
        start = min(from_line, len(job.logs))
        end = min(start + max_lines, len(job.logs))
        lines = job.logs[start:end]
        recommended_poll = _recommend_train_poll_interval_seconds(job, for_logs=True)

    return {
        "job_id": job_id,
        "from_line": start,
        "next_line": end,
        "total_lines": len(job.logs),
        "recommended_poll_interval_seconds": recommended_poll,
        "lines": lines,
    }


@mcp.tool()
def cancel_tx_train(job_id: str, force: bool = False) -> dict[str, Any]:
    """
    Request cancellation for a background tx training job.

    For local jobs, a graceful cancellation flag is set first; if `force=True`,
    SIGTERM is sent immediately. For slurm jobs, cancellation sends `scancel`.
    """
    session_key = _get_current_session_key()
    with _TRAIN_JOBS_LOCK:
        train_jobs_by_id = _get_session_train_jobs_locked(session_key)
        job = _get_train_job_locked(job_id, train_jobs_by_id)
        _sync_train_job_state_locked(job)
        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "job_id": job_id,
                "status": job.status,
                "cancel_requested": job.cancel_requested,
                "message": "Job is already in a terminal state.",
            }

        now_iso = _utc_now_iso()
        now_mono = time.monotonic()
        first_request = not job.cancel_requested

        job.cancel_requested = True
        if first_request:
            job.cancel_requested_at = now_iso
            job.cancel_grace_deadline_monotonic = now_mono + _CANCEL_GRACE_SECONDS
        if job.status in {"queued", "running"}:
            job.status = "cancelling"

        terminate_sent = False
        scancel_sent = False
        scancel_exit_code: int | None = None
        scancel_message: str | None = None

        if job.backend == "slurm":
            if isinstance(job.scheduler_job_id, str) and job.scheduler_job_id.strip():
                if shutil.which("scancel") is None:
                    scancel_message = "`scancel` not found in PATH."
                    _append_train_job_log(job, f"[warning] {scancel_message}")
                else:
                    try:
                        result = subprocess.run(
                            ["scancel", job.scheduler_job_id],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        scancel_exit_code = int(result.returncode)
                        out = (result.stdout or "").strip()
                        err = (result.stderr or "").strip()
                        if result.returncode == 0:
                            scancel_sent = True
                            scancel_message = "scancel request submitted."
                            _append_train_job_log(job, f"[cancelling] Sent scancel for {job.scheduler_job_id}.")
                        else:
                            scancel_message = f"scancel failed with code {result.returncode}. stdout={out!r} stderr={err!r}"
                            _append_train_job_log(job, f"[warning] {scancel_message}")
                    except Exception as exc:
                        scancel_message = f"Failed to invoke scancel: {type(exc).__name__}: {exc}"
                        _append_train_job_log(job, f"[warning] {scancel_message}")
            else:
                scancel_message = "No scheduler job id recorded; cannot send scancel."
                _append_train_job_log(job, f"[warning] {scancel_message}")
        else:
            if isinstance(job.cancel_flag_path, str) and job.cancel_flag_path:
                try:
                    Path(job.cancel_flag_path).touch(exist_ok=True)
                except Exception as exc:
                    _append_train_job_log(
                        job,
                        f"[warning] Failed to set cancellation flag at {job.cancel_flag_path}: {type(exc).__name__}: {exc}",
                    )

            if force and _process_alive(job.process):
                try:
                    job.process.terminate()
                    terminate_sent = True
                    job.terminate_sent_at_monotonic = now_mono
                    _append_train_job_log(job, "[cancelling] Force cancellation requested; sent SIGTERM to worker process.")
                except Exception as exc:
                    _append_train_job_log(
                        job, f"[warning] Failed to force-cancel worker process: {type(exc).__name__}: {exc}"
                    )
            elif first_request:
                _append_train_job_log(
                    job,
                    f"[cancelling] Cancellation requested. Grace period is {_CANCEL_GRACE_SECONDS:.0f}s before force termination.",
                )
            else:
                _append_train_job_log(job, "[cancelling] Cancellation requested.")

        _sync_train_job_state_locked(job)
        poll_interval = _recommend_train_poll_interval_seconds(job)
        return {
            "job_id": job_id,
            "status": job.status,
            "backend": job.backend,
            "scheduler_job_id": job.scheduler_job_id,
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "force": force,
            "terminate_sent": terminate_sent,
            "scancel_sent": scancel_sent,
            "scancel_exit_code": scancel_exit_code,
            "scancel_message": scancel_message,
            "recommended_poll_interval_seconds": poll_interval,
        }


@mcp.tool()
def list_tx_train_jobs(status: str | None = None, limit: int = 100) -> dict[str, Any]:
    """
    List background tx training jobs tracked by this MCP server session.
    """
    if limit <= 0:
        raise ValueError("`limit` must be > 0.")

    status_filter = status.strip().lower() if isinstance(status, str) and status.strip() else None
    session_key = _get_current_session_key()

    with _TRAIN_JOBS_LOCK:
        jobs = list(_get_session_train_jobs_locked(session_key).values())
        for job in jobs:
            _sync_train_job_state_locked(job)

        jobs_sorted = sorted(jobs, key=lambda item: item.created_at, reverse=True)
        if status_filter is not None:
            jobs_sorted = [job for job in jobs_sorted if str(job.status).lower() == status_filter]

        selected = jobs_sorted[:limit]
        payload = [
            {
                "job_id": job.job_id,
                "status": job.status,
                "backend": job.backend,
                "scheduler_job_id": job.scheduler_job_id,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "run_dir": job.run_dir,
                "worker_pid": job.worker_pid,
                "worker_exit_code": job.worker_exit_code,
                "worker_log_path": job.worker_log_path,
                "worker_error_log_path": job.worker_error_log_path,
                "cancel_requested": job.cancel_requested,
                "error": job.error,
                "idempotency_key": job.idempotency_key,
            }
            for job in selected
        ]

    return {
        "jobs": payload,
        "total": len(jobs_sorted),
        "returned": len(payload),
    }


@mcp.tool()
def run_emb_inference(
    input_adata_path: str,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
    config_path: str | None = None,
    embedding_key: str = "X_state",
    protein_embeddings_path: str | None = None,
    batch_size: int | None = None,
    backend: str = "auto",
    backend_profile: str | None = None,
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    cuda_devices: str | None = None,
) -> dict[str, Any]:
    """
    Start STATE embedding inference in the background and return a `job_id` immediately.

    This preserves the legacy tool name while making EMB inference asynchronous.
    Poll status with `get_emb_inference_status`, fetch logs with
    `get_emb_inference_logs`, and cancel with `cancel_emb_inference`.
    """
    return start_emb_inference(
        input_adata_path=input_adata_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        model_folder=model_folder,
        config_path=config_path,
        embedding_key=embedding_key,
        protein_embeddings_path=protein_embeddings_path,
        batch_size=batch_size,
        backend=backend,
        backend_profile=backend_profile,
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
        cuda_devices=cuda_devices,
    )

@mcp.tool()
def start_emb_inference(
    input_adata_path: str,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
    config_path: str | None = None,
    embedding_key: str = "X_state",
    protein_embeddings_path: str | None = None,
    batch_size: int | None = None,
    backend: str = "auto",
    backend_profile: str | None = None,
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    cuda_devices: str | None = None,
) -> dict[str, Any]:
    """
    Start embedding inference in the background and return a `job_id` immediately.
    Poll status with `get_emb_inference_status`, fetch logs with `get_emb_inference_logs`,
    and cancel with `cancel_emb_inference`.

    Returns poll-hint metadata (`recommended_initial_poll_interval_seconds`) for
    clients that want adaptive backoff on long-running jobs.
    """
    resolved = _resolve_emb_inference_request(
        input_adata_path=input_adata_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        model_folder=model_folder,
        config_path=config_path,
        embedding_key=embedding_key,
        protein_embeddings_path=protein_embeddings_path,
        batch_size=batch_size,
    )
    session_key = _get_current_session_key()

    backend_mode, backend_reason = _resolve_backend_mode(backend)

    job_id = uuid4().hex
    job = InferenceJob(
        job_id=job_id,
        status="queued",
        created_at=_utc_now_iso(),
        model_folder=resolved["model_folder"],
        adata_path=resolved["input_adata_path"],
        output_path=resolved["output_path"],
        checkpoint_path=resolved["checkpoint_path"],
        inference_kind="emb",
        resolved_args={
            "config_path": resolved["config_path"],
            "embedding_key": resolved["embedding_key"],
            "protein_embeddings_path": resolved["protein_embeddings_path"],
            "batch_size": resolved["batch_size"],
            "output_mode": resolved["output_mode"],
            "output_adata_path_for_encode": resolved["output_adata_path_for_encode"],
        },
        adata_size_bytes=resolved.get("input_adata_size_bytes"),
        backend=backend_mode,
        backend_profile=backend_profile,
    )
    _append_job_log(job, "[queued] Embedding inference job queued.")

    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        jobs_by_id[job_id] = job

    if backend_mode == "slurm":
        command = _build_emb_transform_cli_args(resolved)
        try:
            submission = _submit_slurm_job(
                job_id=job_id,
                job_name_prefix="state_emb_infer",
                run_dir=None,
                command=command,
                backend_profile=backend_profile,
                slurm_partition=slurm_partition,
                slurm_gpus=slurm_gpus,
                slurm_cpus_per_task=slurm_cpus_per_task,
                slurm_mem=slurm_mem,
                slurm_time=slurm_time,
                default_gpus=1,
            )
        except Exception as exc:
            with _JOBS_LOCK:
                current = _get_session_jobs_locked(session_key).get(job_id)
                if current is not None:
                    current.status = "failed"
                    current.error = f"Failed to submit slurm job: {type(exc).__name__}: {exc}"
                    current.finished_at = _utc_now_iso()
                    _append_job_log(current, f"[failed] {current.error}")
            raise RuntimeError(f"Unable to submit emb inference slurm job: {type(exc).__name__}: {exc}") from exc

        with _JOBS_LOCK:
            current = _get_session_jobs_locked(session_key).get(job_id)
            if current is not None:
                current.scheduler_job_id = str(submission["scheduler_job_id"])
                current.worker_log_path = str(submission["worker_log_path"])
                current.worker_error_log_path = str(submission["worker_error_log_path"])
                current.progress = {
                    "phase": "submitted",
                    "message": "Submitted to slurm.",
                    "scheduler_job_id": current.scheduler_job_id,
                    "slurm_profile_reason": submission.get("profile_reason"),
                }
                _append_job_log(current, f"[submitted] Slurm job id={current.scheduler_job_id}.")
                _sync_job_state_locked(current)
                poll_interval = _recommend_poll_interval_seconds(current)
            else:
                poll_interval = 5.0

        return {
            "status": "started",
            "job_id": job_id,
            "backend": "slurm",
            "scheduler_job_id": submission["scheduler_job_id"],
            "worker_log_path": submission["worker_log_path"],
            "worker_error_log_path": submission["worker_error_log_path"],
            **resolved,
            "recommended_initial_poll_interval_seconds": poll_interval,
        }

    resolved_cuda_devices, cuda_auto_assigned = _resolve_cuda_devices_for_local(cuda_devices, backend_mode)

    mp_ctx = _get_worker_mp_context()
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    cancel_flag_path = str((Path("/tmp") / f"state_emb_cancel_{job_id}.flag").resolve())
    worker_log_path = str((Path("/tmp") / f"state_emb_worker_{job_id}.log").resolve())
    Path(cancel_flag_path).unlink(missing_ok=True)
    process = mp_ctx.Process(
        target=_run_emb_inference_job_worker,
        args=(resolved, cancel_flag_path, child_conn, worker_log_path, resolved_cuda_devices),
        daemon=False,
        name=f"state_emb_infer_{job_id[:8]}",
    )

    job.event_conn = parent_conn
    job.cancel_flag_path = cancel_flag_path
    job.worker_log_path = worker_log_path
    job.process = process

    try:
        process.start()
        try:
            child_conn.close()
        except Exception:
            pass
    except Exception as exc:
        with _JOBS_LOCK:
            current = _get_session_jobs_locked(session_key).get(job_id)
            if current is not None:
                current.status = "failed"
                current.error = f"Failed to launch worker process: {type(exc).__name__}: {exc}"
                current.finished_at = _utc_now_iso()
                _append_job_log(current, f"[failed] {current.error}")
                _release_job_runtime_resources(current)
        try:
            child_conn.close()
        except Exception:
            pass
        raise RuntimeError(f"Unable to start emb inference worker process: {type(exc).__name__}: {exc}") from exc

    with _JOBS_LOCK:
        current = _get_session_jobs_locked(session_key).get(job_id)
        if current is not None:
            current.worker_pid = process.pid
            _sync_job_state_locked(current)
            poll_interval = _recommend_poll_interval_seconds(current)
        else:
            poll_interval = 5.0

    return {
        "status": "started",
        "job_id": job_id,
        **resolved,
        "worker_pid": process.pid,
        "worker_log_path": worker_log_path,
        "cuda_devices": resolved_cuda_devices,
        "cuda_auto_assigned": cuda_auto_assigned,
        "recommended_initial_poll_interval_seconds": poll_interval,
    }


@mcp.tool()
def get_emb_inference_status(job_id: str) -> dict[str, Any]:
    """
    Return status, progress, and adaptive polling hints for a background emb inference job.
    """
    session_key = _get_current_session_key()
    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs_by_id, expected_kind="emb")
        _sync_job_state_locked(job)
        progress = dict(job.progress)
        seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
        recommended_poll = _recommend_poll_interval_seconds(job)
        recommended_log_poll = _recommend_poll_interval_seconds(job, for_logs=True)
        return {
            "job_id": job.job_id,
            "inference_kind": job.inference_kind,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "last_update_at": job.last_update_at,
            "seconds_since_last_update": round(seconds_since_update, 3),
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "error": job.error,
            "model_folder": job.model_folder,
            "input_adata_path": job.adata_path,
            "input_adata_size_bytes": job.adata_size_bytes,
            "output_path": job.output_path,
            "checkpoint_path": job.checkpoint_path,
            "resolved_args": dict(job.resolved_args),
            "progress": progress,
            "last_event": dict(job.last_event) if isinstance(job.last_event, dict) else None,
            "log_line_count": len(job.logs),
            "worker_pid": job.worker_pid,
            "worker_exit_code": job.worker_exit_code,
            "worker_log_path": job.worker_log_path,
            "worker_error_log_path": job.worker_error_log_path,
            "backend": job.backend,
            "backend_profile": job.backend_profile,
            "scheduler_job_id": job.scheduler_job_id,
            "recommended_poll_interval_seconds": recommended_poll,
            "recommended_log_poll_interval_seconds": recommended_log_poll,
        }


@mcp.tool()
def get_emb_inference_logs(job_id: str, from_line: int = 0, max_lines: int = 200) -> dict[str, Any]:
    """
    Return buffered logs for a background emb inference job plus a polling hint.
    """
    if from_line < 0:
        raise ValueError("`from_line` must be >= 0.")
    if max_lines <= 0:
        raise ValueError("`max_lines` must be > 0.")

    session_key = _get_current_session_key()
    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs_by_id, expected_kind="emb")
        _sync_job_state_locked(job)
        start = min(from_line, len(job.logs))
        end = min(start + max_lines, len(job.logs))
        lines = job.logs[start:end]
        recommended_poll = _recommend_poll_interval_seconds(job, for_logs=True)

    return {
        "job_id": job_id,
        "from_line": start,
        "next_line": end,
        "total_lines": len(job.logs),
        "recommended_poll_interval_seconds": recommended_poll,
        "lines": lines,
    }


@mcp.tool()
def cancel_emb_inference(job_id: str, force: bool = False) -> dict[str, Any]:
    """
    Request cancellation for a background emb inference job.

    A graceful cancellation signal is sent first. If `force=True`, the worker
    process is also sent SIGTERM immediately.
    """
    session_key = _get_current_session_key()
    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs_by_id, expected_kind="emb")
        _sync_job_state_locked(job)
        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "job_id": job_id,
                "status": job.status,
                "cancel_requested": job.cancel_requested,
                "message": "Job is already in a terminal state.",
            }

        now_iso = _utc_now_iso()
        now_mono = time.monotonic()
        first_request = not job.cancel_requested

        job.cancel_requested = True
        if first_request:
            job.cancel_requested_at = now_iso
            job.cancel_grace_deadline_monotonic = now_mono + _CANCEL_GRACE_SECONDS
        if job.status in {"queued", "running"}:
            job.status = "cancelling"

        terminate_sent = False
        scancel_sent = False
        scancel_exit_code: int | None = None
        scancel_message: str | None = None

        if job.backend == "slurm":
            if isinstance(job.scheduler_job_id, str) and job.scheduler_job_id.strip():
                if shutil.which("scancel") is None:
                    scancel_message = "`scancel` not found in PATH."
                    _append_job_log(job, f"[warning] {scancel_message}")
                else:
                    try:
                        result = subprocess.run(
                            ["scancel", job.scheduler_job_id],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        scancel_exit_code = int(result.returncode)
                        out = (result.stdout or "").strip()
                        err = (result.stderr or "").strip()
                        if result.returncode == 0:
                            scancel_sent = True
                            scancel_message = "scancel request submitted."
                            _append_job_log(job, f"[cancelling] Sent scancel for {job.scheduler_job_id}.")
                        else:
                            scancel_message = f"scancel failed with code {result.returncode}. stdout={out!r} stderr={err!r}"
                            _append_job_log(job, f"[warning] {scancel_message}")
                    except Exception as exc:
                        scancel_message = f"Failed to invoke scancel: {type(exc).__name__}: {exc}"
                        _append_job_log(job, f"[warning] {scancel_message}")
            else:
                scancel_message = "No scheduler job id recorded; cannot send scancel."
                _append_job_log(job, f"[warning] {scancel_message}")
        else:
            if isinstance(job.cancel_flag_path, str) and job.cancel_flag_path:
                try:
                    Path(job.cancel_flag_path).touch(exist_ok=True)
                except Exception as exc:
                    _append_job_log(
                        job,
                        f"[warning] Failed to set cancellation flag at {job.cancel_flag_path}: {type(exc).__name__}: {exc}",
                    )

            if force and _process_alive(job.process):
                try:
                    job.process.terminate()
                    terminate_sent = True
                    job.terminate_sent_at_monotonic = now_mono
                    _append_job_log(job, "[cancelling] Force cancellation requested; sent SIGTERM to worker process.")
                except Exception as exc:
                    _append_job_log(job, f"[warning] Failed to force-cancel worker process: {type(exc).__name__}: {exc}")
            elif first_request:
                _append_job_log(
                    job,
                    f"[cancelling] Cancellation requested. Grace period is {_CANCEL_GRACE_SECONDS:.0f}s before force termination.",
                )
            else:
                _append_job_log(job, "[cancelling] Cancellation requested.")

        _sync_job_state_locked(job)
        poll_interval = _recommend_poll_interval_seconds(job)
        return {
            "job_id": job_id,
            "status": job.status,
            "backend": job.backend,
            "scheduler_job_id": job.scheduler_job_id,
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "force": force,
            "terminate_sent": terminate_sent,
            "scancel_sent": scancel_sent,
            "scancel_exit_code": scancel_exit_code,
            "scancel_message": scancel_message,
            "recommended_poll_interval_seconds": poll_interval,
        }


@mcp.tool()
def start_tx_inference(
    adata_path: str,
    output_path: str | None = None,
    model_folder: str | None = None,
    checkpoint_path: str | None = None,
    perturbation_column: str | None = None,
    embedding_key: str | None = None,
    cell_type_column: str | None = None,
    include_cell_types: list[str] | None = None,
    batch_column: str | None = None,
    control_perturbation: str | None = None,
    seed: int = 42,
    max_set_len: int | None = None,
    padding_tsv_path: str | None = None,
    simulate_all_perturbations: bool = False,
    virtual_cells_per_perturbation: int | None = None,
    min_cells_per_perturbation: int | None = None,
    max_cells_per_perturbation: int | None = None,
    batched: bool = True,
    set_batch_size: int | None = None,
    quiet: bool = True,
    backend: str = "auto",
    backend_profile: str | None = None,
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    cuda_devices: str | None = None,
) -> dict[str, Any]:
    """
    Start tx inference in the background and return a `job_id` immediately.
    Poll status with `get_tx_inference_status`, fetch logs with `get_tx_inference_logs`,
    and cancel with `cancel_tx_inference`.

    Returns poll-hint metadata (`recommended_initial_poll_interval_seconds`) for
    clients that want adaptive backoff on long-running jobs.
    """
    session_key = _get_current_session_key()
    with _SESSION_SERVER_STATE_LOCK:
        default_model_folder = _get_session_server_state_locked(session_key)["tx_model_folder"]

    infer_args, resolved = _resolve_tx_inference_request(
        adata_path=adata_path,
        output_path=output_path,
        model_folder=model_folder,
        default_model_folder=default_model_folder,
        checkpoint_path=checkpoint_path,
        perturbation_column=perturbation_column,
        embedding_key=embedding_key,
        cell_type_column=cell_type_column,
        include_cell_types=include_cell_types,
        batch_column=batch_column,
        control_perturbation=control_perturbation,
        seed=seed,
        max_set_len=max_set_len,
        padding_tsv_path=padding_tsv_path,
        simulate_all_perturbations=simulate_all_perturbations,
        virtual_cells_per_perturbation=virtual_cells_per_perturbation,
        min_cells_per_perturbation=min_cells_per_perturbation,
        max_cells_per_perturbation=max_cells_per_perturbation,
        batched=batched,
        set_batch_size=set_batch_size,
        quiet=quiet,
    )

    backend_mode, backend_reason = _resolve_backend_mode(backend)

    job_id = uuid4().hex
    job = InferenceJob(
        job_id=job_id,
        status="queued",
        created_at=_utc_now_iso(),
        model_folder=resolved["model_folder"],
        adata_path=resolved["adata_path"],
        output_path=resolved["output_path"],
        checkpoint_path=resolved["checkpoint_path"],
        inference_kind="tx",
        resolved_args=resolved["resolved_args"],
        adata_size_bytes=resolved.get("adata_size_bytes"),
        backend=backend_mode,
        backend_profile=backend_profile,
    )
    _append_job_log(job, "[queued] Inference job queued.")

    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        jobs_by_id[job_id] = job

    if backend_mode == "slurm":
        command = _build_tx_infer_cli_args(infer_args)
        try:
            submission = _submit_slurm_job(
                job_id=job_id,
                job_name_prefix="state_tx_infer",
                run_dir=None,
                command=command,
                backend_profile=backend_profile,
                slurm_partition=slurm_partition,
                slurm_gpus=slurm_gpus,
                slurm_cpus_per_task=slurm_cpus_per_task,
                slurm_mem=slurm_mem,
                slurm_time=slurm_time,
                default_gpus=1,
            )
        except Exception as exc:
            with _JOBS_LOCK:
                current = _get_session_jobs_locked(session_key).get(job_id)
                if current is not None:
                    current.status = "failed"
                    current.error = f"Failed to submit slurm job: {type(exc).__name__}: {exc}"
                    current.finished_at = _utc_now_iso()
                    _append_job_log(current, f"[failed] {current.error}")
            raise RuntimeError(f"Unable to submit tx inference slurm job: {type(exc).__name__}: {exc}") from exc

        with _JOBS_LOCK:
            current = _get_session_jobs_locked(session_key).get(job_id)
            if current is not None:
                current.scheduler_job_id = str(submission["scheduler_job_id"])
                current.worker_log_path = str(submission["worker_log_path"])
                current.worker_error_log_path = str(submission["worker_error_log_path"])
                current.progress = {
                    "phase": "submitted",
                    "message": "Submitted to slurm.",
                    "scheduler_job_id": current.scheduler_job_id,
                    "slurm_profile_reason": submission.get("profile_reason"),
                }
                _append_job_log(current, f"[submitted] Slurm job id={current.scheduler_job_id}.")
                _sync_job_state_locked(current)
                poll_interval = _recommend_poll_interval_seconds(current)
            else:
                poll_interval = 5.0

        return {
            "status": "started",
            "job_id": job_id,
            "backend": "slurm",
            "scheduler_job_id": submission["scheduler_job_id"],
            "worker_log_path": submission["worker_log_path"],
            "worker_error_log_path": submission["worker_error_log_path"],
            **resolved,
            "recommended_initial_poll_interval_seconds": poll_interval,
        }

    resolved_cuda_devices, cuda_auto_assigned = _resolve_cuda_devices_for_local(cuda_devices, backend_mode)

    mp_ctx = _get_worker_mp_context()
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    cancel_flag_path = str((Path("/tmp") / f"state_tx_cancel_{job_id}.flag").resolve())
    worker_log_path = str((Path("/tmp") / f"state_tx_worker_{job_id}.log").resolve())
    Path(cancel_flag_path).unlink(missing_ok=True)
    process = mp_ctx.Process(
        target=_run_inference_job_worker,
        args=(infer_args, cancel_flag_path, child_conn, worker_log_path, resolved_cuda_devices),
        daemon=False,
        name=f"state_tx_infer_{job_id[:8]}",
    )

    job.event_conn = parent_conn
    job.cancel_flag_path = cancel_flag_path
    job.worker_log_path = worker_log_path
    job.process = process

    try:
        process.start()
        try:
            child_conn.close()
        except Exception:
            pass
    except Exception as exc:
        with _JOBS_LOCK:
            current = _get_session_jobs_locked(session_key).get(job_id)
            if current is not None:
                current.status = "failed"
                current.error = f"Failed to launch worker process: {type(exc).__name__}: {exc}"
                current.finished_at = _utc_now_iso()
                _append_job_log(current, f"[failed] {current.error}")
                _release_job_runtime_resources(current)
        try:
            child_conn.close()
        except Exception:
            pass
        raise RuntimeError(f"Unable to start tx inference worker process: {type(exc).__name__}: {exc}") from exc

    with _JOBS_LOCK:
        current = _get_session_jobs_locked(session_key).get(job_id)
        if current is not None:
            current.worker_pid = process.pid
            _sync_job_state_locked(current)
            poll_interval = _recommend_poll_interval_seconds(current)
        else:
            poll_interval = 5.0

    return {
        "status": "started",
        "job_id": job_id,
        **resolved,
        "worker_pid": process.pid,
        "worker_log_path": worker_log_path,
        "cuda_devices": resolved_cuda_devices,
        "cuda_auto_assigned": cuda_auto_assigned,
        "recommended_initial_poll_interval_seconds": poll_interval,
    }


@mcp.tool()
def get_tx_inference_status(job_id: str) -> dict[str, Any]:
    """
    Return status, progress, and adaptive polling hints for a background tx inference job.
    """
    session_key = _get_current_session_key()
    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs_by_id, expected_kind="tx")
        _sync_job_state_locked(job)
        progress = dict(job.progress)
        seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
        recommended_poll = _recommend_poll_interval_seconds(job)
        recommended_log_poll = _recommend_poll_interval_seconds(job, for_logs=True)
        return {
            "job_id": job.job_id,
            "inference_kind": job.inference_kind,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "last_update_at": job.last_update_at,
            "seconds_since_last_update": round(seconds_since_update, 3),
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "error": job.error,
            "model_folder": job.model_folder,
            "adata_path": job.adata_path,
            "adata_size_bytes": job.adata_size_bytes,
            "output_path": job.output_path,
            "checkpoint_path": job.checkpoint_path,
            "resolved_args": dict(job.resolved_args),
            "progress": progress,
            "last_event": dict(job.last_event) if isinstance(job.last_event, dict) else None,
            "log_line_count": len(job.logs),
            "worker_pid": job.worker_pid,
            "worker_exit_code": job.worker_exit_code,
            "worker_log_path": job.worker_log_path,
            "worker_error_log_path": job.worker_error_log_path,
            "backend": job.backend,
            "backend_profile": job.backend_profile,
            "scheduler_job_id": job.scheduler_job_id,
            "recommended_poll_interval_seconds": recommended_poll,
            "recommended_log_poll_interval_seconds": recommended_log_poll,
        }


@mcp.tool()
def get_tx_inference_logs(job_id: str, from_line: int = 0, max_lines: int = 200) -> dict[str, Any]:
    """
    Return buffered logs for a background tx inference job plus a polling hint.
    """
    if from_line < 0:
        raise ValueError("`from_line` must be >= 0.")
    if max_lines <= 0:
        raise ValueError("`max_lines` must be > 0.")

    session_key = _get_current_session_key()
    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs_by_id, expected_kind="tx")
        _sync_job_state_locked(job)
        start = min(from_line, len(job.logs))
        end = min(start + max_lines, len(job.logs))
        lines = job.logs[start:end]
        recommended_poll = _recommend_poll_interval_seconds(job, for_logs=True)

    return {
        "job_id": job_id,
        "from_line": start,
        "next_line": end,
        "total_lines": len(job.logs),
        "recommended_poll_interval_seconds": recommended_poll,
        "lines": lines,
    }


@mcp.tool()
def cancel_tx_inference(job_id: str, force: bool = False) -> dict[str, Any]:
    """
    Request cancellation for a background tx inference job.

    A graceful cancellation signal is sent first. If `force=True`, the worker
    process is also sent SIGTERM immediately.
    """
    session_key = _get_current_session_key()
    with _JOBS_LOCK:
        jobs_by_id = _get_session_jobs_locked(session_key)
        job = _get_job_locked(job_id, jobs_by_id, expected_kind="tx")
        _sync_job_state_locked(job)
        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "job_id": job_id,
                "status": job.status,
                "cancel_requested": job.cancel_requested,
                "message": "Job is already in a terminal state.",
            }

        now_iso = _utc_now_iso()
        now_mono = time.monotonic()
        first_request = not job.cancel_requested

        job.cancel_requested = True
        if first_request:
            job.cancel_requested_at = now_iso
            job.cancel_grace_deadline_monotonic = now_mono + _CANCEL_GRACE_SECONDS
        if job.status in {"queued", "running"}:
            job.status = "cancelling"

        terminate_sent = False
        scancel_sent = False
        scancel_exit_code: int | None = None
        scancel_message: str | None = None

        if job.backend == "slurm":
            if isinstance(job.scheduler_job_id, str) and job.scheduler_job_id.strip():
                if shutil.which("scancel") is None:
                    scancel_message = "`scancel` not found in PATH."
                    _append_job_log(job, f"[warning] {scancel_message}")
                else:
                    try:
                        result = subprocess.run(
                            ["scancel", job.scheduler_job_id],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        scancel_exit_code = int(result.returncode)
                        out = (result.stdout or "").strip()
                        err = (result.stderr or "").strip()
                        if result.returncode == 0:
                            scancel_sent = True
                            scancel_message = "scancel request submitted."
                            _append_job_log(job, f"[cancelling] Sent scancel for {job.scheduler_job_id}.")
                        else:
                            scancel_message = f"scancel failed with code {result.returncode}. stdout={out!r} stderr={err!r}"
                            _append_job_log(job, f"[warning] {scancel_message}")
                    except Exception as exc:
                        scancel_message = f"Failed to invoke scancel: {type(exc).__name__}: {exc}"
                        _append_job_log(job, f"[warning] {scancel_message}")
            else:
                scancel_message = "No scheduler job id recorded; cannot send scancel."
                _append_job_log(job, f"[warning] {scancel_message}")
        else:
            if isinstance(job.cancel_flag_path, str) and job.cancel_flag_path:
                try:
                    Path(job.cancel_flag_path).touch(exist_ok=True)
                except Exception as exc:
                    _append_job_log(
                        job,
                        f"[warning] Failed to set cancellation flag at {job.cancel_flag_path}: {type(exc).__name__}: {exc}",
                    )

            if force and _process_alive(job.process):
                try:
                    job.process.terminate()
                    terminate_sent = True
                    job.terminate_sent_at_monotonic = now_mono
                    _append_job_log(job, "[cancelling] Force cancellation requested; sent SIGTERM to worker process.")
                except Exception as exc:
                    _append_job_log(job, f"[warning] Failed to force-cancel worker process: {type(exc).__name__}: {exc}")
            elif first_request:
                _append_job_log(
                    job,
                    f"[cancelling] Cancellation requested. Grace period is {_CANCEL_GRACE_SECONDS:.0f}s before force termination.",
                )
            else:
                _append_job_log(job, "[cancelling] Cancellation requested.")

        _sync_job_state_locked(job)
        poll_interval = _recommend_poll_interval_seconds(job)

        return {
            "job_id": job_id,
            "status": job.status,
            "backend": job.backend,
            "scheduler_job_id": job.scheduler_job_id,
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "force": force,
            "terminate_sent": terminate_sent,
            "scancel_sent": scancel_sent,
            "scancel_exit_code": scancel_exit_code,
            "scancel_message": scancel_message,
            "recommended_poll_interval_seconds": poll_interval,
        }


# ---------------------------------------------------------------------------
# Preprocessing tools
# ---------------------------------------------------------------------------

def _resolve_preprocess_request(
    *,
    input_paths: list[str] | None,
    input_pattern: str | None,
    exclude_patterns: list[str] | None,
    output_dir: str,
    target_sum: float | None,
    already_log1p: bool,
    perturbation_col: str,
    control_perturbation: str,
    context_col: str | None,
    batch_col: str | None,
    sort_by: list[str] | None,
    gene_set: str | None,
    add_pert_efficiency: bool,
    efficiency_key: str,
    target_fc_key: str,
    eps: float,
    downsample_frac: float,
    num_hvgs: int | None,
    seed: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Validate inputs and return a serializable config dict."""
    resolved_output_dir = str(Path(output_dir).expanduser().resolve())
    resolved_input_paths: list[str] = []
    if input_paths:
        for p in input_paths:
            rp = str(Path(p).expanduser().resolve())
            if not Path(rp).is_file():
                raise FileNotFoundError(f"Input file not found: {rp}")
            resolved_input_paths.append(rp)

    if not resolved_input_paths and not input_pattern:
        raise ValueError("Either `input_paths` or `input_pattern` must be provided.")

    resolved_gene_set: str | None = None
    if gene_set is not None:
        resolved_gene_set = str(Path(gene_set).expanduser().resolve())
        if not Path(resolved_gene_set).is_file():
            raise FileNotFoundError(f"Gene set file not found: {resolved_gene_set}")

    if downsample_frac <= 0 or downsample_frac > 1.0:
        raise ValueError("`downsample_frac` must be in (0, 1].")
    if num_hvgs is not None and num_hvgs <= 0:
        raise ValueError("`num_hvgs` must be > 0 when provided.")
    if target_sum is not None and target_sum <= 0:
        raise ValueError("`target_sum` must be > 0 when provided.")

    return {
        "input_paths": resolved_input_paths,
        "input_pattern": input_pattern,
        "exclude_patterns": exclude_patterns or [],
        "output_dir": resolved_output_dir,
        "target_sum": target_sum,
        "already_log1p": already_log1p,
        "perturbation_col": perturbation_col,
        "control_perturbation": control_perturbation,
        "context_col": context_col,
        "batch_col": batch_col,
        "sort_by": sort_by or [],
        "gene_set": resolved_gene_set,
        "add_pert_efficiency": add_pert_efficiency,
        "efficiency_key": efficiency_key,
        "target_fc_key": target_fc_key,
        "eps": eps,
        "downsample_frac": downsample_frac,
        "num_hvgs": num_hvgs,
        "seed": seed,
        "overwrite": overwrite,
        "dry_run": dry_run,
    }


@mcp.tool()
def start_preprocess_train(
    output_dir: str,
    input_paths: list[str] | None = None,
    input_pattern: str | None = None,
    exclude_patterns: list[str] | None = None,
    target_sum: float | None = None,
    already_log1p: bool = False,
    perturbation_col: str = "target_gene",
    control_perturbation: str = "non-targeting",
    context_col: str | None = None,
    batch_col: str | None = None,
    sort_by: list[str] | None = None,
    gene_set: str | None = None,
    add_pert_efficiency: bool = True,
    efficiency_key: str = "KnockDownEfficiency",
    target_fc_key: str = "KnockDownGeneFC",
    eps: float = 1e-8,
    downsample_frac: float = 1.0,
    num_hvgs: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
    dry_run: bool = False,
    backend: str = "local",
    backend_profile: str | None = None,
    slurm_partition: str | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
) -> dict[str, Any]:
    """
    Start TX training data preprocessing in the background and return a `job_id` immediately.

    Full pert-transform pipeline per file:
    1. Optional gene alignment to gene_set (.npy)
    2. Standardize perturbation column names (renamed to 'perturbation', control to 'control')
    3. Apply context/batch column mapping
    4. Sort cells by specified obs columns (for consecutive-loading compatibility)
    5. Optional expm1 (undo log1p if already_log1p=True)
    6. Optional binomial downsampling
    7. Normalize total counts (sc.pp.normalize_total)
    8. Compute knockdown efficiency (before log1p)
    9. Apply log1p transformation
    10. Compute log fold change (after log1p)
    11. Optional HVG selection (num_hvgs) — stores HVG expression in obsm['X_hvg']
    12. Write output file

    **sort_by**: list of obs columns to sort cells by. Sorting produces contiguous
    groups required by `use_consecutive_loading=True` in TX training. Typical
    value: `["context", "perturbation"]` (or the raw column names, e.g.
    `["cell_type", "perturbation"]`).

    **num_hvgs**: when set, computes a **global** HVG set across all input files
    using pseudobulk aggregation (via adpbulk) followed by seurat_v3 HVG selection.
    This ensures all output files share the same HVG columns. The HVG expression
    sub-matrix is stored in `obsm['X_hvg']`. Useful for `output_space='gene'`
    (legacy: `output_space='hvg'`) training.

    Poll status with `get_preprocess_train_status`, fetch logs with `get_preprocess_train_logs`,
    and cancel with `cancel_preprocess_train`.
    """
    config_dict = _resolve_preprocess_request(
        input_paths=input_paths,
        input_pattern=input_pattern,
        exclude_patterns=exclude_patterns,
        output_dir=output_dir,
        target_sum=target_sum,
        already_log1p=already_log1p,
        perturbation_col=perturbation_col,
        control_perturbation=control_perturbation,
        context_col=context_col,
        batch_col=batch_col,
        sort_by=sort_by,
        gene_set=gene_set,
        add_pert_efficiency=add_pert_efficiency,
        efficiency_key=efficiency_key,
        target_fc_key=target_fc_key,
        eps=eps,
        downsample_frac=downsample_frac,
        num_hvgs=num_hvgs,
        seed=seed,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    session_key = _get_current_session_key()
    backend_mode, backend_reason = _resolve_backend_mode(backend)

    job_id = uuid4().hex
    job = PreprocessJob(
        job_id=job_id,
        status="queued",
        created_at=_utc_now_iso(),
        output_dir=config_dict["output_dir"],
        resolved_config=config_dict,
        backend=backend_mode,
        backend_profile=backend_profile,
    )
    _append_preprocess_job_log(job, "[queued] Preprocess job queued.")

    with _PREPROCESS_JOBS_LOCK:
        preprocess_jobs = _get_session_preprocess_jobs_locked(session_key)
        preprocess_jobs[job_id] = job

    # Local backend (no GPU needed for preprocessing)
    mp_ctx = _get_worker_mp_context()
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    cancel_flag_path = str((Path("/tmp") / f"state_preprocess_cancel_{job_id}.flag").resolve())
    worker_log_path = str((Path("/tmp") / f"state_preprocess_worker_{job_id}.log").resolve())
    Path(cancel_flag_path).unlink(missing_ok=True)
    process = mp_ctx.Process(
        target=_run_preprocess_job_worker,
        args=(config_dict, cancel_flag_path, child_conn, worker_log_path),
        daemon=False,
        name=f"state_preprocess_{job_id[:8]}",
    )

    job.event_conn = parent_conn
    job.cancel_flag_path = cancel_flag_path
    job.worker_log_path = worker_log_path
    job.process = process

    try:
        process.start()
        try:
            child_conn.close()
        except Exception:
            pass
    except Exception as exc:
        with _PREPROCESS_JOBS_LOCK:
            current = _get_session_preprocess_jobs_locked(session_key).get(job_id)
            if current is not None:
                current.status = "failed"
                current.error = f"Failed to launch worker process: {type(exc).__name__}: {exc}"
                current.finished_at = _utc_now_iso()
                _append_preprocess_job_log(current, f"[failed] {current.error}")
                _release_preprocess_job_runtime_resources(current)
        try:
            child_conn.close()
        except Exception:
            pass
        raise RuntimeError(
            f"Unable to start preprocess worker process: {type(exc).__name__}: {exc}"
        ) from exc

    with _PREPROCESS_JOBS_LOCK:
        current = _get_session_preprocess_jobs_locked(session_key).get(job_id)
        if current is not None:
            current.worker_pid = process.pid
            _sync_preprocess_job_state_locked(current)
            poll_interval = _recommend_preprocess_poll_interval_seconds(current)
        else:
            poll_interval = 5.0

    return {
        "status": "started",
        "job_id": job_id,
        "backend": backend_mode,
        "output_dir": config_dict["output_dir"],
        "resolved_config": config_dict,
        "worker_pid": process.pid,
        "worker_log_path": worker_log_path,
        "recommended_initial_poll_interval_seconds": poll_interval,
    }


@mcp.tool()
def get_preprocess_train_status(job_id: str) -> dict[str, Any]:
    """
    Return status, progress, and adaptive polling hints for a background preprocess job.
    """
    session_key = _get_current_session_key()
    with _PREPROCESS_JOBS_LOCK:
        jobs_by_id = _get_session_preprocess_jobs_locked(session_key)
        job = _get_preprocess_job_locked(job_id, jobs_by_id)
        _sync_preprocess_job_state_locked(job)
        progress = dict(job.progress)
        seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
        recommended_poll = _recommend_preprocess_poll_interval_seconds(job)
        recommended_log_poll = _recommend_preprocess_poll_interval_seconds(job, for_logs=True)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "backend": job.backend,
            "backend_profile": job.backend_profile,
            "scheduler_job_id": job.scheduler_job_id,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "last_update_at": job.last_update_at,
            "seconds_since_last_update": round(seconds_since_update, 0),
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "error": job.error,
            "output_dir": job.output_dir,
            "resolved_config": job.resolved_config,
            "progress": progress,
            "last_event": job.last_event,
            "log_line_count": len(job.logs),
            "worker_pid": job.worker_pid,
            "worker_exit_code": job.worker_exit_code,
            "worker_log_path": job.worker_log_path,
            "recommended_poll_interval_seconds": recommended_poll,
            "recommended_log_poll_interval_seconds": recommended_log_poll,
        }


@mcp.tool()
def get_preprocess_train_logs(job_id: str, from_line: int = 0, max_lines: int = 200) -> dict[str, Any]:
    """
    Return buffered logs for a background preprocess job plus a polling hint.
    """
    session_key = _get_current_session_key()
    with _PREPROCESS_JOBS_LOCK:
        jobs_by_id = _get_session_preprocess_jobs_locked(session_key)
        job = _get_preprocess_job_locked(job_id, jobs_by_id)
        _sync_preprocess_job_state_locked(job)

        total_lines = len(job.logs)
        start = max(0, min(from_line, total_lines))
        end = min(start + max_lines, total_lines)
        lines = job.logs[start:end]
        recommended_poll = _recommend_preprocess_poll_interval_seconds(job, for_logs=True)

        return {
            "job_id": job_id,
            "from_line": start,
            "next_line": end,
            "total_lines": total_lines,
            "recommended_poll_interval_seconds": recommended_poll,
            "lines": lines,
        }


@mcp.tool()
def cancel_preprocess_train(job_id: str, force: bool = False) -> dict[str, Any]:
    """
    Request cancellation for a background preprocess job.

    A graceful cancellation signal is sent first. If `force=True`, the worker
    process is also sent SIGTERM immediately.
    """
    session_key = _get_current_session_key()
    terminate_sent = False

    with _PREPROCESS_JOBS_LOCK:
        jobs_by_id = _get_session_preprocess_jobs_locked(session_key)
        job = _get_preprocess_job_locked(job_id, jobs_by_id)
        _sync_preprocess_job_state_locked(job)

        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "job_id": job_id,
                "status": job.status,
                "cancel_requested": job.cancel_requested,
                "message": f"Job already in terminal state: {job.status}.",
            }

        if not job.cancel_requested:
            job.cancel_requested = True
            job.cancel_requested_at = _utc_now_iso()
            job.cancel_grace_deadline_monotonic = time.monotonic() + _CANCEL_GRACE_SECONDS

            cancel_flag_path = job.cancel_flag_path
            if isinstance(cancel_flag_path, str) and cancel_flag_path:
                try:
                    Path(cancel_flag_path).touch()
                except Exception:
                    pass

            if job.status == "running":
                job.status = "cancelling"
            _append_preprocess_job_log(job, "[cancelling] Cancellation requested.")

        if force and _process_alive(job.process):
            try:
                job.process.terminate()  # type: ignore[union-attr]
                job.terminate_sent_at_monotonic = time.monotonic()
                terminate_sent = True
                _append_preprocess_job_log(job, "[cancelling] Force-cancel: sent SIGTERM.")
            except Exception as exc:
                _append_preprocess_job_log(
                    job,
                    f"[warning] Failed to terminate worker: {type(exc).__name__}: {exc}",
                )

        _sync_preprocess_job_state_locked(job)
        poll_interval = _recommend_preprocess_poll_interval_seconds(job)

        return {
            "job_id": job_id,
            "status": job.status,
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "force": force,
            "terminate_sent": terminate_sent,
            "recommended_poll_interval_seconds": poll_interval,
        }


def main() -> None:
    """
    Run the STATE MCP server.

    Default transport is stdio for compatibility with Codex-managed process launch.
    Use `--transport streamable-http` to host as a standalone localhost service.
    """
    parser = ArgumentParser(description="Run the STATE MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=os.getenv("STATE_MCP_TRANSPORT", "stdio"),
        help="MCP transport to run (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("STATE_MCP_HOST"),
        help="Host interface for HTTP transports (default: FastMCP default).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("STATE_MCP_PORT"),
        help="Port for HTTP transports (default: FastMCP default).",
    )
    parser.add_argument(
        "--mount-path",
        default=os.getenv("STATE_MCP_MOUNT_PATH"),
        help="Optional mount path for SSE transport.",
    )
    parser.add_argument(
        "--streamable-http-path",
        default=os.getenv("STATE_MCP_STREAMABLE_HTTP_PATH"),
        help="Path for streamable-http endpoint (default: /mcp).",
    )
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        help="Enable stateless HTTP session mode for streamable-http transport.",
    )

    args = parser.parse_args()

    if args.host:
        mcp.settings.host = args.host
    if args.port is not None:
        if args.port <= 0:
            raise ValueError(f"--port must be a positive integer, got {args.port}")
        mcp.settings.port = args.port
    if args.streamable_http_path:
        mcp.settings.streamable_http_path = args.streamable_http_path
    if args.stateless_http:
        mcp.settings.stateless_http = True

    mount_path = args.mount_path if args.mount_path else None
    mcp.run(transport=args.transport, mount_path=mount_path)


if __name__ == "__main__":
    main()
