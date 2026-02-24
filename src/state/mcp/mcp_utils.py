from __future__ import annotations

import glob
import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

_REQUIRED_RUN_FILES: Final[tuple[str, ...]] = (
    "config.yaml",
    "var_dims.pkl",
    "pert_onehot_map.pt",
)

_REQUIRED_FILE_VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    "data_module": ("data_module.torch", "data_module.pt", "data_module.pkl"),
    "batch_onehot_map": ("batch_onehot_map.torch", "batch_onehot_map.pt", "batch_onehot_map.pkl"),
    "cell_type_onehot_map": ("cell_type_onehot_map.torch", "cell_type_onehot_map.pt", "cell_type_onehot_map.pkl"),
}

_CHECKPOINT_PRIORITY: Final[tuple[str, ...]] = ("final.ckpt", "best.ckpt", "last.ckpt")
_EMB_CHECKPOINT_SUFFIXES: Final[tuple[str, ...]] = (".ckpt", ".safetensors")
_EMB_CHECKPOINT_BASENAME_PRIORITY: Final[tuple[str, ...]] = (
    "model.safetensors",
    "checkpoint.safetensors",
    "last.ckpt",
    "best.ckpt",
    "final.ckpt",
    "checkpoint.ckpt",
)


def normalize_and_validate_run_dir(model_folder: str) -> str:
    """
    Normalize and validate a STATE training run directory path.

    Required artifacts:
    - config.yaml
    - data_module.{torch,pt,pkl}
    - pert_onehot_map.pt
    - var_dims.pkl
    - batch_onehot_map.{torch,pt,pkl}
    - cell_type_onehot_map.{torch,pt,pkl}
    - checkpoints/ containing at least one file
    """
    if not model_folder or not model_folder.strip():
        raise ValueError("`model_folder` must be a non-empty path.")

    path = Path(model_folder).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model folder does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Model folder is not a directory: {path}")

    missing: list[str] = []

    for filename in _REQUIRED_RUN_FILES:
        if not (path / filename).is_file():
            missing.append(filename)

    for base_name, candidates in _REQUIRED_FILE_VARIANTS.items():
        if not any((path / candidate).is_file() for candidate in candidates):
            missing.append(f"{base_name} ({' or '.join(candidates)})")

    checkpoints_dir = path / "checkpoints"
    if not checkpoints_dir.exists():
        missing.append("checkpoints/")
    elif not checkpoints_dir.is_dir():
        missing.append("checkpoints/ (must be a directory)")
    elif not any(entry.is_file() for entry in checkpoints_dir.iterdir()):
        missing.append("checkpoints/ (must contain at least one file)")

    if missing:
        raise ValueError(
            "Path does not look like a STATE run directory. Missing or invalid entries: " + ", ".join(missing)
        )

    return str(path)


def resolve_and_validate_model_folder(model_folder: str | None, default_model_folder: str | None) -> str:
    """
    Resolve an explicit model folder or fall back to the session default.
    """
    candidate = model_folder or default_model_folder
    if candidate is None:
        raise ValueError("No model folder provided and no default model folder is set.")
    return normalize_and_validate_run_dir(candidate)


def resolve_tx_checkpoint_path(
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
    default_model_folder: str | None = None,
) -> str:
    """
    Resolve a TX checkpoint path from either a file path or a validated TX run directory.

    If `checkpoint_path` is provided, it takes precedence.
    """
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"TX checkpoint file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"TX checkpoint path is not a file: {resolved}")
        return str(resolved)

    resolved_model_folder = resolve_and_validate_model_folder(model_folder, default_model_folder)
    checkpoint_dir = Path(resolved_model_folder) / "checkpoints"
    checkpoint_files = _list_checkpoint_files(checkpoint_dir)
    chosen = _choose_checkpoint(checkpoint_dir, checkpoint_files)
    if chosen is None:
        raise FileNotFoundError(
            f"No checkpoint file found in {checkpoint_dir}. "
            "Add checkpoint artifacts or pass `checkpoint_path` explicitly."
        )
    return chosen


def normalize_and_validate_emb_inference_dir(model_folder: str) -> str:
    """
    Normalize and validate a STATE embedding inference directory path.

    Required:
    - at least one checkpoint artifact matching `*.ckpt` or `*.safetensors`

    Optional:
    - config.yaml
    - protein_embeddings.pt
    """
    if not model_folder or not model_folder.strip():
        raise ValueError("`model_folder` must be a non-empty path.")

    path = Path(model_folder).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Embedding model folder does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Embedding model folder is not a directory: {path}")

    checkpoints = _list_emb_checkpoint_candidates(path)
    if not checkpoints:
        expected = ", ".join(_EMB_CHECKPOINT_SUFFIXES)
        raise ValueError(
            "Path does not look like a STATE embedding checkpoint directory. "
            f"Expected at least one checkpoint file with suffix: {expected}. "
            "Optional files: config.yaml, protein_embeddings.pt."
        )

    return str(path)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {config_path}, got {type(data).__name__}")
    return data


def _load_pickle_mapping(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_mapping(path: Path) -> Any:
    if path.suffix == ".pkl":
        return _load_pickle_mapping(path)

    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _pick_existing_file(folder: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _list_checkpoint_files(checkpoint_dir: Path) -> list[str]:
    return sorted(entry.name for entry in checkpoint_dir.iterdir() if entry.is_file())


def _choose_checkpoint(checkpoint_dir: Path, files: list[str]) -> str | None:
    if not files:
        return None
    files_set = set(files)
    for filename in _CHECKPOINT_PRIORITY:
        if filename in files_set:
            return str(checkpoint_dir / filename)
    ckpt_files = sorted(name for name in files if name.endswith(".ckpt"))
    if ckpt_files:
        return str(checkpoint_dir / ckpt_files[0])
    return str(checkpoint_dir / files[0])


def _load_torch_checkpoint(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _list_emb_checkpoint_candidates(folder: Path) -> list[Path]:
    candidates: list[Path] = []
    with os.scandir(folder) as scan_it:
        for entry in scan_it:
            try:
                is_file = entry.is_file()
            except OSError:
                continue
            if not is_file:
                continue
            suffix = Path(entry.name).suffix
            if suffix in _EMB_CHECKPOINT_SUFFIXES:
                candidates.append(folder / entry.name)
    return sorted(candidates, key=lambda p: p.name)


def _choose_emb_checkpoint_from_folder(folder: Path, *, prefer_ckpt: bool = False) -> Path:
    candidates = _list_emb_checkpoint_candidates(folder)
    if not candidates:
        expected = ", ".join(_EMB_CHECKPOINT_SUFFIXES)
        raise FileNotFoundError(
            f"No embedding checkpoint file found in {folder}. "
            f"Expected file suffixes: {expected}"
        )

    if prefer_ckpt:
        ckpt_candidates = [path for path in candidates if path.suffix == ".ckpt"]
        if ckpt_candidates:
            candidates = ckpt_candidates

    candidate_map = {path.name: path for path in candidates}
    for basename in _EMB_CHECKPOINT_BASENAME_PRIORITY:
        picked = candidate_map.get(basename)
        if picked is not None:
            return picked

    # Match CLI behavior by defaulting to lexicographically last candidate.
    return candidates[-1]


def resolve_emb_checkpoint_path(
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
    *,
    prefer_ckpt: bool = False,
) -> str:
    """
    Resolve an embedding checkpoint path from either a file path or model folder.

    If both are provided, `checkpoint_path` takes precedence.
    """
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Embedding checkpoint file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Embedding checkpoint path is not a file: {resolved}")
        if resolved.suffix not in _EMB_CHECKPOINT_SUFFIXES:
            expected = ", ".join(_EMB_CHECKPOINT_SUFFIXES)
            raise ValueError(
                f"Unsupported embedding checkpoint extension {resolved.suffix!r}. "
                f"Expected one of: {expected}"
            )
        return str(resolved)

    if model_folder is None or not model_folder.strip():
        raise ValueError("Provide either `checkpoint_path` or `model_folder` for embedding checkpoint resolution.")

    resolved_folder = Path(normalize_and_validate_emb_inference_dir(model_folder))
    return str(_choose_emb_checkpoint_from_folder(resolved_folder, prefer_ckpt=prefer_ckpt))


def _extract_cfg_from_checkpoint_payload(ckpt_payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(ckpt_payload, dict):
        return None, None

    cfg_yaml: str | None = None
    cfg_source: str | None = None
    if isinstance(ckpt_payload.get("cfg_yaml"), str):
        cfg_yaml = ckpt_payload["cfg_yaml"]
        cfg_source = "cfg_yaml"
    else:
        hp = ckpt_payload.get("hyper_parameters")
        if isinstance(hp, dict) and isinstance(hp.get("cfg_yaml"), str):
            cfg_yaml = hp["cfg_yaml"]
            cfg_source = "hyper_parameters.cfg_yaml"

    if cfg_yaml is None:
        return None, None

    try:
        parsed = yaml.safe_load(cfg_yaml) or {}
    except Exception:
        return None, cfg_source
    if not isinstance(parsed, dict):
        return None, cfg_source
    return parsed, cfg_source


def _infer_checkpoint_kind_from_state_dict(ckpt_payload: Any) -> str:
    if not isinstance(ckpt_payload, dict):
        return "unknown"

    state_dict = ckpt_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        return "unknown"

    keys = {str(k) for k in state_dict.keys()}
    emb_markers = (
        "pe_embedding.weight",
        "gene_embedding_layer.0.weight",
        "transformer_encoder.layers.0.self_attn.in_proj_weight",
    )
    tx_markers = (
        "pert_encoder.0.weight",
        "project_out.0.weight",
        "basal_encoder.0.weight",
    )
    emb_hits = sum(1 for marker in emb_markers if marker in keys)
    tx_hits = sum(1 for marker in tx_markers if marker in keys)

    if emb_hits >= 2 and tx_hits == 0:
        return "state-embedding"
    if tx_hits >= 2 and emb_hits == 0:
        return "state-transition"
    if emb_hits > tx_hits:
        return "state-embedding"
    if tx_hits > emb_hits:
        return "state-transition"
    return "unknown"


def inspect_emb_checkpoint(checkpoint_path: str | None = None, model_folder: str | None = None) -> dict[str, Any]:
    """
    Inspect a STATE embedding checkpoint (single file) and return EMB-specific metadata.

    If both `checkpoint_path` and `model_folder` are provided, `checkpoint_path` takes precedence.
    """
    resolved_checkpoint = Path(
        resolve_emb_checkpoint_path(
            checkpoint_path=checkpoint_path,
            model_folder=model_folder,
            prefer_ckpt=False,
        )
    )
    warnings: list[str] = []

    model_folder_resolved: str | None = None
    if checkpoint_path is None and model_folder is not None and model_folder.strip():
        model_folder_resolved = normalize_and_validate_emb_inference_dir(model_folder)
    elif resolved_checkpoint.parent.exists():
        model_folder_resolved = str(resolved_checkpoint.parent)

    folder_candidates: list[str] = []
    if model_folder_resolved is not None:
        folder_candidates = [str(path) for path in _list_emb_checkpoint_candidates(Path(model_folder_resolved))]

    payload: Any = None
    cfg_data: dict[str, Any] | None = None
    cfg_source: str | None = None
    hyper_parameters: dict[str, Any] = {}
    state_dict: dict[str, Any] = {}
    num_tensors: int | None = None
    total_param_count: int | None = None
    has_cfg_yaml = False
    has_packaged_protein_embeddings = False
    packaged_protein_embedding_count: int | None = None
    inferred_kind = "unknown"
    safetensors_metadata: dict[str, Any] | None = None

    if resolved_checkpoint.suffix == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(resolved_checkpoint), framework="pt", device="cpu") as sf:
                keys = list(sf.keys())
                metadata = sf.metadata() or {}
            num_tensors = len(keys)
            state_dict = {k: None for k in keys[:200]}
            inferred_kind = "state-embedding"
            safetensors_metadata = metadata
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect safetensors checkpoint at {resolved_checkpoint}: {exc}") from exc

        cfg_candidate = resolved_checkpoint.parent / "config.yaml"
        if cfg_candidate.is_file():
            try:
                cfg_data = _load_yaml_config(cfg_candidate)
                cfg_source = "config.yaml"
                has_cfg_yaml = True
            except Exception as exc:
                warnings.append(f"Failed to parse config.yaml near safetensors checkpoint: {exc}")
        else:
            warnings.append(
                "No config.yaml found next to safetensors checkpoint. "
                "This is optional for inspection but required for safetensors-only inference."
            )
    else:
        try:
            payload = _load_torch_checkpoint(resolved_checkpoint)
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding checkpoint at {resolved_checkpoint}: {exc}") from exc

        if not isinstance(payload, dict):
            warnings.append(
                f"Checkpoint payload loaded as {type(payload).__name__}, expected dict-like Lightning checkpoint."
            )

        cfg_data, cfg_source = _extract_cfg_from_checkpoint_payload(payload)
        hyper_parameters = payload.get("hyper_parameters", {}) if isinstance(payload, dict) else {}
        if not isinstance(hyper_parameters, dict):
            hyper_parameters = {}
        state_dict = payload.get("state_dict", {}) if isinstance(payload, dict) else {}
        if not isinstance(state_dict, dict):
            state_dict = {}

        num_tensors = len(state_dict)
        total_param_count = 0
        for tensor in state_dict.values():
            numel_fn = getattr(tensor, "numel", None)
            if callable(numel_fn):
                try:
                    total_param_count += int(numel_fn())
                except Exception:
                    continue

        has_cfg_yaml = isinstance(payload, dict) and isinstance(payload.get("cfg_yaml"), str)
        has_packaged_protein_embeddings = isinstance(payload, dict) and isinstance(payload.get("protein_embeds_dict"), dict)
        if has_packaged_protein_embeddings:
            try:
                packaged_protein_embedding_count = len(payload["protein_embeds_dict"])
            except Exception:
                packaged_protein_embedding_count = None

        inferred_kind = _infer_checkpoint_kind_from_state_dict(payload)
        if inferred_kind != "state-embedding":
            warnings.append(
                f"Checkpoint heuristics inferred kind={inferred_kind!r}; this may not be a STATE embedding checkpoint."
            )

    model_cfg = cfg_data.get("model", {}) if isinstance(cfg_data, dict) else {}
    experiment_cfg = cfg_data.get("experiment", {}) if isinstance(cfg_data, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = {}

    return {
        "checkpoint_type": "emb",
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_filename": resolved_checkpoint.name,
        "checkpoint_extension": resolved_checkpoint.suffix,
        "checkpoint_size_bytes": int(os.path.getsize(resolved_checkpoint)),
        "model_folder": model_folder_resolved,
        "model_folder_checkpoint_candidates": folder_candidates,
        "inferred_checkpoint_kind": inferred_kind,
        "checkpoint_payload_summary": {
            "has_cfg_yaml": has_cfg_yaml,
            "cfg_source": cfg_source,
            "has_hyper_parameters": isinstance(payload, dict) and ("hyper_parameters" in payload),
            "has_state_dict": bool(num_tensors is not None),
            "state_dict_tensor_count": num_tensors,
            "state_dict_total_parameters": total_param_count,
            "state_dict_sample_keys": sorted([str(k) for k in state_dict.keys()])[:20],
            "has_packaged_protein_embeddings": has_packaged_protein_embeddings,
            "packaged_protein_embedding_count": packaged_protein_embedding_count,
            "safetensors_metadata": safetensors_metadata,
        },
        "embedding_config_summary": {
            "experiment_name": _as_optional_str(experiment_cfg.get("name")),
            "checkpoint_path_cfg": _as_optional_str(
                experiment_cfg.get("checkpoint", {}).get("path")
                if isinstance(experiment_cfg.get("checkpoint"), dict)
                else None
            ),
            "model_n_genes": model_cfg.get("n_genes"),
            "model_n_hidden": model_cfg.get("n_hidden"),
            "model_n_latent": model_cfg.get("n_latent"),
            "model_batch_size": model_cfg.get("batch_size"),
            "model_dataset_correction": model_cfg.get("dataset_correction"),
            "optimizer_name": _as_optional_str(cfg_data.get("optimizer", {}).get("name")) if isinstance(cfg_data, dict) else None,
            "hyper_parameter_keys": sorted(hyper_parameters.keys())[:50],
        },
        "warnings": warnings,
    }


def inspect_tx_checkpoint(
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
    default_model_folder: str | None = None,
) -> dict[str, Any]:
    """
    Inspect a STATE TX checkpoint and return TX-focused metadata.
    """
    resolved_checkpoint = Path(
        resolve_tx_checkpoint_path(
            checkpoint_path=checkpoint_path,
            model_folder=model_folder,
            default_model_folder=default_model_folder,
        )
    )
    warnings: list[str] = []

    resolved_model_folder: str | None = None
    candidate_model_folder = model_folder or default_model_folder
    if candidate_model_folder is not None:
        try:
            resolved_model_folder = normalize_and_validate_run_dir(candidate_model_folder)
        except Exception:
            resolved_model_folder = None

    try:
        payload = _load_torch_checkpoint(resolved_checkpoint)
    except Exception as exc:
        raise RuntimeError(f"Failed to load TX checkpoint at {resolved_checkpoint}: {exc}") from exc

    cfg_data, cfg_source = _extract_cfg_from_checkpoint_payload(payload)

    hyper_parameters = payload.get("hyper_parameters", {}) if isinstance(payload, dict) else {}
    if not isinstance(hyper_parameters, dict):
        hyper_parameters = {}

    state_dict = payload.get("state_dict", {}) if isinstance(payload, dict) else {}
    if not isinstance(state_dict, dict):
        state_dict = {}
        warnings.append("Checkpoint did not contain a dict-like `state_dict` payload.")

    num_tensors = len(state_dict)
    total_param_count = 0
    for tensor in state_dict.values():
        numel_fn = getattr(tensor, "numel", None)
        if callable(numel_fn):
            try:
                total_param_count += int(numel_fn())
            except Exception:
                continue

    inferred_kind = _infer_checkpoint_kind_from_state_dict(payload)
    if inferred_kind != "state-transition":
        warnings.append(f"Checkpoint heuristics inferred kind={inferred_kind!r}; expected `state-transition`.")

    model_cfg = cfg_data.get("model", {}) if isinstance(cfg_data, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    data_cfg = cfg_data.get("data", {}) if isinstance(cfg_data, dict) else {}
    data_kwargs = data_cfg.get("kwargs", {}) if isinstance(data_cfg, dict) else {}
    if not isinstance(data_kwargs, dict):
        data_kwargs = {}

    return {
        "checkpoint_type": "tx",
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_filename": resolved_checkpoint.name,
        "checkpoint_extension": resolved_checkpoint.suffix,
        "checkpoint_size_bytes": int(os.path.getsize(resolved_checkpoint)),
        "model_folder": resolved_model_folder,
        "inferred_checkpoint_kind": inferred_kind,
        "checkpoint_payload_summary": {
            "cfg_source": cfg_source,
            "has_cfg_yaml": isinstance(payload, dict) and isinstance(payload.get("cfg_yaml"), str),
            "has_hyper_parameters": isinstance(payload, dict) and ("hyper_parameters" in payload),
            "has_state_dict": isinstance(payload, dict) and ("state_dict" in payload),
            "state_dict_tensor_count": num_tensors,
            "state_dict_total_parameters": total_param_count,
            "state_dict_sample_keys": sorted([str(k) for k in state_dict.keys()])[:20],
            "hyper_parameter_keys": sorted(hyper_parameters.keys())[:50],
        },
        "tx_config_summary": {
            "model_name": _as_optional_str(model_cfg.get("name")),
            "embedding_key": _as_optional_str(data_kwargs.get("embed_key")),
            "perturbation_column": _as_optional_str(data_kwargs.get("pert_col")),
            "cell_type_column": _as_optional_str(data_kwargs.get("cell_type_key")),
            "batch_column": _as_optional_str(data_kwargs.get("batch_col")),
            "control_perturbation": _as_optional_str(data_kwargs.get("control_pert")),
        },
        "warnings": warnings,
    }


def infer_tx_inference_defaults(model_folder: str) -> dict[str, Any]:
    """
    Infer tx inference defaults from a validated STATE run directory.
    """
    resolved = normalize_and_validate_run_dir(model_folder)
    folder = Path(resolved)
    config = _load_yaml_config(folder / "config.yaml")

    data_section = config.get("data", {})
    data_kwargs = data_section.get("kwargs", {}) if isinstance(data_section, dict) else {}
    if not isinstance(data_kwargs, dict):
        data_kwargs = {}

    model_section = config.get("model", {})
    model_kwargs = model_section.get("kwargs", {}) if isinstance(model_section, dict) else {}
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}
    training_section = config.get("training", {})
    if not isinstance(training_section, dict):
        training_section = {}

    raw_train_batch_size = training_section.get("batch_size")
    train_batch_size: int | None
    if isinstance(raw_train_batch_size, (int, float)) and int(raw_train_batch_size) > 0:
        train_batch_size = int(raw_train_batch_size)
    else:
        train_batch_size = None

    checkpoint_dir = folder / "checkpoints"
    checkpoint_files = _list_checkpoint_files(checkpoint_dir)
    checkpoint_path = _choose_checkpoint(checkpoint_dir, checkpoint_files)

    perturbation_column = _as_optional_str(data_kwargs.get("pert_col")) or "drugname_drugconc"
    control_perturbation = _as_optional_str(data_kwargs.get("control_pert"))
    if control_perturbation is None and perturbation_column == "drugname_drugconc":
        control_perturbation = "[('DMSO_TF', 0.0, 'uM')]"
    if control_perturbation is None:
        control_perturbation = "non-targeting"

    return {
        "checkpoint_path": checkpoint_path,
        "perturbation_column": perturbation_column,
        "embedding_key": _as_optional_str(data_kwargs.get("embed_key")),
        "cell_type_column": _as_optional_str(data_kwargs.get("cell_type_key")),
        "batch_column": _as_optional_str(data_kwargs.get("batch_col")),
        "control_perturbation": control_perturbation,
        "output_space": _as_optional_str(data_kwargs.get("output_space")),
        "model_name": _as_optional_str(model_section.get("name")) if isinstance(model_section, dict) else None,
        "nb_loss": bool(model_kwargs.get("nb_loss", False)),
        "batched": True,
        "set_batch_size": train_batch_size,
        "checkpoint_count": len(checkpoint_files),
    }


def inspect_model_folder(model_folder: str) -> dict[str, Any]:
    """
    Inspect a STATE run directory and return inference-relevant metadata.
    """
    resolved = normalize_and_validate_run_dir(model_folder)
    folder = Path(resolved)

    warnings: list[str] = []

    config = _load_yaml_config(folder / "config.yaml")

    data_section = config.get("data", {})
    data_kwargs = data_section.get("kwargs", {}) if isinstance(data_section, dict) else {}
    if not isinstance(data_kwargs, dict):
        data_kwargs = {}

    model_section = config.get("model", {})
    model_kwargs = model_section.get("kwargs", {}) if isinstance(model_section, dict) else {}
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}

    var_dims_path = folder / "var_dims.pkl"
    var_dims: dict[str, Any] = {}
    try:
        raw_dims = _load_pickle_mapping(var_dims_path)
        if isinstance(raw_dims, dict):
            var_dims = raw_dims
        else:
            warnings.append(f"var_dims.pkl contained {type(raw_dims).__name__}; expected dict.")
    except Exception as exc:
        warnings.append(f"Failed to load var_dims.pkl: {exc}")

    artifacts: dict[str, Any] = {
        "config": str(folder / "config.yaml"),
        "var_dims": str(var_dims_path),
        "pert_onehot_map": str(folder / "pert_onehot_map.pt"),
        "data_module": None,
        "batch_onehot_map": None,
        "cell_type_onehot_map": None,
    }

    data_module_path = _pick_existing_file(folder, _REQUIRED_FILE_VARIANTS["data_module"])
    batch_map_path = _pick_existing_file(folder, _REQUIRED_FILE_VARIANTS["batch_onehot_map"])
    celltype_map_path = _pick_existing_file(folder, _REQUIRED_FILE_VARIANTS["cell_type_onehot_map"])
    if data_module_path is not None:
        artifacts["data_module"] = str(data_module_path)
    if batch_map_path is not None:
        artifacts["batch_onehot_map"] = str(batch_map_path)
    if celltype_map_path is not None:
        artifacts["cell_type_onehot_map"] = str(celltype_map_path)

    mapping_sizes: dict[str, int | None] = {
        "perturbations": None,
        "batches": None,
        "cell_types": None,
    }
    mapping_files = {
        "perturbations": folder / "pert_onehot_map.pt",
        "batches": batch_map_path,
        "cell_types": celltype_map_path,
    }
    for key, path in mapping_files.items():
        if path is None:
            continue
        try:
            mapping = _load_mapping(path)
            if isinstance(mapping, dict):
                mapping_sizes[key] = len(mapping)
            else:
                warnings.append(f"{path.name} loaded as {type(mapping).__name__}; expected dict.")
        except Exception as exc:
            warnings.append(f"Failed to load {path.name}: {exc}")

    checkpoint_dir = folder / "checkpoints"
    checkpoint_files = _list_checkpoint_files(checkpoint_dir)
    defaults = infer_tx_inference_defaults(resolved)

    return {
        "model_folder": resolved,
        "artifacts": artifacts,
        "checkpoint_files": checkpoint_files,
        "checkpoint_count": len(checkpoint_files),
        "recommended_checkpoint": defaults.get("checkpoint_path"),
        "config_summary": {
            "run_name": _as_optional_str(config.get("name")),
            "output_dir": _as_optional_str(config.get("output_dir")),
            "model_name": _as_optional_str(model_section.get("name")) if isinstance(model_section, dict) else None,
            "output_space": _as_optional_str(data_kwargs.get("output_space")),
            "embedding_key": _as_optional_str(data_kwargs.get("embed_key")),
            "perturbation_column": _as_optional_str(data_kwargs.get("pert_col")),
            "cell_type_column": _as_optional_str(data_kwargs.get("cell_type_key")),
            "batch_column": _as_optional_str(data_kwargs.get("batch_col")),
            "control_perturbation": _as_optional_str(data_kwargs.get("control_pert")),
            "nb_loss": bool(model_kwargs.get("nb_loss", False)),
        },
        "dimensions": {
            "input_dim": var_dims.get("input_dim"),
            "output_dim": var_dims.get("output_dim"),
            "gene_dim": var_dims.get("gene_dim"),
            "hvg_dim": var_dims.get("hvg_dim"),
            "pert_dim": var_dims.get("pert_dim"),
            "batch_dim": var_dims.get("batch_dim"),
        },
        "mapping_sizes": mapping_sizes,
        "inferred_inference_defaults": defaults,
        "warnings": warnings,
    }


def _decode_string_like(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _safe_decode_array(values: Any) -> list[str]:
    return [_decode_string_like(v) for v in values]


def _array_shape_and_dtype(obj: Any) -> tuple[list[int] | None, str | None, bool]:
    try:
        import h5py
    except Exception:
        return None, None, False

    if isinstance(obj, h5py.Dataset):
        try:
            shape = [int(x) for x in obj.shape]
        except Exception:
            shape = None
        try:
            dtype = str(obj.dtype)
        except Exception:
            dtype = None
        return shape, dtype, False

    if isinstance(obj, h5py.Group):
        keys = set(obj.keys())
        if {"indptr", "indices", "data"}.issubset(keys):
            shape_attr = obj.attrs.get("shape")
            shape: list[int] | None = None
            if shape_attr is not None:
                try:
                    shape = [int(x) for x in shape_attr]
                except Exception:
                    shape = None
            dtype = None
            try:
                dtype = str(obj["data"].dtype)
            except Exception:
                dtype = None
            return shape, dtype, True
    return None, None, False


def _summarize_obs_column(
    obs_group: Any,
    column: str,
    *,
    n_obs: int | None,
    max_top_values: int,
    max_scan_values: int,
    warnings: list[str],
) -> dict[str, Any]:
    try:
        import h5py
    except Exception as exc:
        return {
            "name": column,
            "dtype": "unknown",
            "n_unique": 0,
            "top_values": [],
            "warnings": [f"h5py unavailable while summarizing obs column {column!r}: {exc}"],
        }

    obj = obs_group[column]
    top_values: list[dict[str, Any]] = []
    n_unique = 0
    sampled = False
    dtype_summary = "unknown"
    col_warnings: list[str] = []

    if isinstance(obj, h5py.Group) and {"categories", "codes"}.issubset(set(obj.keys())):
        categories = _safe_decode_array(obj["categories"][:])
        codes = np.asarray(obj["codes"][:])
        valid_codes = codes[codes >= 0]
        try:
            bincount = np.bincount(valid_codes.astype(np.int64), minlength=len(categories))
            nonzero = np.where(bincount > 0)[0]
            n_unique = int(len(nonzero))
            ranked = sorted(nonzero.tolist(), key=lambda i: int(bincount[i]), reverse=True)[:max_top_values]
            top_values = [
                {"value": categories[i], "count": int(bincount[i])}
                for i in ranked
            ]
        except Exception:
            if valid_codes.size > 0:
                unique_codes, counts = np.unique(valid_codes, return_counts=True)
                n_unique = int(len(unique_codes))
                pairs = sorted(
                    ((int(c), int(n)) for c, n in zip(unique_codes.tolist(), counts.tolist(), strict=False)),
                    key=lambda x: x[1],
                    reverse=True,
                )[:max_top_values]
                top_values = [
                    {
                        "value": categories[idx] if 0 <= idx < len(categories) else str(idx),
                        "count": count,
                    }
                    for idx, count in pairs
                ]
        dtype_summary = f"categorical[{str(obj['categories'].dtype)}]"
    elif isinstance(obj, h5py.Dataset):
        total_values = int(obj.shape[0]) if obj.shape else int(n_obs or 0)
        scan_n = min(total_values, max_scan_values)
        sampled = scan_n < total_values
        if sampled:
            col_warnings.append(
                f"Column {column!r} scanned on first {scan_n} / {total_values} values for cardinality summary."
            )
        try:
            if len(obj.shape) > 1:
                raw_values = obj[:scan_n, ...]
                decoded = [str(v.tolist() if hasattr(v, "tolist") else v) for v in raw_values]
            else:
                raw_values = obj[:scan_n]
                decoded = _safe_decode_array(raw_values)
            uniques, counts = np.unique(np.asarray(decoded, dtype=object), return_counts=True)
            n_unique = int(len(uniques))
            pairs = sorted(
                zip(uniques.tolist(), counts.tolist(), strict=False),
                key=lambda x: int(x[1]),
                reverse=True,
            )[:max_top_values]
            top_values = [{"value": str(val), "count": int(count)} for val, count in pairs]
        except Exception as exc:
            col_warnings.append(f"Unable to summarize column values: {type(exc).__name__}: {exc}")
        dtype_summary = str(obj.dtype)
    else:
        col_warnings.append(f"Unsupported obs column backing type: {type(obj).__name__}")

    warnings.extend(col_warnings)
    return {
        "name": column,
        "dtype": dtype_summary,
        "n_unique": n_unique,
        "top_values": top_values,
        "sampled": sampled,
    }


def _find_candidate_columns(obs_names: list[str]) -> dict[str, list[str]]:
    lower_to_name = {name.lower(): name for name in obs_names}

    def _rank(priority: list[str], tokens: list[str]) -> list[str]:
        ranked: list[str] = []
        for key in priority:
            candidate = lower_to_name.get(key.lower())
            if candidate is not None and candidate not in ranked:
                ranked.append(candidate)
        for name in obs_names:
            lname = name.lower()
            if any(tok in lname for tok in tokens) and name not in ranked:
                ranked.append(name)
        return ranked

    return {
        "perturbation": _rank(
            [
                "gene",
                "target_gene",
                "perturbation",
                "drugname_drugconc",
                "condition",
                "guide",
                "sgRNA",
                "pert_col",
            ],
            ["pert", "drug", "target", "guide", "sg"],
        ),
        "cell_type": _rank(
            [
                "cell_type",
                "celltype",
                "cell_line",
                "celltype_col",
                "cellType",
                "ctype",
            ],
            ["cell", "ctype", "line"],
        ),
        "batch": _rank(
            [
                "gem_group",
                "batch",
                "batch_col",
                "donor",
                "plate",
                "lane",
                "sample",
                "experiment",
                "replicate",
            ],
            ["batch", "gem", "donor", "plate", "lane", "sample", "rep"],
        ),
    }


def inspect_adata_schema(adata_path: str, max_top_values: int = 25) -> dict[str, Any]:
    """
    Inspect an AnnData file and return schema metadata useful for guided ST training setup.
    """
    if max_top_values <= 0:
        raise ValueError("`max_top_values` must be > 0.")

    resolved = str(Path(adata_path).expanduser().resolve())
    adata_obj = Path(resolved)
    if not adata_obj.exists():
        raise FileNotFoundError(f"AnnData file not found: {resolved}")
    if not adata_obj.is_file():
        raise ValueError(f"AnnData path is not a file: {resolved}")

    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(f"h5py is required to inspect AnnData schemas: {exc}") from exc

    warnings: list[str] = []
    obs_columns: list[dict[str, Any]] = []
    obsm_summaries: list[dict[str, Any]] = []

    n_obs: int | None = None
    n_vars: int | None = None
    x_shape: list[int] | None = None
    x_dtype: str | None = None
    x_is_sparse = False

    with h5py.File(resolved, "r") as f:
        if "X" in f:
            x_shape, x_dtype, x_is_sparse = _array_shape_and_dtype(f["X"])
            if x_shape is not None and len(x_shape) == 2:
                n_obs, n_vars = int(x_shape[0]), int(x_shape[1])
        else:
            warnings.append("Missing top-level X matrix.")

        if n_obs is None and "obs" in f:
            obs_group = f["obs"]
            if isinstance(obs_group, h5py.Group):
                if "_index" in obs_group:
                    try:
                        n_obs = int(obs_group["_index"].shape[0])
                    except Exception:
                        pass
                if n_obs is None:
                    for key in obs_group.keys():
                        obj = obs_group[key]
                        if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 1:
                            n_obs = int(obj.shape[0])
                            break
                        if isinstance(obj, h5py.Group) and "codes" in obj:
                            n_obs = int(obj["codes"].shape[0])
                            break

        if n_vars is None and "var" in f:
            var_group = f["var"]
            if isinstance(var_group, h5py.Group):
                if "_index" in var_group:
                    try:
                        n_vars = int(var_group["_index"].shape[0])
                    except Exception:
                        pass
                if n_vars is None:
                    for key in var_group.keys():
                        obj = var_group[key]
                        if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 1:
                            n_vars = int(obj.shape[0])
                            break
                        if isinstance(obj, h5py.Group) and "codes" in obj:
                            n_vars = int(obj["codes"].shape[0])
                            break

        if "obs" in f and isinstance(f["obs"], h5py.Group):
            obs_group = f["obs"]
            for key in sorted(obs_group.keys()):
                if key == "_index":
                    continue
                obs_columns.append(
                    _summarize_obs_column(
                        obs_group,
                        key,
                        n_obs=n_obs,
                        max_top_values=max_top_values,
                        max_scan_values=200000,
                        warnings=warnings,
                    )
                )
        else:
            warnings.append("Missing obs group in AnnData file.")

        if "obsm" in f and isinstance(f["obsm"], h5py.Group):
            obsm_group = f["obsm"]
            for key in sorted(obsm_group.keys()):
                shape, dtype, is_sparse = _array_shape_and_dtype(obsm_group[key])
                obsm_summaries.append(
                    {
                        "key": key,
                        "shape": shape,
                        "dtype": dtype,
                        "is_sparse": is_sparse,
                    }
                )

    obs_names = [entry["name"] for entry in obs_columns]
    candidate_columns = _find_candidate_columns(obs_names)

    perturbation_candidates = candidate_columns.get("perturbation", [])[:3]
    scan_columns = set(perturbation_candidates) if perturbation_candidates else set(obs_names)
    control_tokens = ("non-target", "control", "dmso", "vehicle", "nt_", "nt-", "ctrl")
    control_candidates: list[dict[str, Any]] = []
    n_obs_float = float(n_obs or 0)
    for summary in obs_columns:
        if summary["name"] not in scan_columns:
            continue
        for item in summary.get("top_values", []):
            value = str(item.get("value", ""))
            count = int(item.get("count", 0))
            lower_value = value.lower()
            if any(token in lower_value for token in control_tokens):
                control_candidates.append(
                    {
                        "column": summary["name"],
                        "value": value,
                        "count": count,
                        "fraction": round((count / n_obs_float), 6) if n_obs_float > 0 else None,
                    }
                )
    control_candidates = sorted(control_candidates, key=lambda x: int(x.get("count") or 0), reverse=True)

    return {
        "adata_path": resolved,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "x": {
            "shape": x_shape,
            "dtype": x_dtype,
            "is_sparse": x_is_sparse,
        },
        "obs_columns": obs_columns,
        "obsm": obsm_summaries,
        "candidate_columns": candidate_columns,
        "control_label_candidates": control_candidates,
        "warnings": warnings,
    }


def _is_glob_pattern(text: str) -> bool:
    return any(char in text for char in "*?[]{}")


def _expand_braces(pattern: str) -> list[str]:
    match = re.search(r"\{([^}]+)\}", pattern)
    if match is None:
        return [pattern]
    before = pattern[: match.start()]
    after = pattern[match.end() :]
    options = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if not options:
        return [pattern]
    expanded: list[str] = []
    for option in options:
        expanded.extend(_expand_braces(before + option + after))
    return expanded


def _find_dataset_files(dataset_path: str) -> list[str]:
    path_text = str(dataset_path).strip()
    if not path_text:
        return []

    files: list[Path] = []
    if _is_glob_pattern(path_text):
        patterns = _expand_braces(path_text)
        for pattern in patterns:
            if pattern.endswith(".h5") or pattern.endswith(".h5ad"):
                matches = [Path(p) for p in sorted(glob.glob(pattern))]
                files.extend(matches)
            else:
                files.extend(Path(p) for p in sorted(glob.glob(pattern.rstrip("/") + "/*.h5")))
                files.extend(Path(p) for p in sorted(glob.glob(pattern.rstrip("/") + "/*.h5ad")))
    else:
        path = Path(path_text).expanduser()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.h5")))
            files.extend(sorted(path.glob("*.h5ad")))

    deduped = sorted({str(p.resolve()) for p in files if p.exists() and p.is_file()})
    return deduped


def _load_toml_mapping(toml_path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except Exception:
        import tomli as tomllib  # type: ignore

    with toml_path.open("rb") as f:
        payload = tomllib.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level TOML mapping in {toml_path}, got {type(payload).__name__}")
    return payload


_CONTROL_LABEL_TOKENS: Final[tuple[str, ...]] = (
    "non-target",
    "control",
    "dmso",
    "vehicle",
    "nt_",
    "nt-",
    "ctrl",
)


def _validate_dataset_paths_mapping(dataset_paths: dict[str, Any]) -> dict[str, str]:
    if not isinstance(dataset_paths, dict) or not dataset_paths:
        raise ValueError("`dataset_paths` must be a non-empty mapping of dataset name -> path/glob.")

    normalized: dict[str, str] = {}
    for raw_name, raw_path in sorted(dataset_paths.items()):
        name = str(raw_name).strip()
        path = str(raw_path).strip()
        if not name:
            raise ValueError("Dataset names in `dataset_paths` must be non-empty.")
        if not path:
            raise ValueError(f"Dataset path for {name!r} must be non-empty.")
        normalized[name] = path
    return normalized


def _pick_column_candidate(candidates: Any, fallback: str) -> str:
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return fallback


def _resolve_split_columns(
    dataset_paths: dict[str, str],
    cell_type_column: str | None,
    perturbation_column: str | None,
) -> tuple[str, str, dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    resolved_cell_type = (
        cell_type_column.strip() if isinstance(cell_type_column, str) and cell_type_column.strip() else None
    )
    resolved_perturbation = (
        perturbation_column.strip()
        if isinstance(perturbation_column, str) and perturbation_column.strip()
        else None
    )

    sample_schema: dict[str, Any] | None = None
    if resolved_cell_type is not None and resolved_perturbation is not None:
        return resolved_cell_type, resolved_perturbation, sample_schema, warnings

    sample_file: str | None = None
    for dataset_name, dataset_path in sorted(dataset_paths.items()):
        files = _find_dataset_files(dataset_path)
        if files:
            sample_file = files[0]
            break
        warnings.append(f"Dataset {dataset_name!r} did not resolve to files while inferring split columns.")

    if sample_file is None:
        raise ValueError(
            "Unable to infer split columns because no dataset files could be resolved. "
            "Provide `cell_type_column` and `perturbation_column` explicitly."
        )

    sample_schema = inspect_adata_schema(sample_file, max_top_values=10)
    candidate_columns = sample_schema.get("candidate_columns", {})
    if not isinstance(candidate_columns, dict):
        candidate_columns = {}

    if resolved_cell_type is None:
        resolved_cell_type = _pick_column_candidate(candidate_columns.get("cell_type"), "cell_type")
        warnings.append(
            f"Inferred `cell_type_column={resolved_cell_type!r}` from sample AnnData schema at {sample_file}."
        )
    if resolved_perturbation is None:
        resolved_perturbation = _pick_column_candidate(candidate_columns.get("perturbation"), "gene")
        warnings.append(
            f"Inferred `perturbation_column={resolved_perturbation!r}` from sample AnnData schema at {sample_file}."
        )

    return resolved_cell_type, resolved_perturbation, sample_schema, warnings


def _extract_obs_categorical(obj: Any) -> tuple[list[str], np.ndarray] | None:
    try:
        import h5py
    except Exception:
        return None

    if not isinstance(obj, h5py.Group):
        return None
    if not {"categories", "codes"}.issubset(set(obj.keys())):
        return None

    categories = _safe_decode_array(obj["categories"][:])
    codes = np.asarray(obj["codes"][:], dtype=np.int64)
    return categories, codes


def _read_obs_column_values_as_strings(obs_group: Any, column: str) -> np.ndarray:
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(f"h5py is required for split source inspection: {exc}") from exc

    obj = obs_group[column]
    categorical = _extract_obs_categorical(obj)
    if categorical is not None:
        categories, codes = categorical
        out = np.empty(codes.shape[0], dtype=object)
        out[:] = ""
        valid = codes >= 0
        if np.any(valid):
            valid_codes = codes[valid]
            mapped = [
                categories[int(code)] if 0 <= int(code) < len(categories) else str(int(code))
                for code in valid_codes.tolist()
            ]
            out[valid] = np.asarray(mapped, dtype=object)
        return out

    if isinstance(obj, h5py.Dataset):
        raw = obj[:]
        if len(obj.shape) > 1:
            return np.asarray(
                [
                    str(item.tolist() if hasattr(item, "tolist") else item)
                    for item in raw
                ],
                dtype=object,
            )
        return np.asarray(_safe_decode_array(raw), dtype=object)

    raise ValueError(f"Unsupported obs column type for {column!r}: {type(obj).__name__}")


def _scan_split_context_counts(
    dataset_paths: dict[str, str],
    *,
    cell_type_column: str,
    perturbation_column: str,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]], dict[str, list[str]], list[str], list[str]]:
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(f"h5py is required for split source inspection: {exc}") from exc

    context_counts: dict[str, dict[str, int]] = {}
    context_meta: dict[str, dict[str, Any]] = {}
    dataset_files: dict[str, list[str]] = {}
    warnings: list[str] = []
    errors: list[str] = []

    for dataset_name, dataset_path in sorted(dataset_paths.items()):
        files = _find_dataset_files(dataset_path)
        dataset_files[dataset_name] = files
        if not files:
            warnings.append(f"Dataset {dataset_name!r} did not resolve to any .h5/.h5ad files from {dataset_path!r}.")
            continue

        for file_path in files:
            try:
                with h5py.File(file_path, "r") as handle:
                    if "obs" not in handle or not isinstance(handle["obs"], h5py.Group):
                        warnings.append(f"File {file_path} is missing an `obs` group; skipping.")
                        continue

                    obs_group = handle["obs"]
                    if cell_type_column not in obs_group:
                        warnings.append(
                            f"File {file_path} is missing obs/{cell_type_column!r}; skipping for split inspection."
                        )
                        continue
                    if perturbation_column not in obs_group:
                        warnings.append(
                            f"File {file_path} is missing obs/{perturbation_column!r}; skipping for split inspection."
                        )
                        continue

                    ct_obj = obs_group[cell_type_column]
                    pert_obj = obs_group[perturbation_column]
                    ct_categorical = _extract_obs_categorical(ct_obj)
                    pert_categorical = _extract_obs_categorical(pert_obj)

                    if ct_categorical is not None and pert_categorical is not None:
                        ct_categories, ct_codes = ct_categorical
                        pert_categories, pert_codes = pert_categorical
                        if ct_codes.shape[0] != pert_codes.shape[0]:
                            warnings.append(
                                f"File {file_path} has mismatched row counts between "
                                f"{cell_type_column!r} and {perturbation_column!r}; skipping."
                            )
                            continue

                        valid = (ct_codes >= 0) & (pert_codes >= 0)
                        if np.any(valid):
                            pair_codes = np.stack(
                                (ct_codes[valid].astype(np.int64), pert_codes[valid].astype(np.int64)),
                                axis=1,
                            )
                            unique_pairs, pair_counts = np.unique(pair_codes, axis=0, return_counts=True)
                            for (ct_code, pert_code), count in zip(
                                unique_pairs.tolist(),
                                pair_counts.tolist(),
                                strict=False,
                            ):
                                ct_index = int(ct_code)
                                pert_index = int(pert_code)
                                ct_name = (
                                    ct_categories[ct_index]
                                    if 0 <= ct_index < len(ct_categories)
                                    else str(ct_index)
                                )
                                pert_name = (
                                    pert_categories[pert_index]
                                    if 0 <= pert_index < len(pert_categories)
                                    else str(pert_index)
                                )
                                ct_name = str(ct_name).strip()
                                pert_name = str(pert_name).strip()
                                if not ct_name or not pert_name:
                                    continue

                                context = f"{dataset_name}.{ct_name}"
                                counts_by_pert = context_counts.setdefault(context, {})
                                counts_by_pert[pert_name] = counts_by_pert.get(pert_name, 0) + int(count)
                                meta = context_meta.setdefault(
                                    context,
                                    {"dataset": dataset_name, "cell_type": ct_name, "source_files": set()},
                                )
                                meta["source_files"].add(file_path)
                        continue

                    cell_types = _read_obs_column_values_as_strings(obs_group, cell_type_column)
                    perturbations = _read_obs_column_values_as_strings(obs_group, perturbation_column)
                    if cell_types.shape[0] != perturbations.shape[0]:
                        warnings.append(
                            f"File {file_path} has mismatched row counts between "
                            f"{cell_type_column!r} and {perturbation_column!r}; skipping."
                        )
                        continue

                    for ct_name_raw, pert_name_raw in zip(cell_types.tolist(), perturbations.tolist(), strict=False):
                        ct_name = str(ct_name_raw).strip()
                        pert_name = str(pert_name_raw).strip()
                        if not ct_name or not pert_name:
                            continue
                        context = f"{dataset_name}.{ct_name}"
                        counts_by_pert = context_counts.setdefault(context, {})
                        counts_by_pert[pert_name] = counts_by_pert.get(pert_name, 0) + 1
                        meta = context_meta.setdefault(
                            context,
                            {"dataset": dataset_name, "cell_type": ct_name, "source_files": set()},
                        )
                        meta["source_files"].add(file_path)
            except Exception as exc:
                errors.append(
                    f"Failed to inspect {file_path} for split metadata: {type(exc).__name__}: {exc}"
                )

    for meta in context_meta.values():
        source_files = sorted(str(p) for p in meta.get("source_files", set()))
        meta["source_files"] = source_files
        meta["source_file_count"] = len(source_files)

    return context_counts, context_meta, dataset_files, warnings, errors


def _infer_control_candidates(context_counts: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    aggregate: dict[str, int] = {}
    for counts in context_counts.values():
        for perturbation, count in counts.items():
            aggregate[perturbation] = aggregate.get(perturbation, 0) + int(count)

    ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
    candidates: list[dict[str, Any]] = []
    for perturbation, count in ranked:
        lower = perturbation.lower()
        if any(token in lower for token in _CONTROL_LABEL_TOKENS):
            candidates.append({"value": perturbation, "count": int(count)})
    return candidates


def _summarize_context_counts(
    context_counts: dict[str, dict[str, int]],
    context_meta: dict[str, dict[str, Any]],
    *,
    control_perturbation: str | None,
    max_top_perturbations: int,
    max_contexts: int,
) -> tuple[list[dict[str, Any]], int]:
    if max_contexts <= 0:
        raise ValueError("`max_contexts` must be > 0.")

    contexts = sorted(context_counts.keys())
    total_contexts = len(contexts)
    selected_contexts = contexts[:max_contexts]

    summaries: list[dict[str, Any]] = []
    for context in selected_contexts:
        counts = context_counts.get(context, {})
        total_cells = int(sum(int(v) for v in counts.values()))
        control_cells = int(counts.get(control_perturbation, 0)) if control_perturbation is not None else 0
        non_control_cells = total_cells - control_cells
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        top_perturbations = [
            {"perturbation": perturbation, "count": int(count)}
            for perturbation, count in ranked[:max_top_perturbations]
        ]
        meta = context_meta.get(context, {})
        summaries.append(
            {
                "context": context,
                "dataset": str(meta.get("dataset") or context.split(".", 1)[0]),
                "cell_type": str(meta.get("cell_type") or context.split(".", 1)[-1]),
                "source_file_count": int(meta.get("source_file_count") or 0),
                "source_files": list(meta.get("source_files") or []),
                "total_cells": total_cells,
                "control_cells": control_cells,
                "non_control_cells": non_control_cells,
                "unique_perturbations": len(counts),
                "top_perturbations": top_perturbations,
            }
        )

    return summaries, total_contexts


def _normalize_fraction(name: str, value: float) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"`{name}` must be within [0, 1], got {value!r}.")
    return parsed


def _deterministic_context_seed(base_seed: int, context: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{context}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _random_split_perturbations(
    perturbations: list[str],
    *,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    context: str,
) -> tuple[list[str], list[str]]:
    shuffled = list(perturbations)
    rng = np.random.default_rng(_deterministic_context_seed(seed, context))
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)
    if n_val + n_test > n_total:
        raise ValueError(
            f"Invalid fractions for context {context!r}: val={val_fraction}, test={test_fraction}, "
            f"sum exceeds 1.0."
        )

    val_items = sorted(shuffled[:n_val])
    test_items = sorted(shuffled[n_val : n_val + n_test])
    return val_items, test_items


def _normalize_zeroshot_contexts(zeroshot_contexts: dict[str, Any] | None) -> dict[str, str]:
    if zeroshot_contexts is None:
        return {}
    if not isinstance(zeroshot_contexts, dict):
        raise ValueError("`zeroshot_contexts` must be a mapping of `dataset.cell_type` -> split.")

    normalized: dict[str, str] = {}
    for raw_context, raw_split in sorted(zeroshot_contexts.items()):
        context = str(raw_context).strip()
        split = str(raw_split).strip().lower()
        if not context:
            raise ValueError("`zeroshot_contexts` contains an empty context key.")
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid zeroshot split {raw_split!r} for context {context!r}. "
                "Expected one of: train, val, test."
            )
        normalized[context] = split
    return normalized


def _normalize_fewshot_overrides(
    fewshot_overrides: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    if fewshot_overrides is None:
        return {}, []
    if not isinstance(fewshot_overrides, dict):
        raise ValueError("`fewshot_overrides` must be a mapping of context -> {val/test lists}.")

    warnings: list[str] = []
    normalized: dict[str, dict[str, list[str]]] = {}
    for raw_context, raw_spec in sorted(fewshot_overrides.items()):
        context = str(raw_context).strip()
        if not context:
            raise ValueError("`fewshot_overrides` contains an empty context key.")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Fewshot override for context {context!r} must be a mapping.")

        val_raw = raw_spec.get("val", [])
        test_raw = raw_spec.get("test", [])
        if not isinstance(val_raw, list):
            raise ValueError(f"Fewshot override for context {context!r} has non-list `val`.")
        if not isinstance(test_raw, list):
            raise ValueError(f"Fewshot override for context {context!r} has non-list `test`.")

        val_values = sorted({str(item).strip() for item in val_raw if str(item).strip()})
        test_values = sorted({str(item).strip() for item in test_raw if str(item).strip()})
        overlap = sorted(set(val_values) & set(test_values))
        if overlap:
            warnings.append(
                f"Fewshot context {context!r} has {len(overlap)} perturbations in both val and test; "
                "this is allowed and will duplicate those perturbations across both splits."
            )

        normalized[context] = {"val": val_values, "test": test_values}

    return normalized, warnings


def _toml_quote_double(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_quote_single(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def _toml_list_literal(values: list[str]) -> str:
    return "[" + ", ".join(_toml_quote_single(item) for item in values) + "]"


def _render_tx_split_toml(
    *,
    dataset_paths: dict[str, str],
    training_section: dict[str, str],
    zeroshot_section: dict[str, str],
    fewshot_section: dict[str, dict[str, list[str]]],
) -> str:
    lines: list[str] = []
    lines.append("[datasets]")
    for dataset_name, dataset_path in sorted(dataset_paths.items()):
        lines.append(f"{_toml_quote_double(dataset_name)} = {_toml_quote_double(dataset_path)}")
    lines.append("")

    lines.append("[training]")
    for dataset_name, split in sorted(training_section.items()):
        lines.append(f"{_toml_quote_double(dataset_name)} = {_toml_quote_double(split)}")
    lines.append("")

    lines.append("[zeroshot]")
    for context, split in sorted(zeroshot_section.items()):
        lines.append(f"{_toml_quote_double(context)} = {_toml_quote_double(split)}")
    lines.append("")

    lines.append("[fewshot]")
    for context, spec in sorted(fewshot_section.items()):
        escaped_context = context.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'[fewshot."{escaped_context}"]')
        lines.append(f"val = {_toml_list_literal(list(spec.get('val', [])))}")
        lines.append(f"test = {_toml_list_literal(list(spec.get('test', [])))}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def inspect_tx_split_sources(
    dataset_paths: dict[str, Any],
    cell_type_column: str | None = None,
    perturbation_column: str | None = None,
    control_perturbation: str | None = None,
    max_top_perturbations: int = 10,
    max_contexts: int = 200,
) -> dict[str, Any]:
    """
    Inspect dataset sources for TX split/TOML authoring and return context-level summaries.

    This expands glob-style dataset paths, scans contexts (`dataset.cell_type`) and perturbation
    counts, and suggests split-relevant columns when omitted.
    """
    if max_top_perturbations <= 0:
        raise ValueError("`max_top_perturbations` must be > 0.")
    if max_contexts <= 0:
        raise ValueError("`max_contexts` must be > 0.")

    normalized_paths = _validate_dataset_paths_mapping(dataset_paths)
    resolved_cell_type, resolved_perturbation, sample_schema, column_warnings = _resolve_split_columns(
        normalized_paths,
        cell_type_column=cell_type_column,
        perturbation_column=perturbation_column,
    )

    context_counts, context_meta, dataset_files, scan_warnings, scan_errors = _scan_split_context_counts(
        normalized_paths,
        cell_type_column=resolved_cell_type,
        perturbation_column=resolved_perturbation,
    )
    control_candidates = _infer_control_candidates(context_counts)

    resolved_control = (
        control_perturbation.strip() if isinstance(control_perturbation, str) and control_perturbation.strip() else None
    )
    if resolved_control is None and control_candidates:
        resolved_control = str(control_candidates[0]["value"])
        column_warnings.append(
            f"Inferred control perturbation label {resolved_control!r} from perturbation distributions."
        )

    context_summaries, total_contexts = _summarize_context_counts(
        context_counts,
        context_meta,
        control_perturbation=resolved_control,
        max_top_perturbations=max_top_perturbations,
        max_contexts=max_contexts,
    )

    datasets_payload = [
        {
            "name": dataset_name,
            "path": dataset_path,
            "resolved_file_count": len(dataset_files.get(dataset_name, [])),
            "sample_files": list(dataset_files.get(dataset_name, []))[:3],
        }
        for dataset_name, dataset_path in sorted(normalized_paths.items())
    ]

    candidate_columns = {}
    if isinstance(sample_schema, dict):
        raw_candidates = sample_schema.get("candidate_columns", {})
        if isinstance(raw_candidates, dict):
            candidate_columns = raw_candidates

    return {
        "datasets": datasets_payload,
        "resolved_columns": {
            "cell_type_column": resolved_cell_type,
            "perturbation_column": resolved_perturbation,
            "control_perturbation": resolved_control,
        },
        "candidate_columns": candidate_columns,
        "context_count_total": total_contexts,
        "context_count_returned": len(context_summaries),
        "contexts": context_summaries,
        "control_candidates": control_candidates[:25],
        "warnings": column_warnings + scan_warnings,
        "errors": scan_errors,
    }


def plan_tx_split_toml(
    dataset_paths: dict[str, Any],
    training_datasets: list[str] | None = None,
    cell_type_column: str | None = None,
    perturbation_column: str | None = None,
    control_perturbation: str | None = None,
    random_holdout_contexts: list[str] | None = None,
    random_test_fraction: float = 0.7,
    random_val_fraction: float = 0.0,
    random_seed: int = 0,
    zeroshot_contexts: dict[str, Any] | None = None,
    fewshot_overrides: dict[str, Any] | None = None,
    output_path: str | None = None,
    overwrite: bool = False,
    include_toml: bool = False,
    preview_lines: int = 120,
    max_contexts_in_summary: int = 200,
) -> dict[str, Any]:
    """
    Build a TX split TOML from dataset paths plus random/manual split directives.

    Common pattern:
    - set `[training]` datasets
    - optionally set full-context `[zeroshot]` entries
    - optionally generate random fewshot perturbation holdouts for specific contexts
      (e.g., hold out 70% perturbations in `replogle.rpe1`)
    - optionally overlay manual `fewshot_overrides`

    Set `output_path` to write the TOML to disk.
    """
    if preview_lines <= 0:
        raise ValueError("`preview_lines` must be > 0.")
    if max_contexts_in_summary <= 0:
        raise ValueError("`max_contexts_in_summary` must be > 0.")

    normalized_paths = _validate_dataset_paths_mapping(dataset_paths)
    resolved_cell_type, resolved_perturbation, _, column_warnings = _resolve_split_columns(
        normalized_paths,
        cell_type_column=cell_type_column,
        perturbation_column=perturbation_column,
    )

    context_counts, context_meta, dataset_files, scan_warnings, scan_errors = _scan_split_context_counts(
        normalized_paths,
        cell_type_column=resolved_cell_type,
        perturbation_column=resolved_perturbation,
    )
    control_candidates = _infer_control_candidates(context_counts)
    resolved_control = (
        control_perturbation.strip() if isinstance(control_perturbation, str) and control_perturbation.strip() else None
    )
    warnings: list[str] = []
    warnings.extend(column_warnings)
    warnings.extend(scan_warnings)
    warnings.extend(scan_errors)

    if resolved_control is None and control_candidates:
        resolved_control = str(control_candidates[0]["value"])
        warnings.append(
            f"Inferred control perturbation label {resolved_control!r} from perturbation distributions."
        )
    elif resolved_control is not None:
        observed = any(resolved_control in counts for counts in context_counts.values())
        if not observed:
            warnings.append(
                f"Requested control perturbation label {resolved_control!r} was not observed in scanned contexts."
            )

    test_fraction = _normalize_fraction("random_test_fraction", random_test_fraction)
    val_fraction = _normalize_fraction("random_val_fraction", random_val_fraction)
    if test_fraction + val_fraction > 1.0:
        raise ValueError(
            "`random_test_fraction + random_val_fraction` must be <= 1.0."
        )
    if random_seed < 0:
        raise ValueError("`random_seed` must be >= 0.")

    if training_datasets is None:
        selected_training_datasets = sorted(normalized_paths.keys())
    else:
        cleaned = sorted({str(item).strip() for item in training_datasets if str(item).strip()})
        unknown = sorted(set(cleaned) - set(normalized_paths.keys()))
        if unknown:
            raise ValueError(
                "Unknown dataset names in `training_datasets`: " + ", ".join(unknown)
            )
        selected_training_datasets = cleaned
    training_section = {dataset: "train" for dataset in selected_training_datasets}

    zeroshot_section = _normalize_zeroshot_contexts(zeroshot_contexts)
    fewshot_manual, manual_warnings = _normalize_fewshot_overrides(fewshot_overrides)
    warnings.extend(manual_warnings)

    available_contexts = set(context_counts.keys())
    unknown_zeroshot_contexts = sorted(set(zeroshot_section.keys()) - available_contexts)
    if unknown_zeroshot_contexts:
        raise ValueError(
            "Unknown contexts in `zeroshot_contexts`: " + ", ".join(unknown_zeroshot_contexts)
        )
    unknown_manual_contexts = sorted(set(fewshot_manual.keys()) - available_contexts)
    if unknown_manual_contexts:
        raise ValueError(
            "Unknown contexts in `fewshot_overrides`: " + ", ".join(unknown_manual_contexts)
        )

    random_contexts = sorted(
        {str(item).strip() for item in (random_holdout_contexts or []) if str(item).strip()}
    )
    unknown_random_contexts = sorted(set(random_contexts) - available_contexts)
    if unknown_random_contexts:
        raise ValueError(
            "Unknown contexts in `random_holdout_contexts`: " + ", ".join(unknown_random_contexts)
        )

    random_assignments: dict[str, dict[str, list[str]]] = {}
    random_assignment_summary: list[dict[str, Any]] = []
    for context in random_contexts:
        if context in zeroshot_section:
            warnings.append(
                f"Skipping random holdout for context {context!r} because it is already assigned in `zeroshot_contexts`."
            )
            continue

        perts = sorted(context_counts[context].keys())
        if resolved_control is not None:
            perts = [pert for pert in perts if pert != resolved_control]

        val_items, test_items = _random_split_perturbations(
            perts,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            seed=random_seed,
            context=context,
        )
        if test_fraction > 0 and perts and not test_items:
            warnings.append(
                f"Context {context!r} has too few perturbations for requested "
                f"test fraction {test_fraction}; test split resolved to 0 perturbations."
            )
        if val_fraction > 0 and perts and not val_items:
            warnings.append(
                f"Context {context!r} has too few perturbations for requested "
                f"val fraction {val_fraction}; val split resolved to 0 perturbations."
            )

        random_assignments[context] = {"val": val_items, "test": test_items}
        random_assignment_summary.append(
            {
                "context": context,
                "non_control_perturbation_count": len(perts),
                "val_count": len(val_items),
                "test_count": len(test_items),
                "train_count": max(0, len(perts) - len(val_items) - len(test_items)),
                "val_preview": val_items[:20],
                "test_preview": test_items[:20],
            }
        )

    fewshot_section = dict(random_assignments)
    for context, spec in fewshot_manual.items():
        if context in fewshot_section:
            warnings.append(
                f"Manual fewshot override replaced random assignment for context {context!r}."
            )
        fewshot_section[context] = {"val": list(spec.get("val", [])), "test": list(spec.get("test", []))}

    toml_text = _render_tx_split_toml(
        dataset_paths=normalized_paths,
        training_section=training_section,
        zeroshot_section=zeroshot_section,
        fewshot_section=fewshot_section,
    )

    written_output_path: str | None = None
    if output_path is not None and str(output_path).strip():
        resolved_output = Path(output_path).expanduser().resolve()
        if resolved_output.exists() and not overwrite:
            raise FileExistsError(
                f"Output TOML already exists at {resolved_output}. Set `overwrite=True` to replace it."
            )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(toml_text, encoding="utf-8")
        written_output_path = str(resolved_output)

    split_cell_counts = {"train": 0, "val": 0, "test": 0}
    context_assignments: list[dict[str, Any]] = []
    assignment_mode_counts: dict[str, int] = {}
    for context in sorted(context_counts.keys()):
        counts = context_counts[context]
        dataset_name = context.split(".", 1)[0]
        total_cells = int(sum(int(v) for v in counts.values()))
        control_cells = int(counts.get(resolved_control, 0)) if resolved_control is not None else 0

        if context in zeroshot_section:
            split = zeroshot_section[context]
            mode = f"zeroshot:{split}"
            train_cells = control_cells
            val_cells = 0
            test_cells = 0
            non_control_cells = total_cells - control_cells
            if split == "train":
                train_cells += non_control_cells
            elif split == "val":
                val_cells += non_control_cells
            elif split == "test":
                test_cells += non_control_cells
        elif context in fewshot_section:
            mode = "fewshot"
            spec = fewshot_section[context]
            val_set = set(spec.get("val", []))
            test_set = set(spec.get("test", []))
            train_cells = 0
            val_cells = 0
            test_cells = 0
            for perturbation, count in counts.items():
                count_int = int(count)
                if resolved_control is not None and perturbation == resolved_control:
                    train_cells += count_int
                    continue
                in_val = perturbation in val_set
                in_test = perturbation in test_set
                if in_val:
                    val_cells += count_int
                if in_test:
                    test_cells += count_int
                if not in_val and not in_test:
                    train_cells += count_int
        elif dataset_name in training_section:
            mode = "training"
            train_cells = total_cells
            val_cells = 0
            test_cells = 0
        else:
            mode = "excluded"
            train_cells = 0
            val_cells = 0
            test_cells = 0

        split_cell_counts["train"] += int(train_cells)
        split_cell_counts["val"] += int(val_cells)
        split_cell_counts["test"] += int(test_cells)
        assignment_mode_counts[mode] = assignment_mode_counts.get(mode, 0) + 1
        context_assignments.append(
            {
                "context": context,
                "mode": mode,
                "total_cells": total_cells,
                "control_cells": control_cells,
                "assigned_cells": {
                    "train": int(train_cells),
                    "val": int(val_cells),
                    "test": int(test_cells),
                },
            }
        )

    preview = toml_text.splitlines()[:preview_lines]
    if len(toml_text.splitlines()) > preview_lines:
        preview.append("... (truncated)")

    context_assignments = context_assignments[:max_contexts_in_summary]
    context_summaries, total_contexts = _summarize_context_counts(
        context_counts,
        context_meta,
        control_perturbation=resolved_control,
        max_top_perturbations=8,
        max_contexts=max_contexts_in_summary,
    )

    return {
        "status": "ready",
        "datasets": [
            {
                "name": dataset_name,
                "path": dataset_path,
                "resolved_file_count": len(dataset_files.get(dataset_name, [])),
                "sample_files": list(dataset_files.get(dataset_name, []))[:3],
            }
            for dataset_name, dataset_path in sorted(normalized_paths.items())
        ],
        "resolved_columns": {
            "cell_type_column": resolved_cell_type,
            "perturbation_column": resolved_perturbation,
            "control_perturbation": resolved_control,
        },
        "training_datasets": selected_training_datasets,
        "zeroshot_contexts": zeroshot_section,
        "fewshot_context_count": len(fewshot_section),
        "random_assignment_summary": random_assignment_summary,
        "split_cell_counts": split_cell_counts,
        "context_count_total": total_contexts,
        "context_summaries": context_summaries,
        "context_assignment_mode_counts": assignment_mode_counts,
        "context_assignments": context_assignments,
        "control_candidates": control_candidates[:25],
        "toml_preview": preview,
        "output_path": written_output_path,
        "toml_text": toml_text if include_toml else None,
        "warnings": warnings,
    }


def inspect_tx_split_toml(
    toml_path: str,
    sample_files_per_dataset: int = 1,
    inspect_sample_schemas: bool = True,
) -> dict[str, Any]:
    """
    Inspect a STATE TX split TOML and return training/split and schema-relevant metadata.
    """
    if sample_files_per_dataset <= 0:
        raise ValueError("`sample_files_per_dataset` must be > 0.")

    resolved = str(Path(toml_path).expanduser().resolve())
    toml_obj = Path(resolved)
    if not toml_obj.exists():
        raise FileNotFoundError(f"TOML config not found: {resolved}")
    if not toml_obj.is_file():
        raise ValueError(f"TOML path is not a file: {resolved}")

    payload = _load_toml_mapping(toml_obj)
    errors: list[str] = []
    warnings: list[str] = []

    datasets_section = payload.get("datasets", {})
    training_section = payload.get("training", {})
    zeroshot_section = payload.get("zeroshot", {})
    fewshot_section = payload.get("fewshot", {})

    if not isinstance(datasets_section, dict):
        errors.append("`[datasets]` must be a mapping.")
        datasets_section = {}
    if not isinstance(training_section, dict):
        errors.append("`[training]` must be a mapping.")
        training_section = {}
    if not isinstance(zeroshot_section, dict):
        errors.append("`[zeroshot]` must be a mapping when present.")
        zeroshot_section = {}
    if not isinstance(fewshot_section, dict):
        errors.append("`[fewshot]` must be a mapping when present.")
        fewshot_section = {}

    if not datasets_section:
        errors.append("Missing required `[datasets]` section or it is empty.")
    if not training_section:
        errors.append("Missing required `[training]` section or it is empty.")

    dataset_entries: list[dict[str, Any]] = []
    dataset_sample_files: dict[str, list[str]] = {}
    for dataset_name, raw_path in sorted(datasets_section.items()):
        path_text = str(raw_path)
        expanded_path = str(Path(path_text).expanduser())
        kind = "glob" if _is_glob_pattern(path_text) else ("file" if Path(expanded_path).is_file() else "dir")
        files = _find_dataset_files(path_text)
        exists = bool(files) if kind == "glob" else Path(expanded_path).exists()
        sample_files = files[:sample_files_per_dataset]
        dataset_sample_files[dataset_name] = sample_files
        if not exists:
            warnings.append(f"Dataset path for {dataset_name!r} does not resolve to files: {path_text}")
        dataset_entries.append(
            {
                "name": str(dataset_name),
                "path": path_text,
                "path_kind": kind,
                "exists": bool(exists),
                "sample_files": sample_files,
            }
        )

    referenced_datasets = set(training_section.keys())
    for context in zeroshot_section.keys():
        dataset_name = str(context).split(".", 1)[0]
        referenced_datasets.add(dataset_name)
    for context in fewshot_section.keys():
        dataset_name = str(context).split(".", 1)[0]
        referenced_datasets.add(dataset_name)
    missing_dataset_paths = sorted(referenced_datasets - set(datasets_section.keys()))
    if missing_dataset_paths:
        errors.append(
            "Datasets referenced in training/splits but missing from `[datasets]`: "
            + ", ".join(missing_dataset_paths)
        )

    zeroshot_entries: list[dict[str, Any]] = []
    for context, split in sorted(zeroshot_section.items()):
        split_str = str(split)
        if split_str not in {"val", "test", "train"}:
            warnings.append(
                f"Unexpected zeroshot split value for {context!r}: {split_str!r} (expected 'val' or 'test')."
            )
        zeroshot_entries.append({"context": str(context), "split": split_str})

    fewshot_entries: list[dict[str, Any]] = []
    for context, split_cfg in sorted(fewshot_section.items()):
        if not isinstance(split_cfg, dict):
            warnings.append(f"Fewshot context {context!r} is not a mapping; skipping split counts.")
            fewshot_entries.append({"context": str(context), "val_count": 0, "test_count": 0})
            continue
        val_list = split_cfg.get("val", [])
        test_list = split_cfg.get("test", [])
        if not isinstance(val_list, list):
            warnings.append(f"Fewshot context {context!r} has non-list `val`; treating as empty.")
            val_list = []
        if not isinstance(test_list, list):
            warnings.append(f"Fewshot context {context!r} has non-list `test`; treating as empty.")
            test_list = []
        fewshot_entries.append(
            {
                "context": str(context),
                "val_count": int(len(val_list)),
                "test_count": int(len(test_list)),
            }
        )

    schema_from_samples: dict[str, Any] = {
        "obs_common": [],
        "obsm_common": [],
        "dataset_diffs": [],
    }
    if inspect_sample_schemas:
        obs_sets: dict[str, set[str]] = {}
        obsm_sets: dict[str, set[str]] = {}
        for dataset_name, sample_files in dataset_sample_files.items():
            if not sample_files:
                continue
            try:
                schema = inspect_adata_schema(sample_files[0], max_top_values=5)
            except Exception as exc:
                warnings.append(
                    f"Failed to inspect sample schema for dataset {dataset_name!r} ({sample_files[0]}): "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            obs_sets[dataset_name] = {str(col.get("name")) for col in schema.get("obs_columns", [])}
            obsm_sets[dataset_name] = {str(item.get("key")) for item in schema.get("obsm", [])}

        if obs_sets:
            obs_common = set.intersection(*obs_sets.values())
            schema_from_samples["obs_common"] = sorted(obs_common)
            obs_union = set.union(*obs_sets.values())
            for dataset_name, columns in sorted(obs_sets.items()):
                missing = sorted(obs_union - columns)
                if missing:
                    schema_from_samples["dataset_diffs"].append(
                        f"{dataset_name}: missing obs columns seen elsewhere: {', '.join(missing[:10])}"
                    )
        if obsm_sets:
            obsm_common = set.intersection(*obsm_sets.values())
            schema_from_samples["obsm_common"] = sorted(obsm_common)
            obsm_union = set.union(*obsm_sets.values())
            for dataset_name, keys in sorted(obsm_sets.items()):
                missing = sorted(obsm_union - keys)
                if missing:
                    schema_from_samples["dataset_diffs"].append(
                        f"{dataset_name}: missing obsm keys seen elsewhere: {', '.join(missing[:10])}"
                    )

    return {
        "toml_path": resolved,
        "datasets": dataset_entries,
        "training": {str(k): str(v) for k, v in training_section.items()},
        "zeroshot": zeroshot_entries,
        "fewshot": fewshot_entries,
        "schema_from_samples": schema_from_samples,
        "errors": errors,
        "warnings": warnings,
    }
