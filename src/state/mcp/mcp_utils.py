from __future__ import annotations

import glob
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
