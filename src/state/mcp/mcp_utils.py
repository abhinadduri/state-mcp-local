from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Final

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
