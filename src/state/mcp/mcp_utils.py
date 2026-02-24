from __future__ import annotations

import os
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
_EMB_CHECKPOINT_SUFFIXES: Final[tuple[str, ...]] = (".ckpt", ".safetensors")
_EMB_CHECKPOINT_BASENAME_PRIORITY: Final[tuple[str, ...]] = (
    "last.ckpt",
    "best.ckpt",
    "final.ckpt",
    "checkpoint.ckpt",
    "model.safetensors",
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
