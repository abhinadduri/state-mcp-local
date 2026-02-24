from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


def _resolve_backend_mode(backend: str) -> tuple[str, str]:
    mode = str(backend or "auto").strip().lower()
    if mode not in {"auto", "local", "slurm"}:
        raise ValueError("`backend` must be one of: 'auto', 'local', 'slurm'.")

    if mode == "local":
        return "local", "Explicit backend override."
    if mode == "slurm":
        if shutil.which("sbatch") is None:
            raise ValueError("`backend='slurm'` requested, but `sbatch` was not found in PATH.")
        return "slurm", "Explicit backend override."

    if shutil.which("sbatch") is not None:
        return "slurm", "Detected `sbatch` in PATH; using slurm backend."
    return "local", "No `sbatch` detected; using local backend."


def _resolve_slurm_backend_profile_args(
    backend_profile: str | None,
    env_prefix: str = "STATE_MCP_SLURM_PROFILE_",
) -> tuple[list[str], str]:
    if backend_profile is None or not str(backend_profile).strip():
        return [], "No slurm profile arguments provided."

    profile_text = str(backend_profile).strip()
    env_key = env_prefix + "".join(
        char.upper() if char.isalnum() else "_" for char in profile_text
    )
    env_value = os.getenv(env_key)
    if isinstance(env_value, str) and env_value.strip():
        return shlex.split(env_value), f"Loaded slurm args from ${env_key}."

    if profile_text.startswith("-") or " " in profile_text:
        return shlex.split(profile_text), "Parsed backend_profile as raw sbatch arguments."

    return [f"--partition={profile_text}"], "Interpreted backend_profile as a slurm partition name."


def _parse_sbatch_job_id(raw_stdout: str) -> str | None:
    if not isinstance(raw_stdout, str):
        return None
    for raw_line in raw_stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        token = line.split(";", 1)[0].strip().split()[0]
        if token:
            return token
    return None


def _submit_slurm_job(
    *,
    job_id: str,
    job_name_prefix: str,
    run_dir: str | None,
    command: list[str],
    backend_profile: str | None,
    env_prefix: str = "STATE_MCP_SLURM_PROFILE_",
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
    default_gpus: int | None = None,
) -> dict[str, Any]:
    if shutil.which("sbatch") is None:
        raise RuntimeError("`sbatch` was not found in PATH.")

    if run_dir is None or not str(run_dir).strip():
        run_dir_path = (Path("/tmp") / f"{job_name_prefix}_{job_id}").resolve()
    else:
        run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    profile_args, profile_reason = _resolve_slurm_backend_profile_args(backend_profile, env_prefix)

    resource_args: list[str] = []
    if slurm_partition is not None:
        resource_args.extend(["--partition", slurm_partition])
    effective_gpus = slurm_gpus if slurm_gpus is not None else default_gpus
    if effective_gpus is not None:
        resource_args.extend(["--gres", f"gpu:{effective_gpus}"])
    if slurm_cpus_per_task is not None:
        resource_args.extend(["--cpus-per-task", str(slurm_cpus_per_task)])
    if slurm_mem is not None:
        resource_args.extend(["--mem", slurm_mem])
    if slurm_time is not None:
        resource_args.extend(["--time", slurm_time])

    src_root = str(Path(__file__).resolve().parents[2])
    venv_python = Path(src_root) / ".venv" / "bin" / "python"
    if venv_python.is_file():
        python_exe = str(venv_python.resolve())
    else:
        python_exe = str(Path(sys.executable).resolve())
    full_cmd = [python_exe, *command]
    full_cmd_str = " ".join(shlex.quote(arg) for arg in full_cmd)
    wrapped_command = f"PYTHONPATH={shlex.quote(src_root)}:${{PYTHONPATH:-}} {full_cmd_str}"

    out_template = str(run_dir_path / "slurm-%j.out")
    err_template = str(run_dir_path / "slurm-%j.err")
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"{job_name_prefix}_{job_id[:8]}",
        "--chdir",
        str(run_dir_path),
        "--output",
        out_template,
        "--error",
        err_template,
        *resource_args,
        *profile_args,
        "--wrap",
        wrapped_command,
    ]

    result = subprocess.run(
        sbatch_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch submission failed with exit code {result.returncode}. stdout={stdout!r} stderr={stderr!r}"
        )

    scheduler_job_id = _parse_sbatch_job_id(stdout)
    if scheduler_job_id is None:
        raise RuntimeError(f"Unable to parse slurm job id from sbatch output: {stdout!r}")

    return {
        "scheduler_job_id": scheduler_job_id,
        "worker_log_path": str(run_dir_path / f"slurm-{scheduler_job_id}.out"),
        "worker_error_log_path": str(run_dir_path / f"slurm-{scheduler_job_id}.err"),
        "submission_command": sbatch_cmd,
        "profile_reason": profile_reason,
    }


def _submit_tx_train_slurm_job(
    *,
    job_id: str,
    run_dir: str | None,
    hydra_overrides: list[str],
    backend_profile: str | None,
    slurm_partition: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    slurm_time: str | None = None,
) -> dict[str, Any]:
    command = ["-m", "state", "tx", "train", *hydra_overrides]
    return _submit_slurm_job(
        job_id=job_id,
        job_name_prefix="state_tx_train",
        run_dir=run_dir,
        command=command,
        backend_profile=backend_profile,
        env_prefix="STATE_MCP_TX_TRAIN_SLURM_PROFILE_",
        slurm_partition=slurm_partition,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_mem=slurm_mem,
        slurm_time=slurm_time,
    )


def _query_slurm_state(scheduler_job_id: str) -> str | None:
    if not scheduler_job_id:
        return None

    if shutil.which("sacct") is not None:
        try:
            result = subprocess.run(
                ["sacct", "-j", scheduler_job_id, "--format=State", "-n", "-P"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and isinstance(result.stdout, str) and result.stdout.strip():
                for raw_line in result.stdout.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    state = line.split("|", 1)[0].strip()
                    if not state:
                        continue
                    return state.split()[0].strip().upper()
        except Exception:
            pass

    if shutil.which("squeue") is not None:
        try:
            result = subprocess.run(
                ["squeue", "-h", "-j", scheduler_job_id, "-o", "%T"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and isinstance(result.stdout, str) and result.stdout.strip():
                return result.stdout.strip().splitlines()[0].strip().upper()
        except Exception:
            pass

    return None


def _map_slurm_state_to_job_status(state: str) -> str:
    normalized = str(state or "").strip().upper()
    if normalized in {
        "PENDING",
        "CONFIGURING",
        "RESIZING",
        "SUSPENDED",
        "REQUEUE_FED",
        "REQUEUED",
    }:
        return "queued"
    if normalized in {"RUNNING", "COMPLETING", "STAGE_OUT"}:
        return "running"
    if normalized in {"COMPLETED"}:
        return "succeeded"
    if normalized in {"CANCELLED", "PREEMPTED"}:
        return "cancelled"
    if normalized in {
        "FAILED",
        "BOOT_FAIL",
        "DEADLINE",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
    }:
        return "failed"
    return "running"


def _build_tx_infer_cli_args(infer_args: Namespace) -> list[str]:
    args: list[str] = ["-m", "state", "tx", "infer"]
    args.extend(["--checkpoint", str(infer_args.checkpoint)])
    args.extend(["--adata", str(infer_args.adata)])
    args.extend(["--model-dir", str(infer_args.model_dir)])
    args.extend(["--output", str(infer_args.output)])
    args.extend(["--pert-col", str(infer_args.pert_col)])
    if infer_args.embed_key is not None:
        args.extend(["--embed-key", str(infer_args.embed_key)])
    if infer_args.celltype_col is not None:
        args.extend(["--celltype-col", str(infer_args.celltype_col)])
    if infer_args.celltypes is not None:
        args.extend(["--celltypes", str(infer_args.celltypes)])
    if infer_args.batch_col is not None:
        args.extend(["--batch-col", str(infer_args.batch_col)])
    if infer_args.control_pert is not None:
        args.extend(["--control-pert", str(infer_args.control_pert)])
    args.extend(["--seed", str(infer_args.seed)])
    if infer_args.max_set_len is not None:
        args.extend(["--max-set-len", str(infer_args.max_set_len)])
    if getattr(infer_args, "quiet", False):
        args.append("--quiet")
    if infer_args.tsv is not None:
        args.extend(["--tsv", str(infer_args.tsv)])
    if getattr(infer_args, "all_perts", False):
        args.append("--all-perts")
    if infer_args.virtual_cells_per_pert is not None:
        args.extend(["--virtual-cells-per-pert", str(infer_args.virtual_cells_per_pert)])
    if infer_args.min_cells is not None:
        args.extend(["--min-cells", str(infer_args.min_cells)])
    if infer_args.max_cells is not None:
        args.extend(["--max-cells", str(infer_args.max_cells)])
    if getattr(infer_args, "batched", True):
        args.append("--batched")
    else:
        args.append("--no-batched")
    if infer_args.set_batch_size is not None:
        args.extend(["--set-batch-size", str(infer_args.set_batch_size)])
    return args


def _build_emb_transform_cli_args(resolved: dict[str, Any]) -> list[str]:
    args: list[str] = ["-m", "state", "emb", "transform"]
    args.extend(["--input", str(resolved["input_adata_path"])])
    args.extend(["--output", str(resolved["output_path"])])
    args.extend(["--checkpoint", str(resolved["checkpoint_path"])])
    if resolved.get("model_folder") is not None:
        args.extend(["--model-folder", str(resolved["model_folder"])])
    if resolved.get("config_path") is not None:
        args.extend(["--config", str(resolved["config_path"])])
    args.extend(["--embed-key", str(resolved["embedding_key"])])
    if resolved.get("protein_embeddings_path") is not None:
        args.extend(["--protein-embeddings", str(resolved["protein_embeddings_path"])])
    if resolved.get("batch_size") is not None:
        args.extend(["--batch-size", str(resolved["batch_size"])])
    return args
