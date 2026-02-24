from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .mcp_utils import (
    infer_tx_inference_defaults,
    inspect_emb_checkpoint as inspect_emb_checkpoint_file,
    inspect_model_folder,
    inspect_tx_checkpoint as inspect_tx_checkpoint_file,
    normalize_and_validate_emb_inference_dir,
    normalize_and_validate_run_dir,
    resolve_emb_checkpoint_path,
    resolve_and_validate_model_folder,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time dependency guidance
    raise ImportError(
        "MCP support requires the `mcp` package. Install it in this environment to run `state.mcp`."
    ) from exc


mcp = FastMCP("state")

# Session-local defaults for this MCP server process.
_SERVER_STATE: dict[str, str | None] = {"tx_model_folder": None}

_TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
_JOB_LOG_LIMIT = 5000
_MAX_EVENT_DRAIN = 2000
_HEARTBEAT_INTERVAL_SECONDS = 10.0
_CANCEL_GRACE_SECONDS = 30.0
_TERMINATE_GRACE_SECONDS = 10.0


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _get_worker_mp_context() -> mp.context.BaseContext:
    # Prefer fork on POSIX to avoid `spawn` import/path edge cases in tool-hosted environments.
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


@dataclass
class InferenceJob:
    job_id: str
    status: str
    created_at: str
    model_folder: str
    adata_path: str
    output_path: str
    checkpoint_path: str
    resolved_args: dict[str, Any]
    inference_kind: str = "tx"
    progress: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    cancel_requested_at: str | None = None
    cancel_grace_deadline_monotonic: float | None = None
    terminate_sent_at_monotonic: float | None = None
    last_event: dict[str, Any] | None = None
    last_update_at: str = field(default_factory=lambda: _utc_now_iso())
    last_update_monotonic: float = field(default_factory=time.monotonic)
    adata_size_bytes: int | None = None
    worker_pid: int | None = None
    worker_exit_code: int | None = None
    worker_log_path: str | None = None
    process: mp.Process | None = None
    cancel_flag_path: str | None = None
    event_conn: Any | None = None


_JOBS: dict[str, InferenceJob] = {}
_JOBS_LOCK = threading.Lock()


class EmbInferenceCancelledError(RuntimeError):
    pass


def _get_job_locked(job_id: str, *, expected_kind: str | None = None) -> InferenceJob:
    job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f"No inference job found for job_id={job_id!r}")
    if expected_kind is not None and job.inference_kind != expected_kind:
        raise KeyError(f"Job {job_id!r} is `{job.inference_kind}` inference, expected `{expected_kind}`.")
    return job


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch_job(job: InferenceJob) -> None:
    job.last_update_at = _utc_now_iso()
    job.last_update_monotonic = time.monotonic()


def _append_job_log(job: InferenceJob, message: str) -> None:
    _touch_job(job)
    line = f"{_utc_now_iso()} {message}"
    job.logs.append(line)
    if len(job.logs) > _JOB_LOG_LIMIT:
        del job.logs[: len(job.logs) - _JOB_LOG_LIMIT]


def _event_progress_percent(event: dict[str, Any]) -> float | None:
    progress_value = event.get("progress")
    total_value = event.get("total")
    if isinstance(progress_value, (int, float)) and isinstance(total_value, (int, float)) and total_value > 0:
        percent = (float(progress_value) / float(total_value)) * 100.0
        return max(0.0, min(100.0, percent))

    done = event.get("perturbations_done")
    total = event.get("perturbations_total")
    if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
        percent = (float(done) / float(total)) * 100.0
        return max(0.0, min(100.0, percent))

    done = event.get("cells_done")
    total = event.get("cells_total")
    if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
        percent = (float(done) / float(total)) * 100.0
        return max(0.0, min(100.0, percent))

    return None


def _record_job_event(job: InferenceJob, event: dict[str, Any]) -> None:
    _touch_job(job)
    job.last_event = dict(event)
    progress = {
        "kind": event.get("kind"),
        "phase": event.get("phase"),
        "group": event.get("group"),
        "groups_total": event.get("groups_total"),
        "group_index": event.get("group_index"),
        "group_perturbations_done": event.get("group_perturbations_done"),
        "group_perturbations_total": event.get("group_perturbations_total"),
        "perturbations_done": event.get("perturbations_done"),
        "perturbations_total": event.get("perturbations_total"),
        "cells_done": event.get("cells_done"),
        "cells_total": event.get("cells_total"),
        "internal_padding_tokens": event.get("internal_padding_tokens"),
        "message": event.get("message"),
    }
    percent = _event_progress_percent(event)
    if percent is not None:
        progress["percent"] = percent
    job.progress = progress

    message = event.get("message")
    if not isinstance(message, str) or not message.strip():
        return

    kind = str(event.get("kind") or "event")
    should_log = True
    if kind == "progress":
        done = event.get("perturbations_done")
        total = event.get("perturbations_total")
        if isinstance(done, int) and isinstance(total, int) and total > 0:
            should_log = done in {1, total} or done % 25 == 0

    if should_log:
        _append_job_log(job, f"[{kind}] {message}")


def _process_alive(process: mp.Process | None) -> bool:
    if process is None:
        return False
    try:
        return process.is_alive()
    except Exception:
        return False


def _signal_name_for_exit_code(exit_code: int) -> str | None:
    if exit_code >= 0:
        return None
    signal_number = -exit_code
    try:
        return signal.Signals(signal_number).name
    except Exception:
        return None


def _release_job_runtime_resources(job: InferenceJob) -> None:
    process = job.process
    if process is not None:
        try:
            if process.exitcode is not None:
                job.worker_exit_code = int(process.exitcode)
        except Exception:
            pass
        try:
            process.join(timeout=0)
        except Exception:
            pass
    event_conn = job.event_conn
    if event_conn is not None:
        try:
            event_conn.close()
        except Exception:
            pass
    cancel_flag_path = job.cancel_flag_path
    if isinstance(cancel_flag_path, str) and cancel_flag_path:
        try:
            Path(cancel_flag_path).unlink(missing_ok=True)
        except Exception:
            pass
    job.process = None
    job.event_conn = None
    job.cancel_flag_path = None


def _apply_worker_message(job: InferenceJob, message: dict[str, Any]) -> None:
    if not isinstance(message, dict):
        return

    msg_type = message.get("type")
    if msg_type == "event":
        event = message.get("event")
        if isinstance(event, dict):
            _record_job_event(job, event)
        return

    if msg_type == "heartbeat":
        _touch_job(job)
        current_progress = dict(job.progress)
        current_progress["worker_heartbeat_at"] = job.last_update_at
        job.progress = current_progress
        return

    if msg_type == "status":
        _touch_job(job)
        status_value = message.get("status")
        if isinstance(status_value, str) and status_value:
            if not (status_value == "running" and job.cancel_requested):
                job.status = status_value
        started_at = message.get("started_at")
        if isinstance(started_at, str) and started_at:
            job.started_at = started_at
        finished_at = message.get("finished_at")
        if isinstance(finished_at, str) and finished_at:
            job.finished_at = finished_at
        error = message.get("error")
        if isinstance(error, str) and error.strip():
            job.error = error
        note = message.get("message")
        if isinstance(note, str) and note.strip():
            _append_job_log(job, note)
        return

    if msg_type == "traceback":
        trace = message.get("traceback")
        if isinstance(trace, str) and trace.strip():
            _append_job_log(job, trace.strip())
        return


def _drain_job_events_locked(job: InferenceJob) -> None:
    event_conn = job.event_conn
    if event_conn is None:
        return

    drained = 0
    while drained < _MAX_EVENT_DRAIN:
        try:
            if not event_conn.poll(0):
                break
            message = event_conn.recv()
        except (EOFError, OSError, ValueError):
            break
        drained += 1
        _apply_worker_message(job, message)

    if drained >= _MAX_EVENT_DRAIN:
        _append_job_log(job, f"[warning] Drained {_MAX_EVENT_DRAIN} worker events; additional events remain queued.")


def _recommend_poll_interval_seconds(job: InferenceJob, *, for_logs: bool = False) -> float | None:
    if job.status in _TERMINAL_JOB_STATUSES:
        return None

    if job.status == "queued":
        interval = 3.0
    elif job.status == "cancelling":
        interval = 4.0
    else:
        phase = str(job.progress.get("phase") or "")
        kind = str(job.progress.get("kind") or "")
        if kind == "progress":
            interval = 3.0
        elif phase in {"initializing", "config_loaded", "model_loaded", "adata_loaded"}:
            interval = 15.0
        else:
            interval = 8.0

    adata_size = int(job.adata_size_bytes or 0)
    if adata_size >= 10 * 1024**3:
        interval = max(interval, 25.0)
    elif adata_size >= 2 * 1024**3:
        interval = max(interval, 12.0)

    seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
    if job.status == "running" and seconds_since_update >= 120.0:
        interval = max(interval, 20.0)

    if for_logs:
        interval = max(interval * 1.5, 6.0)

    return round(interval, 1)


def _sync_job_state_locked(job: InferenceJob) -> None:
    _drain_job_events_locked(job)

    process = job.process
    if process is None:
        return

    if job.worker_pid is None and process.pid is not None:
        job.worker_pid = int(process.pid)

    now = time.monotonic()
    if job.cancel_requested and _process_alive(process):
        deadline = job.cancel_grace_deadline_monotonic
        if deadline is not None and now >= deadline and job.terminate_sent_at_monotonic is None:
            try:
                process.terminate()
                job.terminate_sent_at_monotonic = now
                _append_job_log(job, "[cancelling] Cancellation grace elapsed; sent SIGTERM to worker process.")
            except Exception as exc:
                _append_job_log(job, f"[warning] Failed to terminate worker process cleanly: {type(exc).__name__}: {exc}")
        elif (
            job.terminate_sent_at_monotonic is not None
            and now >= (job.terminate_sent_at_monotonic + _TERMINATE_GRACE_SECONDS)
            and _process_alive(process)
        ):
            try:
                process.kill()
                job.terminate_sent_at_monotonic = now
                _append_job_log(job, "[cancelling] Worker did not exit after SIGTERM; sent SIGKILL.")
            except Exception as exc:
                _append_job_log(job, f"[warning] Failed to force-kill worker process: {type(exc).__name__}: {exc}")

    if _process_alive(process):
        return

    try:
        exit_code = process.exitcode
    except Exception:
        exit_code = None
    if isinstance(exit_code, int):
        job.worker_exit_code = exit_code

    if job.status not in _TERMINAL_JOB_STATUSES:
        if job.cancel_requested:
            job.status = "cancelled"
            if not job.error:
                if isinstance(exit_code, int) and exit_code < 0:
                    signal_name = _signal_name_for_exit_code(exit_code)
                    if signal_name:
                        job.error = f"Inference cancelled by request (worker terminated by {signal_name})."
                    else:
                        job.error = f"Inference cancelled by request (worker exit code {exit_code})."
                else:
                    job.error = "Inference cancelled by request."
            _append_job_log(job, "[cancelled] Inference cancelled.")
        elif exit_code == 0:
            job.status = "succeeded"
            _append_job_log(job, "[succeeded] Inference job completed.")
        else:
            job.status = "failed"
            if not job.error:
                job.error = f"Worker process exited unexpectedly with code {exit_code}."
            _append_job_log(job, f"[failed] {job.error}")

    if job.finished_at is None:
        job.finished_at = _utc_now_iso()

    _release_job_runtime_resources(job)


def _resolve_tx_inference_request(
    *,
    adata_path: str,
    output_path: str | None,
    model_folder: str | None,
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

    resolved_model_folder = resolve_and_validate_model_folder(model_folder, _SERVER_STATE["tx_model_folder"])
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
        prefer_ckpt=True,
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


def _emit_worker_message(event_conn: Any, payload: dict[str, Any]) -> None:
    try:
        event_conn.send(payload)
    except Exception:
        return


def _run_inference_job_worker(infer_args: Namespace, cancel_flag_path: str, event_conn: Any, worker_log_path: str) -> None:
    from .._cli._tx._infer import InferenceCancelledError, run_tx_infer

    log_handle = None
    if worker_log_path:
        try:
            Path(worker_log_path).parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(worker_log_path, "a", encoding="utf-8", buffering=1)
            os.dup2(log_handle.fileno(), 1)
            os.dup2(log_handle.fileno(), 2)
            sys.stdout = log_handle
            sys.stderr = log_handle
        except Exception:
            log_handle = None

    heartbeat_stop = threading.Event()

    def emit(payload: dict[str, Any]) -> None:
        _emit_worker_message(event_conn, payload)

    def heartbeat_loop() -> None:
        while not heartbeat_stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            emit({"type": "heartbeat"})

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        daemon=True,
        name="state_tx_worker_heartbeat",
    )
    heartbeat_thread.start()

    def cancel_check() -> bool:
        try:
            return Path(cancel_flag_path).exists()
        except Exception:
            return False

    def progress_callback(event: dict[str, Any]) -> None:
        emit({"type": "event", "event": dict(event)})

    setattr(infer_args, "cancel_check", cancel_check)
    setattr(infer_args, "progress_callback", progress_callback)
    emit(
        {
            "type": "status",
            "status": "running",
            "started_at": _utc_now_iso(),
            "message": "Inference job started.",
        }
    )

    try:
        run_tx_infer(infer_args)
    except InferenceCancelledError as exc:
        emit(
            {
                "type": "status",
                "status": "cancelled",
                "finished_at": _utc_now_iso(),
                "error": str(exc),
                "message": f"[cancelled] {exc}",
            }
        )
    except BaseException as exc:  # pragma: no cover - defensive guard around worker process.
        emit(
            {
                "type": "status",
                "status": "failed",
                "finished_at": _utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "message": f"[failed] {type(exc).__name__}: {exc}",
            }
        )
        emit({"type": "traceback", "traceback": traceback.format_exc().strip()})
    else:
        emit(
            {
                "type": "status",
                "status": "succeeded",
                "finished_at": _utc_now_iso(),
                "message": "[succeeded] Inference job completed.",
            }
        )
    finally:
        heartbeat_stop.set()
        try:
            event_conn.close()
        except Exception:
            pass
        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass


def _run_emb_inference_job_worker(
    resolved: dict[str, Any],
    cancel_flag_path: str,
    event_conn: Any,
    worker_log_path: str,
) -> None:
    log_handle = None
    if worker_log_path:
        try:
            Path(worker_log_path).parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(worker_log_path, "a", encoding="utf-8", buffering=1)
            os.dup2(log_handle.fileno(), 1)
            os.dup2(log_handle.fileno(), 2)
            sys.stdout = log_handle
            sys.stderr = log_handle
        except Exception:
            log_handle = None

    heartbeat_stop = threading.Event()

    def emit(payload: dict[str, Any]) -> None:
        _emit_worker_message(event_conn, payload)

    def heartbeat_loop() -> None:
        while not heartbeat_stop.wait(_HEARTBEAT_INTERVAL_SECONDS):
            emit({"type": "heartbeat"})

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        daemon=True,
        name="state_emb_worker_heartbeat",
    )
    heartbeat_thread.start()

    def cancellation_requested() -> bool:
        try:
            return Path(cancel_flag_path).exists()
        except Exception:
            return False

    def ensure_not_cancelled(phase: str) -> None:
        if cancellation_requested():
            raise EmbInferenceCancelledError(f"Embedding inference cancelled during {phase}.")

    emit(
        {
            "type": "status",
            "status": "running",
            "started_at": _utc_now_iso(),
            "message": "Embedding inference job started.",
        }
    )

    try:
        output_path_resolved = resolved.get("output_path")
        if isinstance(output_path_resolved, str):
            Path(output_path_resolved).parent.mkdir(parents=True, exist_ok=True)

        emit(
            {
                "type": "event",
                "event": {
                    "kind": "progress",
                    "phase": "initializing",
                    "progress": 0,
                    "total": 100,
                    "message": "Preparing embedding inference.",
                },
            }
        )

        from ..emb.inference import Inference

        cfg_override = None
        if isinstance(resolved.get("config_path"), str):
            from omegaconf import OmegaConf

            cfg_override = OmegaConf.load(resolved["config_path"])

        protein_embeddings_override = None
        if isinstance(resolved.get("protein_embeddings_path"), str):
            import torch

            protein_embeddings_override = torch.load(
                resolved["protein_embeddings_path"],
                map_location="cpu",
                weights_only=False,
            )

        ensure_not_cancelled("initialization")

        inferer = Inference(cfg=cfg_override, protein_embeds=protein_embeddings_override)
        emit(
            {
                "type": "event",
                "event": {
                    "kind": "progress",
                    "phase": "initializing",
                    "progress": 10,
                    "total": 100,
                    "message": "Loading embedding checkpoint.",
                },
            }
        )
        inferer.load_model(resolved["checkpoint_path"])

        ensure_not_cancelled("checkpoint loading")

        emit(
            {
                "type": "event",
                "event": {
                    "kind": "progress",
                    "phase": "model_loaded",
                    "progress": 30,
                    "total": 100,
                    "message": "Running embedding forward pass.",
                },
            }
        )
        embeddings = inferer.encode_adata(
            input_adata_path=resolved["input_adata_path"],
            output_adata_path=resolved["output_adata_path_for_encode"],
            emb_key=resolved["embedding_key"],
            batch_size=resolved["batch_size"],
        )

        ensure_not_cancelled("embedding forward pass")

        if isinstance(output_path_resolved, str) and output_path_resolved.lower().endswith(".npy"):
            import numpy as np

            emit(
                {
                    "type": "event",
                    "event": {
                        "kind": "progress",
                        "phase": "writing_output",
                        "progress": 90,
                        "total": 100,
                        "message": "Writing embeddings to .npy output.",
                    },
                }
            )
            np.save(output_path_resolved, embeddings)

        emb_shape: list[int] | None = None
        emb_dtype: str | None = None
        if hasattr(embeddings, "shape"):
            try:
                emb_shape = [int(x) for x in embeddings.shape]
            except Exception:
                emb_shape = None
        if hasattr(embeddings, "dtype"):
            try:
                emb_dtype = str(embeddings.dtype)
            except Exception:
                emb_dtype = None

        emit(
            {
                "type": "event",
                "event": {
                    "kind": "progress",
                    "phase": "completed",
                    "progress": 100,
                    "total": 100,
                    "embeddings_shape": emb_shape,
                    "embeddings_dtype": emb_dtype,
                    "message": "Embedding inference completed.",
                },
            }
        )
        emit(
            {
                "type": "status",
                "status": "succeeded",
                "finished_at": _utc_now_iso(),
                "message": "[succeeded] Embedding inference completed.",
            }
        )
    except EmbInferenceCancelledError as exc:
        emit(
            {
                "type": "status",
                "status": "cancelled",
                "finished_at": _utc_now_iso(),
                "error": str(exc),
                "message": f"[cancelled] {exc}",
            }
        )
    except BaseException as exc:  # pragma: no cover - defensive guard around worker process.
        emit(
            {
                "type": "status",
                "status": "failed",
                "finished_at": _utc_now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "message": f"[failed] {type(exc).__name__}: {exc}",
            }
        )
        emit({"type": "traceback", "traceback": traceback.format_exc().strip()})
    finally:
        heartbeat_stop.set()
        try:
            event_conn.close()
        except Exception:
            pass
        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass


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
    _SERVER_STATE["tx_model_folder"] = resolved
    return {"status": "ok", "tx_model_folder": resolved}


@mcp.tool()
def get_tx_model_folder() -> dict[str, str | None]:
    """
    Return the current default STATE TX model run folder for this MCP server session.
    """
    return {"tx_model_folder": _SERVER_STATE["tx_model_folder"]}


@mcp.tool()
def clear_tx_model_folder() -> dict[str, str]:
    """
    Clear the current default STATE TX model run folder for this MCP server session.
    """
    _SERVER_STATE["tx_model_folder"] = None
    return {"status": "cleared"}


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
    return inspect_tx_checkpoint_file(
        checkpoint_path=checkpoint_path,
        model_folder=model_folder,
        default_model_folder=_SERVER_STATE["tx_model_folder"],
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

    candidate = model_folder or _SERVER_STATE["tx_model_folder"]
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
def run_emb_inference(
    input_adata_path: str,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    model_folder: str | None = None,
    config_path: str | None = None,
    embedding_key: str = "X_state",
    protein_embeddings_path: str | None = None,
    batch_size: int | None = None,
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
    )
    _append_job_log(job, "[queued] Embedding inference job queued.")

    mp_ctx = _get_worker_mp_context()
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    cancel_flag_path = str((Path("/tmp") / f"state_emb_cancel_{job_id}.flag").resolve())
    worker_log_path = str((Path("/tmp") / f"state_emb_worker_{job_id}.log").resolve())
    Path(cancel_flag_path).unlink(missing_ok=True)
    process = mp_ctx.Process(
        target=_run_emb_inference_job_worker,
        args=(resolved, cancel_flag_path, child_conn, worker_log_path),
        daemon=True,
        name=f"state_emb_infer_{job_id[:8]}",
    )

    job.event_conn = parent_conn
    job.cancel_flag_path = cancel_flag_path
    job.worker_log_path = worker_log_path
    job.process = process

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    try:
        process.start()
        try:
            child_conn.close()
        except Exception:
            pass
    except Exception as exc:
        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
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
        current = _JOBS.get(job_id)
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
        "recommended_initial_poll_interval_seconds": poll_interval,
    }


@mcp.tool()
def get_emb_inference_status(job_id: str) -> dict[str, Any]:
    """
    Return status, progress, and adaptive polling hints for a background emb inference job.
    """
    with _JOBS_LOCK:
        job = _get_job_locked(job_id, expected_kind="emb")
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

    with _JOBS_LOCK:
        job = _get_job_locked(job_id, expected_kind="emb")
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
    with _JOBS_LOCK:
        job = _get_job_locked(job_id, expected_kind="emb")
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

        if isinstance(job.cancel_flag_path, str) and job.cancel_flag_path:
            try:
                Path(job.cancel_flag_path).touch(exist_ok=True)
            except Exception as exc:
                _append_job_log(
                    job,
                    f"[warning] Failed to set cancellation flag at {job.cancel_flag_path}: {type(exc).__name__}: {exc}",
                )

        terminate_sent = False
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
            "cancel_requested": job.cancel_requested,
            "cancel_requested_at": job.cancel_requested_at,
            "force": force,
            "terminate_sent": terminate_sent,
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
) -> dict[str, Any]:
    """
    Start tx inference in the background and return a `job_id` immediately.
    Poll status with `get_tx_inference_status`, fetch logs with `get_tx_inference_logs`,
    and cancel with `cancel_tx_inference`.

    Returns poll-hint metadata (`recommended_initial_poll_interval_seconds`) for
    clients that want adaptive backoff on long-running jobs.
    """
    infer_args, resolved = _resolve_tx_inference_request(
        adata_path=adata_path,
        output_path=output_path,
        model_folder=model_folder,
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
    )
    _append_job_log(job, "[queued] Inference job queued.")

    mp_ctx = _get_worker_mp_context()
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    cancel_flag_path = str((Path("/tmp") / f"state_tx_cancel_{job_id}.flag").resolve())
    worker_log_path = str((Path("/tmp") / f"state_tx_worker_{job_id}.log").resolve())
    Path(cancel_flag_path).unlink(missing_ok=True)
    process = mp_ctx.Process(
        target=_run_inference_job_worker,
        args=(infer_args, cancel_flag_path, child_conn, worker_log_path),
        daemon=True,
        name=f"state_tx_infer_{job_id[:8]}",
    )

    job.event_conn = parent_conn
    job.cancel_flag_path = cancel_flag_path
    job.worker_log_path = worker_log_path
    job.process = process

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    try:
        process.start()
        try:
            child_conn.close()
        except Exception:
            pass
    except Exception as exc:
        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
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
        current = _JOBS.get(job_id)
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
        "recommended_initial_poll_interval_seconds": poll_interval,
    }


@mcp.tool()
def get_tx_inference_status(job_id: str) -> dict[str, Any]:
    """
    Return status, progress, and adaptive polling hints for a background tx inference job.
    """
    with _JOBS_LOCK:
        job = _get_job_locked(job_id, expected_kind="tx")
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

    with _JOBS_LOCK:
        job = _get_job_locked(job_id, expected_kind="tx")
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
    with _JOBS_LOCK:
        job = _get_job_locked(job_id, expected_kind="tx")
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

        if isinstance(job.cancel_flag_path, str) and job.cancel_flag_path:
            try:
                Path(job.cancel_flag_path).touch(exist_ok=True)
            except Exception as exc:
                _append_job_log(
                    job,
                    f"[warning] Failed to set cancellation flag at {job.cancel_flag_path}: {type(exc).__name__}: {exc}",
                )

        terminate_sent = False
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
