from __future__ import annotations

import csv
import multiprocessing as mp
import os
import shlex
import signal
import shutil
import subprocess
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

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time dependency guidance
    raise ImportError(
        "MCP support requires the `mcp` package. Install it in this environment to run `state.mcp`."
    ) from exc


mcp = FastMCP("state")

# Session-scoped state, partitioned by MCP client/session key.
_FALLBACK_SESSION_KEY = "__default__"
_SESSION_SERVER_STATE: dict[str, dict[str, str | None]] = {}
_SESSION_SERVER_STATE_LOCK = threading.Lock()

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


_JOBS: dict[str, dict[str, InferenceJob]] = {}
_JOBS_LOCK = threading.Lock()


@dataclass
class TrainJob:
    job_id: str
    status: str
    created_at: str
    run_dir: str | None
    backend: str
    backend_profile: str | None
    hydra_overrides: list[str]
    resolved_plan: dict[str, Any]
    idempotency_key: str | None = None
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
    worker_pid: int | None = None
    worker_exit_code: int | None = None
    worker_log_path: str | None = None
    worker_error_log_path: str | None = None
    worker_log_read_offset: int = 0
    scheduler_job_id: str | None = None
    process: mp.Process | None = None
    cancel_flag_path: str | None = None
    event_conn: Any | None = None


_TRAIN_JOBS: dict[str, dict[str, TrainJob]] = {}
_TRAIN_JOBS_LOCK = threading.Lock()
_TRAIN_IDEMPOTENCY: dict[str, dict[str, str]] = {}


class EmbInferenceCancelledError(RuntimeError):
    pass


class TxTrainCancelledError(RuntimeError):
    pass


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


def _get_session_server_state_locked(session_key: str) -> dict[str, str | None]:
    state = _SESSION_SERVER_STATE.get(session_key)
    if state is None:
        state = {"tx_model_folder": None}
        _SESSION_SERVER_STATE[session_key] = state
    return state


def _get_session_jobs_locked(session_key: str) -> dict[str, InferenceJob]:
    jobs = _JOBS.get(session_key)
    if jobs is None:
        jobs = {}
        _JOBS[session_key] = jobs
    return jobs


def _get_session_train_jobs_locked(session_key: str) -> dict[str, TrainJob]:
    jobs = _TRAIN_JOBS.get(session_key)
    if jobs is None:
        jobs = {}
        _TRAIN_JOBS[session_key] = jobs
    return jobs


def _get_session_train_idempotency_locked(session_key: str) -> dict[str, str]:
    mapping = _TRAIN_IDEMPOTENCY.get(session_key)
    if mapping is None:
        mapping = {}
        _TRAIN_IDEMPOTENCY[session_key] = mapping
    return mapping


def _get_job_locked(
    job_id: str,
    jobs_by_id: dict[str, InferenceJob],
    *,
    expected_kind: str | None = None,
) -> InferenceJob:
    job = jobs_by_id.get(job_id)
    if job is None:
        raise KeyError(f"No inference job found for job_id={job_id!r}")
    if expected_kind is not None and job.inference_kind != expected_kind:
        raise KeyError(f"Job {job_id!r} is `{job.inference_kind}` inference, expected `{expected_kind}`.")
    return job


def _get_train_job_locked(job_id: str, jobs_by_id: dict[str, TrainJob]) -> TrainJob:
    job = jobs_by_id.get(job_id)
    if job is None:
        raise KeyError(f"No training job found for job_id={job_id!r}")
    return job


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _touch_job(job: InferenceJob) -> None:
    job.last_update_at = _utc_now_iso()
    job.last_update_monotonic = time.monotonic()


def _append_job_log(job: InferenceJob, message: str) -> None:
    _touch_job(job)
    line = f"{_utc_now_iso()} {message}"
    job.logs.append(line)
    if len(job.logs) > _JOB_LOG_LIMIT:
        del job.logs[: len(job.logs) - _JOB_LOG_LIMIT]


def _touch_train_job(job: TrainJob) -> None:
    job.last_update_at = _utc_now_iso()
    job.last_update_monotonic = time.monotonic()


def _append_train_job_log(job: TrainJob, message: str) -> None:
    _touch_train_job(job)
    line = f"{_utc_now_iso()} {message}"
    job.logs.append(line)
    if len(job.logs) > _JOB_LOG_LIMIT:
        del job.logs[: len(job.logs) - _JOB_LOG_LIMIT]


def _ingest_train_worker_log_lines(job: TrainJob) -> None:
    log_path = job.worker_log_path
    if not isinstance(log_path, str) or not log_path.strip():
        return
    path_obj = Path(log_path)
    if not path_obj.is_file():
        return

    try:
        with path_obj.open("rb") as f:
            f.seek(job.worker_log_read_offset)
            chunk = f.read()
            job.worker_log_read_offset = f.tell()
    except Exception:
        return

    if not chunk:
        return
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        return
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        _append_train_job_log(job, f"[worker] {line}")


def _extract_override_value(overrides: list[str], key: str) -> str | None:
    needle = key.strip()
    for override in reversed(overrides):
        if "=" not in override:
            continue
        raw_key, raw_value = override.split("=", 1)
        normalized_key = raw_key.strip().lstrip("+~")
        if normalized_key == needle:
            return raw_value.strip()
    return None


def _extract_train_max_steps(job: TrainJob) -> int | None:
    max_steps_raw = _extract_override_value(job.hydra_overrides, "training.max_steps")
    if max_steps_raw is None:
        return None
    try:
        value = int(float(max_steps_raw))
    except Exception:
        return None
    if value <= 0:
        return None
    return value


def _extract_train_metrics_progress(job: TrainJob) -> dict[str, Any] | None:
    if not isinstance(job.run_dir, str) or not job.run_dir.strip():
        return None
    metrics_path = Path(job.run_dir) / "version_0" / "metrics.csv"
    if not metrics_path.is_file():
        return None

    last_row: dict[str, str] | None = None
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                if any((value is not None and str(value).strip()) for value in row.values()):
                    last_row = {str(k): str(v) for k, v in row.items() if k is not None}
    except Exception:
        return None

    if last_row is None:
        return None

    progress: dict[str, Any] = {"phase": "training"}
    step_raw = last_row.get("step")
    current_step: int | None = None
    if isinstance(step_raw, str) and step_raw.strip():
        try:
            current_step = int(float(step_raw))
        except Exception:
            current_step = None
    max_steps = _extract_train_max_steps(job)
    if current_step is not None:
        progress["current_step"] = current_step
    if max_steps is not None:
        progress["max_steps"] = max_steps
    if current_step is not None and max_steps is not None and max_steps > 0:
        percent = max(0.0, min(100.0, (float(current_step) / float(max_steps)) * 100.0))
        progress["percent"] = round(percent, 3)

    metric_priority = (
        "val/expression_loss",
        "val/embedding_loss",
        "train/expression_loss",
        "train/embedding_loss",
    )
    for metric_name in metric_priority:
        value_raw = last_row.get(metric_name)
        if not isinstance(value_raw, str) or not value_raw.strip():
            continue
        try:
            metric_value = float(value_raw)
        except Exception:
            continue
        progress["last_metric"] = {"name": metric_name, "value": metric_value}
        break
    return progress

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


def _record_train_job_event(job: TrainJob, event: dict[str, Any]) -> None:
    _touch_train_job(job)
    job.last_event = dict(event)

    progress = {
        "kind": event.get("kind"),
        "phase": event.get("phase"),
        "message": event.get("message"),
        "current_step": event.get("current_step"),
        "max_steps": event.get("max_steps"),
    }
    percent = _event_progress_percent(event)
    if percent is None:
        current_step = event.get("current_step")
        max_steps = event.get("max_steps")
        if isinstance(current_step, (int, float)) and isinstance(max_steps, (int, float)) and max_steps > 0:
            percent = (float(current_step) / float(max_steps)) * 100.0
    if percent is not None:
        progress["percent"] = max(0.0, min(100.0, float(percent)))
    job.progress = progress

    message = event.get("message")
    if isinstance(message, str) and message.strip():
        kind = str(event.get("kind") or "event")
        _append_train_job_log(job, f"[{kind}] {message}")


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


def _release_train_job_runtime_resources(job: TrainJob) -> None:
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


def _apply_train_worker_message(job: TrainJob, message: dict[str, Any]) -> None:
    if not isinstance(message, dict):
        return

    msg_type = message.get("type")
    if msg_type == "event":
        event = message.get("event")
        if isinstance(event, dict):
            _record_train_job_event(job, event)
        return

    if msg_type == "heartbeat":
        _touch_train_job(job)
        current_progress = dict(job.progress)
        current_progress["worker_heartbeat_at"] = job.last_update_at
        job.progress = current_progress
        return

    if msg_type == "status":
        _touch_train_job(job)
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
            _append_train_job_log(job, note)
        return

    if msg_type == "traceback":
        trace = message.get("traceback")
        if isinstance(trace, str) and trace.strip():
            _append_train_job_log(job, trace.strip())
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


def _drain_train_job_events_locked(job: TrainJob) -> None:
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
        _apply_train_worker_message(job, message)

    if drained >= _MAX_EVENT_DRAIN:
        _append_train_job_log(job, f"[warning] Drained {_MAX_EVENT_DRAIN} worker events; additional events remain queued.")


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


def _recommend_train_poll_interval_seconds(job: TrainJob, *, for_logs: bool = False) -> float | None:
    if job.status in _TERMINAL_JOB_STATUSES:
        return None

    if job.status == "queued":
        interval = 4.0
    elif job.status == "cancelling":
        interval = 4.0
    else:
        phase = str(job.progress.get("phase") or "")
        if phase in {"validating", "config_loaded", "trainer_initializing"}:
            interval = 10.0
        elif phase == "training":
            interval = 6.0
        else:
            interval = 8.0

    seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
    if job.status == "running" and seconds_since_update >= 180.0:
        interval = max(interval, 20.0)

    if for_logs:
        interval = max(interval * 1.5, 6.0)

    return round(interval, 1)


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


def _map_slurm_state_to_train_status(state: str) -> str:
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


def _sync_train_job_state_locked(job: TrainJob) -> None:
    _ingest_train_worker_log_lines(job)

    metrics_progress = _extract_train_metrics_progress(job)
    if isinstance(metrics_progress, dict):
        current_progress = dict(job.progress)
        current_progress.update(metrics_progress)
        job.progress = current_progress
        _touch_train_job(job)

    if job.backend == "slurm":
        if isinstance(job.scheduler_job_id, str) and job.scheduler_job_id.strip():
            state = _query_slurm_state(job.scheduler_job_id)
            if isinstance(state, str) and state:
                mapped = _map_slurm_state_to_train_status(state)
                current_progress = dict(job.progress)
                current_progress["scheduler_state"] = state
                job.progress = current_progress
                _touch_train_job(job)

                if mapped == "running" and job.started_at is None:
                    job.started_at = _utc_now_iso()

                if mapped in _TERMINAL_JOB_STATUSES:
                    if job.status not in _TERMINAL_JOB_STATUSES:
                        job.status = mapped
                    if mapped == "failed" and not job.error:
                        job.error = f"Slurm job ended in state {state}."
                    if mapped == "cancelled" and not job.error:
                        job.error = f"Slurm job cancelled (state={state})."
                    if job.finished_at is None:
                        job.finished_at = _utc_now_iso()
                elif job.status != "cancelling":
                    job.status = mapped
        return

    _drain_train_job_events_locked(job)

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
                _append_train_job_log(job, "[cancelling] Cancellation grace elapsed; sent SIGTERM to worker process.")
            except Exception as exc:
                _append_train_job_log(
                    job,
                    f"[warning] Failed to terminate worker process cleanly: {type(exc).__name__}: {exc}",
                )
        elif (
            job.terminate_sent_at_monotonic is not None
            and now >= (job.terminate_sent_at_monotonic + _TERMINATE_GRACE_SECONDS)
            and _process_alive(process)
        ):
            try:
                process.kill()
                job.terminate_sent_at_monotonic = now
                _append_train_job_log(job, "[cancelling] Worker did not exit after SIGTERM; sent SIGKILL.")
            except Exception as exc:
                _append_train_job_log(
                    job,
                    f"[warning] Failed to force-kill worker process: {type(exc).__name__}: {exc}",
                )

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
                        job.error = f"Training cancelled by request (worker terminated by {signal_name})."
                    else:
                        job.error = f"Training cancelled by request (worker exit code {exit_code})."
                else:
                    job.error = "Training cancelled by request."
            _append_train_job_log(job, "[cancelled] Training cancelled.")
        elif exit_code == 0:
            job.status = "succeeded"
            _append_train_job_log(job, "[succeeded] Training job completed.")
        else:
            job.status = "failed"
            if not job.error:
                job.error = f"Worker process exited unexpectedly with code {exit_code}."
            _append_train_job_log(job, f"[failed] {job.error}")

    if job.finished_at is None:
        job.finished_at = _utc_now_iso()

    _release_train_job_runtime_resources(job)


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
    if not model_preset_clean:
        validation_errors.append("`model_preset` must be a non-empty model config name.")

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
    if resolved_output_space is None:
        resolved_output_space = "gene" if resolved_embed_key in {None, "X_hvg"} else "all"
    if resolved_output_space not in {"gene", "all", "embedding"}:
        validation_errors.append(
            f"`output_space` must be one of 'gene', 'all', or 'embedding'; got {resolved_output_space!r}."
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
            "resolved_model": {
                "model_preset": model_preset_clean,
                "embed_key": resolved_embed_key,
                "output_space": resolved_output_space,
            },
            "resolved_training": {
                "max_steps": max_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "val_freq": val_freq,
                "seed": seed,
            },
        },
        "toml_inspection": toml_inspection,
    }


def _resolve_slurm_backend_profile_args(backend_profile: str | None) -> tuple[list[str], str]:
    if backend_profile is None or not str(backend_profile).strip():
        return [], "No slurm profile arguments provided."

    profile_text = str(backend_profile).strip()
    env_key = "STATE_MCP_TX_TRAIN_SLURM_PROFILE_" + "".join(
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


def _submit_tx_train_slurm_job(
    *,
    job_id: str,
    run_dir: str | None,
    hydra_overrides: list[str],
    backend_profile: str | None,
) -> dict[str, Any]:
    if shutil.which("sbatch") is None:
        raise RuntimeError("`sbatch` was not found in PATH.")

    if run_dir is None or not str(run_dir).strip():
        run_dir_path = (Path("/tmp") / f"state_tx_train_{job_id}").resolve()
    else:
        run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    profile_args, profile_reason = _resolve_slurm_backend_profile_args(backend_profile)

    python_exe = str(Path(sys.executable).resolve())
    src_root = str(Path(__file__).resolve().parents[2])
    train_cmd = [python_exe, "-m", "state", "tx", "train", *hydra_overrides]
    train_cmd_str = " ".join(shlex.quote(arg) for arg in train_cmd)
    wrapped_command = f"PYTHONPATH={shlex.quote(src_root)}:${{PYTHONPATH:-}} {train_cmd_str}"

    out_template = str(run_dir_path / "slurm-%j.out")
    err_template = str(run_dir_path / "slurm-%j.err")
    sbatch_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"state_tx_train_{job_id[:8]}",
        "--chdir",
        str(run_dir_path),
        "--output",
        out_template,
        "--error",
        err_template,
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


def _emit_worker_message(event_conn: Any, payload: dict[str, Any]) -> None:
    try:
        event_conn.send(payload)
    except Exception:
        return


def _run_tx_train_job_worker(
    hydra_overrides: list[str],
    cancel_flag_path: str,
    event_conn: Any,
    worker_log_path: str,
) -> None:
    from ..__main__ import load_hydra_config
    from .._cli._tx._train import run_tx_train

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
        name="state_tx_train_worker_heartbeat",
    )
    heartbeat_thread.start()

    def cancellation_requested() -> bool:
        try:
            return Path(cancel_flag_path).exists()
        except Exception:
            return False

    def ensure_not_cancelled(phase: str) -> None:
        if cancellation_requested():
            raise TxTrainCancelledError(f"Training cancelled during {phase}.")

    emit(
        {
            "type": "status",
            "status": "running",
            "started_at": _utc_now_iso(),
            "message": "Training job started.",
        }
    )

    try:
        emit(
            {
                "type": "event",
                "event": {
                    "kind": "progress",
                    "phase": "config_loading",
                    "current_step": 0,
                    "message": "Loading Hydra configuration.",
                },
            }
        )
        ensure_not_cancelled("configuration loading")
        cfg = load_hydra_config("tx", hydra_overrides)
        max_steps = None
        try:
            max_steps = int(cfg["training"]["max_steps"])
        except Exception:
            max_steps = None
        emit(
            {
                "type": "event",
                "event": {
                    "kind": "progress",
                    "phase": "trainer_initializing",
                    "current_step": 0,
                    "max_steps": max_steps,
                    "message": "Configuration loaded; preparing trainer.",
                },
            }
        )
        ensure_not_cancelled("trainer initialization")
        run_tx_train(cfg)
        emit(
            {
                "type": "status",
                "status": "succeeded",
                "finished_at": _utc_now_iso(),
                "message": "[succeeded] Training job completed.",
            }
        )
    except TxTrainCancelledError as exc:
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
    extra_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build and validate a curated TX training plan without launching training.

    Returns `status`:
    - `needs_input`: required fields are missing
    - `invalid`: values failed validation
    - `ready`: launch-ready plan with resolved Hydra overrides
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
    extra_overrides: list[str] | None = None,
    idempotency_key: str | None = None,
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
        extra_overrides=extra_overrides,
        idempotency_key=idempotency_key,
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
    extra_overrides: list[str] | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """
    Start TX training in the background and return a `job_id` immediately.

    Recommended workflow:
    1) call `plan_tx_train` until status is `ready`
    2) call `start_tx_train`
    3) poll via `get_tx_train_status` and `get_tx_train_logs`
    4) optionally cancel via `cancel_tx_train`
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
        try:
            submission = _submit_tx_train_slurm_job(
                job_id=job_id,
                run_dir=run_dir,
                hydra_overrides=resolved_overrides,
                backend_profile=profile_value,
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
        args=(resolved_overrides, cancel_flag_path, child_conn, worker_log_path),
        daemon=True,
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
    session_key = _get_current_session_key()

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
        jobs_by_id = _get_session_jobs_locked(session_key)
        jobs_by_id[job_id] = job

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
        jobs_by_id = _get_session_jobs_locked(session_key)
        jobs_by_id[job_id] = job

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
