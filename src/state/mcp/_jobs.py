from __future__ import annotations

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALLBACK_SESSION_KEY = "__default__"
_TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
_JOB_LOG_LIMIT = 5000
_MAX_EVENT_DRAIN = 2000
_HEARTBEAT_INTERVAL_SECONDS = 10.0
_CANCEL_GRACE_SECONDS = 30.0
_TERMINATE_GRACE_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_worker_mp_context() -> mp.context.BaseContext:
    # Prefer fork on POSIX to avoid `spawn` import/path edge cases in tool-hosted environments.
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
    worker_error_log_path: str | None = None
    worker_log_read_offset: int = 0
    backend: str = "local"
    backend_profile: str | None = None
    scheduler_job_id: str | None = None
    process: mp.Process | None = None
    cancel_flag_path: str | None = None
    event_conn: Any | None = None


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


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

class EmbInferenceCancelledError(RuntimeError):
    pass


class TxTrainCancelledError(RuntimeError):
    pass


class PreprocessCancelledError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# State containers + locks
# ---------------------------------------------------------------------------

_SESSION_SERVER_STATE: dict[str, dict[str, str | None]] = {}
_SESSION_SERVER_STATE_LOCK = threading.Lock()

_JOBS: dict[str, dict[str, InferenceJob]] = {}
_JOBS_LOCK = threading.Lock()

_TRAIN_JOBS: dict[str, dict[str, TrainJob]] = {}
_TRAIN_JOBS_LOCK = threading.Lock()
_TRAIN_IDEMPOTENCY: dict[str, dict[str, str]] = {}

_PREPROCESS_JOBS: dict[str, dict[str, "PreprocessJob"]] = {}
_PREPROCESS_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PreprocessJob dataclass + helpers
# ---------------------------------------------------------------------------

@dataclass
class PreprocessJob:
    job_id: str
    status: str
    created_at: str
    output_dir: str
    resolved_config: dict[str, Any]
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
    backend: str = "local"
    backend_profile: str | None = None
    scheduler_job_id: str | None = None
    process: mp.Process | None = None
    cancel_flag_path: str | None = None
    event_conn: Any | None = None


def _get_session_preprocess_jobs_locked(session_key: str) -> dict[str, PreprocessJob]:
    jobs = _PREPROCESS_JOBS.get(session_key)
    if jobs is None:
        jobs = {}
        _PREPROCESS_JOBS[session_key] = jobs
    return jobs


def _get_preprocess_job_locked(job_id: str, jobs_by_id: dict[str, PreprocessJob]) -> PreprocessJob:
    job = jobs_by_id.get(job_id)
    if job is None:
        raise KeyError(f"No preprocess job found for job_id={job_id!r}")
    return job


def _touch_preprocess_job(job: PreprocessJob) -> None:
    job.last_update_at = _utc_now_iso()
    job.last_update_monotonic = time.monotonic()


def _append_preprocess_job_log(job: PreprocessJob, message: str) -> None:
    _touch_preprocess_job(job)
    line = f"{_utc_now_iso()} {message}"
    job.logs.append(line)
    if len(job.logs) > _JOB_LOG_LIMIT:
        del job.logs[: len(job.logs) - _JOB_LOG_LIMIT]
