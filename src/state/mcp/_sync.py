from __future__ import annotations

import csv
import multiprocessing as mp
import signal
import time
from pathlib import Path
from typing import Any

from ._jobs import (
    InferenceJob,
    PreprocessJob,
    TrainJob,
    _CANCEL_GRACE_SECONDS,
    _MAX_EVENT_DRAIN,
    _TERMINAL_JOB_STATUSES,
    _TERMINATE_GRACE_SECONDS,
    _append_job_log,
    _append_preprocess_job_log,
    _append_train_job_log,
    _touch_job,
    _touch_preprocess_job,
    _touch_train_job,
    _utc_now_iso,
)
from ._slurm import _map_slurm_state_to_job_status, _query_slurm_state


# ---------------------------------------------------------------------------
# Log ingestion
# ---------------------------------------------------------------------------


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


def _ingest_inference_worker_log_lines(job: InferenceJob) -> None:
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
        _append_job_log(job, f"[worker] {line}")


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Worker message handling
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Event draining
# ---------------------------------------------------------------------------


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
        _append_train_job_log(
            job, f"[warning] Drained {_MAX_EVENT_DRAIN} worker events; additional events remain queued."
        )


# ---------------------------------------------------------------------------
# Poll intervals
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# State sync
# ---------------------------------------------------------------------------


def _sync_job_state_locked(job: InferenceJob) -> None:
    if job.backend == "slurm":
        _ingest_inference_worker_log_lines(job)
        if isinstance(job.scheduler_job_id, str) and job.scheduler_job_id.strip():
            state = _query_slurm_state(job.scheduler_job_id)
            if isinstance(state, str) and state:
                mapped = _map_slurm_state_to_job_status(state)
                current_progress = dict(job.progress)
                current_progress["scheduler_state"] = state
                job.progress = current_progress
                _touch_job(job)

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
                _append_job_log(
                    job, f"[warning] Failed to terminate worker process cleanly: {type(exc).__name__}: {exc}"
                )
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
                mapped = _map_slurm_state_to_job_status(state)
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


# ---------------------------------------------------------------------------
# Preprocess job helpers
# ---------------------------------------------------------------------------


def _ingest_preprocess_worker_log_lines(job: PreprocessJob) -> None:
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
        _append_preprocess_job_log(job, f"[worker] {line}")


def _record_preprocess_job_event(job: PreprocessJob, event: dict[str, Any]) -> None:
    _touch_preprocess_job(job)
    job.last_event = dict(event)
    progress: dict[str, Any] = {
        "kind": event.get("kind"),
        "phase": event.get("phase"),
        "files_done": event.get("files_done"),
        "files_total": event.get("files_total"),
        "current_file": event.get("current_file"),
        "message": event.get("message"),
    }
    files_done = event.get("files_done")
    files_total = event.get("files_total")
    if isinstance(files_done, int) and isinstance(files_total, int) and files_total > 0:
        progress["percent"] = round((files_done / files_total) * 100.0, 3)
    job.progress = progress

    message = event.get("message")
    if isinstance(message, str) and message.strip():
        kind = str(event.get("kind") or "event")
        _append_preprocess_job_log(job, f"[{kind}] {message}")


def _apply_preprocess_worker_message(job: PreprocessJob, message: dict[str, Any]) -> None:
    if not isinstance(message, dict):
        return

    msg_type = message.get("type")
    if msg_type == "event":
        event = message.get("event")
        if isinstance(event, dict):
            _record_preprocess_job_event(job, event)
        return

    if msg_type == "heartbeat":
        _touch_preprocess_job(job)
        current_progress = dict(job.progress)
        current_progress["worker_heartbeat_at"] = job.last_update_at
        job.progress = current_progress
        return

    if msg_type == "status":
        _touch_preprocess_job(job)
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
            _append_preprocess_job_log(job, note)
        return

    if msg_type == "traceback":
        trace = message.get("traceback")
        if isinstance(trace, str) and trace.strip():
            _append_preprocess_job_log(job, trace.strip())
        return


def _drain_preprocess_job_events_locked(job: PreprocessJob) -> None:
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
        _apply_preprocess_worker_message(job, message)

    if drained >= _MAX_EVENT_DRAIN:
        _append_preprocess_job_log(
            job, f"[warning] Drained {_MAX_EVENT_DRAIN} worker events; additional events remain queued."
        )


def _release_preprocess_job_runtime_resources(job: PreprocessJob) -> None:
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


def _recommend_preprocess_poll_interval_seconds(job: PreprocessJob, *, for_logs: bool = False) -> float | None:
    if job.status in _TERMINAL_JOB_STATUSES:
        return None
    if job.status == "queued":
        interval = 3.0
    elif job.status == "cancelling":
        interval = 4.0
    else:
        interval = 5.0

    seconds_since_update = max(0.0, time.monotonic() - float(job.last_update_monotonic))
    if job.status == "running" and seconds_since_update >= 120.0:
        interval = max(interval, 15.0)

    if for_logs:
        interval = max(interval * 1.5, 6.0)

    return round(interval, 1)


def _sync_preprocess_job_state_locked(job: PreprocessJob) -> None:
    _ingest_preprocess_worker_log_lines(job)

    if job.backend == "slurm":
        if isinstance(job.scheduler_job_id, str) and job.scheduler_job_id.strip():
            state = _query_slurm_state(job.scheduler_job_id)
            if isinstance(state, str) and state:
                mapped = _map_slurm_state_to_job_status(state)
                current_progress = dict(job.progress)
                current_progress["scheduler_state"] = state
                job.progress = current_progress
                _touch_preprocess_job(job)

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

    _drain_preprocess_job_events_locked(job)

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
                _append_preprocess_job_log(
                    job, "[cancelling] Cancellation grace elapsed; sent SIGTERM to worker process."
                )
            except Exception as exc:
                _append_preprocess_job_log(
                    job, f"[warning] Failed to terminate worker process cleanly: {type(exc).__name__}: {exc}"
                )
        elif (
            job.terminate_sent_at_monotonic is not None
            and now >= (job.terminate_sent_at_monotonic + _TERMINATE_GRACE_SECONDS)
            and _process_alive(process)
        ):
            try:
                process.kill()
                job.terminate_sent_at_monotonic = now
                _append_preprocess_job_log(job, "[cancelling] Worker did not exit after SIGTERM; sent SIGKILL.")
            except Exception as exc:
                _append_preprocess_job_log(
                    job, f"[warning] Failed to force-kill worker process: {type(exc).__name__}: {exc}"
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
                job.error = "Preprocessing cancelled by request."
            _append_preprocess_job_log(job, "[cancelled] Preprocessing cancelled.")
        elif exit_code == 0:
            job.status = "succeeded"
            _append_preprocess_job_log(job, "[succeeded] Preprocessing completed.")
        else:
            job.status = "failed"
            if not job.error:
                job.error = f"Worker process exited unexpectedly with code {exit_code}."
            _append_preprocess_job_log(job, f"[failed] {job.error}")

    if job.finished_at is None:
        job.finished_at = _utc_now_iso()

    _release_preprocess_job_runtime_resources(job)
