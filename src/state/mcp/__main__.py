from __future__ import annotations

import threading
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio

from .mcp_utils import (
    infer_tx_inference_defaults,
    inspect_model_folder,
    normalize_and_validate_run_dir,
    resolve_and_validate_model_folder,
)

try:
    from mcp.server.fastmcp import Context, FastMCP
except ImportError as exc:  # pragma: no cover - import-time dependency guidance
    raise ImportError(
        "MCP support requires the `mcp` package. Install it in this environment to run `state.mcp`."
    ) from exc


mcp = FastMCP("state")

# Session-local defaults for this MCP server process.
_SERVER_STATE: dict[str, str | None] = {"model_folder": None}

_TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
_JOB_LOG_LIMIT = 5000


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
    progress: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    last_event: dict[str, Any] | None = None
    thread: threading.Thread | None = None


_JOBS: dict[str, InferenceJob] = {}
_JOBS_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_job_log(job: InferenceJob, message: str) -> None:
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


def _make_ctx_notifier(ctx: Context):
    """Create a worker-thread-safe callback that forwards progress/logs to MCP context notifications."""
    state = {"last_progress_emit_ts": 0.0, "last_percent": -1.0}

    def notify(event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "event")
        message = event.get("message")
        percent = _event_progress_percent(event)

        if isinstance(message, str) and message.strip() and kind != "progress":
            try:
                anyio.from_thread.run(ctx.info, message)
            except Exception:
                pass

        if percent is None:
            return

        now = time.monotonic()
        should_emit = (
            percent in {0.0, 100.0}
            or abs(percent - float(state["last_percent"])) >= 0.5
            or (now - float(state["last_progress_emit_ts"])) >= 2.0
        )
        if not should_emit:
            return

        try:
            anyio.from_thread.run(
                ctx.report_progress,
                progress=percent,
                total=100.0,
                message=message if isinstance(message, str) else None,
            )
        except Exception:
            return

        state["last_progress_emit_ts"] = now
        state["last_percent"] = percent

    return notify


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

    resolved_model_folder = resolve_and_validate_model_folder(model_folder, _SERVER_STATE["model_folder"])
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


def _run_inference_job(job_id: str, infer_args: Namespace) -> None:
    from .._cli._tx._infer import InferenceCancelledError, run_tx_infer

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = _utc_now_iso()
        _append_job_log(job, "Inference job started.")

    def cancel_check() -> bool:
        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
            if current is None:
                return True
            return bool(current.cancel_requested)

    def progress_callback(event: dict[str, Any]) -> None:
        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
            if current is None:
                return
            _record_job_event(current, event)

    setattr(infer_args, "cancel_check", cancel_check)
    setattr(infer_args, "progress_callback", progress_callback)

    try:
        run_tx_infer(infer_args)
    except InferenceCancelledError as exc:
        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
            if current is not None:
                current.status = "cancelled"
                current.error = str(exc)
                current.finished_at = _utc_now_iso()
                _append_job_log(current, f"[cancelled] {exc}")
        return
    except Exception as exc:
        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
            if current is not None:
                current.status = "failed"
                current.error = f"{type(exc).__name__}: {exc}"
                current.finished_at = _utc_now_iso()
                _append_job_log(current, f"[failed] {type(exc).__name__}: {exc}")
                _append_job_log(current, traceback.format_exc().strip())
        return

    with _JOBS_LOCK:
        current = _JOBS.get(job_id)
        if current is not None:
            current.status = "succeeded"
            current.finished_at = _utc_now_iso()
            _append_job_log(current, "[succeeded] Inference job completed.")


@mcp.tool()
def set_model_folder(model_folder: str) -> dict[str, str]:
    """
    Set the default STATE model run folder for this MCP server session.

    The folder must contain STATE run artifacts:
    `config.yaml`, `data_module.{torch,pt,pkl}`, `pert_onehot_map.pt`, `var_dims.pkl`,
    batch/cell-type one-hot maps (`.torch`, `.pt`, or `.pkl`), and
    `checkpoints/` with at least one file.
    """
    resolved = normalize_and_validate_run_dir(model_folder)
    _SERVER_STATE["model_folder"] = resolved
    return {"status": "ok", "model_folder": resolved}


@mcp.tool()
def get_model_folder() -> dict[str, str | None]:
    """
    Return the current default STATE model run folder for this MCP server session.
    """
    return {"model_folder": _SERVER_STATE["model_folder"]}


@mcp.tool()
def clear_model_folder() -> dict[str, str]:
    """
    Clear the current default STATE model run folder for this MCP server session.
    """
    _SERVER_STATE["model_folder"] = None
    return {"status": "cleared"}


@mcp.tool()
def inspect_folder(model_folder: str | None = None) -> dict[str, Any]:
    """
    Inspect a STATE run folder for attributes like embed key (cell embedding that was used), pert col (expected column defining perturbation), etc.

    If `model_folder` is omitted, the current session default is used.
    """
    resolved = resolve_and_validate_model_folder(model_folder, _SERVER_STATE["model_folder"])
    return inspect_model_folder(resolved)


@mcp.tool()
async def run_tx_inference(
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
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Run `state tx infer` using defaults inferred from a validated model run folder.
    Streams progress notifications through MCP context when available.
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

    notifier = _make_ctx_notifier(ctx) if ctx is not None else None

    def progress_callback(event: dict[str, Any]) -> None:
        if notifier is not None:
            notifier(event)

    setattr(infer_args, "progress_callback", progress_callback)
    setattr(infer_args, "cancel_check", lambda: False)

    if ctx is not None:
        await ctx.info("Starting synchronous tx inference.")

    from .._cli._tx._infer import run_tx_infer

    await anyio.to_thread.run_sync(run_tx_infer, infer_args)

    return {
        "status": "ok",
        **resolved,
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
        resolved_args=resolved["resolved_args"],
    )
    _append_job_log(job, "[queued] Inference job queued.")

    thread = threading.Thread(
        target=_run_inference_job,
        args=(job_id, infer_args),
        daemon=True,
        name=f"state_tx_infer_{job_id[:8]}",
    )
    job.thread = thread

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    thread.start()

    return {
        "status": "started",
        "job_id": job_id,
        **resolved,
    }


@mcp.tool()
def get_tx_inference_status(job_id: str) -> dict[str, Any]:
    """
    Return status and progress for a background tx inference job.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise KeyError(f"No inference job found for job_id={job_id!r}")

        progress = dict(job.progress)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
            "model_folder": job.model_folder,
            "adata_path": job.adata_path,
            "output_path": job.output_path,
            "checkpoint_path": job.checkpoint_path,
            "resolved_args": dict(job.resolved_args),
            "progress": progress,
            "last_event": dict(job.last_event) if isinstance(job.last_event, dict) else None,
            "log_line_count": len(job.logs),
        }


@mcp.tool()
def get_tx_inference_logs(job_id: str, from_line: int = 0, max_lines: int = 200) -> dict[str, Any]:
    """
    Return buffered logs for a background tx inference job.
    """
    if from_line < 0:
        raise ValueError("`from_line` must be >= 0.")
    if max_lines <= 0:
        raise ValueError("`max_lines` must be > 0.")

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise KeyError(f"No inference job found for job_id={job_id!r}")

        start = min(from_line, len(job.logs))
        end = min(start + max_lines, len(job.logs))
        lines = job.logs[start:end]

    return {
        "job_id": job_id,
        "from_line": start,
        "next_line": end,
        "total_lines": len(job.logs),
        "lines": lines,
    }


@mcp.tool()
def cancel_tx_inference(job_id: str) -> dict[str, Any]:
    """
    Request cancellation for a background tx inference job.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise KeyError(f"No inference job found for job_id={job_id!r}")

        if job.status in _TERMINAL_JOB_STATUSES:
            return {
                "job_id": job_id,
                "status": job.status,
                "cancel_requested": job.cancel_requested,
                "message": "Job is already in a terminal state.",
            }

        job.cancel_requested = True
        if job.status in {"queued", "running"}:
            job.status = "cancelling"
        _append_job_log(job, "[cancelling] Cancellation requested.")

        return {
            "job_id": job_id,
            "status": job.status,
            "cancel_requested": job.cancel_requested,
        }


def main() -> None:
    """Run the STATE MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
