from __future__ import annotations

import os
import sys
import threading
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any

from ._jobs import (
    EmbInferenceCancelledError,
    TxTrainCancelledError,
    _HEARTBEAT_INTERVAL_SECONDS,
    _utc_now_iso,
)


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
