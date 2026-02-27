"""Lightweight NVIDIA GPU query utilities.

This module has **zero** STATE-specific imports — only stdlib (subprocess, shutil).
It is designed for future extraction into a shared package usable by any MCP server
(e.g. Boltz, AlphaFold3) that needs local GPU device management.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def is_nvidia_smi_available() -> bool:
    """Return True if ``nvidia-smi`` is on PATH."""
    return shutil.which("nvidia-smi") is not None


def _safe_int(s: str) -> int | None:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def _query_gpu_uuid_map() -> dict[int, str]:
    """Return ``{device_index: gpu_uuid}`` mapping."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
    except Exception:
        return {}

    mapping: dict[int, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) < 2:
            continue
        idx = _safe_int(parts[0])
        if idx is not None:
            mapping[idx] = parts[1]
    return mapping


def _query_gpu_processes() -> dict[str, list[dict[str, Any]]]:
    """Return ``{gpu_uuid: [{pid, process_name, used_gpu_memory_mb}]}``."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
    except Exception:
        return {}

    procs: dict[str, list[dict[str, Any]]] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        uuid = parts[0]
        entry: dict[str, Any] = {
            "pid": _safe_int(parts[1]),
            "process_name": parts[2],
            "used_gpu_memory_mb": _safe_int(parts[3]),
        }
        procs.setdefault(uuid, []).append(entry)
    return procs


def query_gpu_devices() -> list[dict[str, Any]]:
    """Query all local NVIDIA GPUs and return per-device status.

    Returns a list of dicts, each containing:
    - ``index``: device index (int)
    - ``name``: GPU model name
    - ``memory_total_mb``, ``memory_used_mb``, ``memory_free_mb``
    - ``utilization_percent``
    - ``temperature_celsius``
    - ``processes``: list of ``{pid, process_name, used_gpu_memory_mb}``

    Returns an empty list if ``nvidia-smi`` is unavailable or fails.
    """
    if not is_nvidia_smi_available():
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except Exception:
        return []

    uuid_map = _query_gpu_uuid_map()
    processes_by_uuid = _query_gpu_processes()

    devices: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx = _safe_int(parts[0])
        device: dict[str, Any] = {
            "index": idx,
            "name": parts[1],
            "memory_total_mb": _safe_int(parts[2]),
            "memory_used_mb": _safe_int(parts[3]),
            "memory_free_mb": _safe_int(parts[4]),
            "utilization_percent": _safe_int(parts[5]),
            "temperature_celsius": _safe_int(parts[6]),
            "processes": [],
        }
        uuid = uuid_map.get(idx) if idx is not None else None
        if uuid is not None:
            device["processes"] = processes_by_uuid.get(uuid, [])
        devices.append(device)
    return devices


def find_free_devices(
    n: int = 1,
    max_utilization: int = 50,
    min_free_memory_mb: int = 1000,
) -> list[int]:
    """Return up to *n* device indices that pass utilization and memory thresholds.

    Devices are sorted by ascending utilization, then descending free memory
    (prefer least-loaded, most-available).
    """
    devices = query_gpu_devices()
    candidates: list[dict[str, Any]] = []
    for d in devices:
        util = d.get("utilization_percent")
        free = d.get("memory_free_mb")
        if util is None or free is None:
            continue
        if util <= max_utilization and free >= min_free_memory_mb:
            candidates.append(d)
    candidates.sort(key=lambda d: (d["utilization_percent"], -(d["memory_free_mb"] or 0)))
    return [d["index"] for d in candidates[:n]]
