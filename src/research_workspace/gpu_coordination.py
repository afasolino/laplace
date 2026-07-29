"""Fail-closed GPU ownership classification for cooperative local workloads."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

GpuCoordinationStatus = Literal[
    "GPU_CLEAR",
    "GPU_CLEAR_LAPLACE_OWNED_ONLY",
    "BLOCKED_BY_SPECDEC_ACTIVE",
    "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
]

_PROTECTED_MARKERS = (
    "/home/giando/projects/specdec_ladder",
    "specdec_ladder",
    "speculative decoding",
    "humaneval",
    "quantization",
    "recovery",
    "adaptation",
    "qwen",
    "drafter",
    "verifier",
)


@dataclass(frozen=True)
class ProcessEvidence:
    pid: int
    parent_pid: int
    executable_name: str
    command_sha256: str
    cwd_classification: str
    protected_markers: tuple[str, ...]


ProcessReader = Callable[[int], ProcessEvidence]
ParentReader = Callable[[int], int]


def _safe_read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError("gpu_process_identity_unavailable") from exc


def read_process_evidence(pid: int) -> ProcessEvidence:
    """Read a sanitized identity without retaining command arguments or full paths."""

    if pid <= 0:
        raise RuntimeError("gpu_process_identity_invalid")
    proc = Path("/proc") / str(pid)
    command_raw = _safe_read(proc / "cmdline")
    command_text = command_raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    try:
        cwd = os.readlink(proc / "cwd")
    except OSError as exc:
        raise RuntimeError("gpu_process_identity_unavailable") from exc
    status = _safe_read(proc / "status").decode("utf-8", errors="replace")
    parent_line = next(
        (line for line in status.splitlines() if line.startswith("PPid:")),
        "",
    )
    parent_value = parent_line.partition(":")[2].strip()
    if not command_text or not parent_value.isdigit():
        raise RuntimeError("gpu_process_identity_unavailable")
    combined = f"{command_text}\n{cwd}".casefold()
    markers = tuple(marker for marker in _PROTECTED_MARKERS if marker in combined)
    cwd_classification = (
        "SPECDEC_PROTECTED"
        if any(marker in cwd.casefold() for marker in _PROTECTED_MARKERS)
        else "OTHER"
    )
    executable = Path(command_text.split(maxsplit=1)[0]).name[:120]
    return ProcessEvidence(
        pid=pid,
        parent_pid=int(parent_value),
        executable_name=executable,
        command_sha256=hashlib.sha256(command_raw).hexdigest(),
        cwd_classification=cwd_classification,
        protected_markers=markers,
    )


def _is_descendant(
    pid: int,
    allowed_roots: frozenset[int],
    *,
    reader: ProcessReader,
) -> bool:
    current = pid
    seen: set[int] = set()
    for _ in range(16):
        if current in allowed_roots:
            return True
        if current <= 1 or current in seen:
            return False
        seen.add(current)
        current = reader(current).parent_pid
    raise RuntimeError("gpu_process_parent_chain_too_deep")


def classify_compute_ownership(
    compute_pids: tuple[int, ...],
    *,
    allowed_laplace_roots: tuple[int, ...] = (),
    reader: ProcessReader = read_process_evidence,
) -> dict[str, object]:
    """Classify all compute PIDs; any unresolved identity fails closed."""

    if not compute_pids:
        return {
            "status": "GPU_CLEAR",
            "compute_pids": [],
            "processes": [],
        }
    allowed = frozenset(allowed_laplace_roots)
    processes: list[ProcessEvidence] = []
    owned: list[int] = []
    try:
        for pid in sorted(set(compute_pids)):
            evidence = reader(pid)
            processes.append(evidence)
            if allowed and _is_descendant(pid, allowed, reader=reader):
                owned.append(pid)
    except (OSError, RuntimeError, ValueError):
        return {
            "status": "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP",
            "compute_pids": sorted(set(compute_pids)),
            "processes": [
                {
                    "pid": item.pid,
                    "parent_pid": item.parent_pid,
                    "executable_name": item.executable_name,
                    "command_sha256": item.command_sha256,
                    "cwd_classification": item.cwd_classification,
                    "protected_markers": list(item.protected_markers),
                }
                for item in processes
            ],
            "reason": "process_identity_or_parent_chain_unavailable",
        }
    protected = [
        item
        for item in processes
        if item.protected_markers and item.pid not in owned
    ]
    serialized = [
        {
            "pid": item.pid,
            "parent_pid": item.parent_pid,
            "executable_name": item.executable_name,
            "command_sha256": item.command_sha256,
            "cwd_classification": item.cwd_classification,
            "protected_markers": list(item.protected_markers),
            "laplace_owned": item.pid in owned,
        }
        for item in processes
    ]
    if protected:
        status: GpuCoordinationStatus = "BLOCKED_BY_SPECDEC_ACTIVE"
    elif len(owned) == len(processes):
        status = "GPU_CLEAR_LAPLACE_OWNED_ONLY"
    else:
        status = "BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP"
    return {
        "status": status,
        "compute_pids": sorted(set(compute_pids)),
        "processes": serialized,
    }
