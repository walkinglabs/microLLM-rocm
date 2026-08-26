"""Small schema-versioned Python span profiler for scripts and API calls."""

from __future__ import annotations

import contextvars
import csv
import functools
import inspect
import json
import threading
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar


_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "microllm_profile_depth", default=0)
_LOCK = threading.Lock()
_F = TypeVar("_F", bound=Callable[..., Any])


class ProfileScope(AbstractContextManager["ProfileScope"]):
    def __init__(self, name: str, *, output: str | Path,
                 phase: str = "python", run_id: str | None = None,
                 metadata: Mapping[str, Any] | None = None) -> None:
        if not name or not phase:
            raise ValueError("profile name and phase must be non-empty")
        self.name = name
        self.output = Path(output)
        self.phase = phase
        self.run_id = run_id or uuid.uuid4().hex
        self.metadata = dict(metadata or {})
        json.dumps(self.metadata, allow_nan=False)
        self._token: contextvars.Token[int] | None = None
        self._depth = 0
        self._start_ns = 0

    def __enter__(self) -> "ProfileScope":
        self._depth = _DEPTH.get()
        self._token = _DEPTH.set(self._depth + 1)
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        finish_ns = time.perf_counter_ns()
        if self._token is None:
            raise RuntimeError("profile scope exited before entering")
        _DEPTH.reset(self._token)
        record = {
            "schema_version": 1,
            "framework": "microllm-python",
            "record_type": "python_profile_span",
            "run_id": self.run_id,
            "phase": self.phase,
            "kind": "python_span",
            "name": self.name,
            "depth": self._depth,
            "thread_id": threading.get_ident(),
            "start_ns": self._start_ns,
            "duration_ns": finish_ns - self._start_ns,
            "status": "error" if exception_type is not None else "pass",
            "exception_type": (exception_type.__name__
                               if exception_type is not None else None),
            "metadata": self.metadata,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with _LOCK, self.output.open("a", encoding="utf-8") as stream:
            stream.write(line)
        return False


def profile_scope(name: str, *, output: str | Path,
                  phase: str = "python", run_id: str | None = None,
                  metadata: Mapping[str, Any] | None = None) -> ProfileScope:
    return ProfileScope(name, output=output, phase=phase,
                        run_id=run_id, metadata=metadata)


def profile(function: _F | None = None, *, name: str | None = None,
            output: str | Path, phase: str = "python",
            run_id: str | None = None,
            metadata: Mapping[str, Any] | None = None):
    """Decorate a sync or async callable and append one JSONL span per call."""
    def decorate(target: _F) -> _F:
        span_name = name or target.__qualname__
        if inspect.iscoroutinefunction(target):
            @functools.wraps(target)
            async def async_wrapper(*args, **kwargs):
                with profile_scope(span_name, output=output, phase=phase,
                                   run_id=run_id, metadata=metadata):
                    return await target(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            with profile_scope(span_name, output=output, phase=phase,
                               run_id=run_id, metadata=metadata):
                return target(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    return decorate if function is None else decorate(function)


def export_perfetto(input_jsonl: str | Path,
                    output_json: str | Path) -> dict[str, Any]:
    """Convert microLLM Python span JSONL to Chrome/Perfetto Trace Event JSON."""
    source = Path(input_jsonl)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line]
    if not rows:
        raise ValueError("profile JSONL is empty")
    for row in rows:
        if (row.get("schema_version") != 1 or
                row.get("record_type") != "python_profile_span" or
                int(row.get("start_ns", -1)) < 0 or
                int(row.get("duration_ns", -1)) < 0):
            raise ValueError("profile JSONL contains an incompatible record")
    origin = min(int(row["start_ns"]) for row in rows)
    events = []
    for row in rows:
        events.append({
            "name": row["name"], "cat": row["phase"], "ph": "X",
            "ts": (int(row["start_ns"]) - origin) / 1000.0,
            "dur": int(row["duration_ns"]) / 1000.0,
            "pid": 0, "tid": int(row.get("thread_id", 0)),
            "args": {"status": row["status"], "depth": row["depth"],
                     "run_id": row["run_id"],
                     "exception_type": row["exception_type"],
                     **row["metadata"]},
        })
    document = {"traceEvents": events,
                "displayTimeUnit": "ms",
                "metadata": {"source": "microllm-python", "schema_version": 1}}
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return {"events": len(events), "output": str(destination),
            "origin_ns": origin}


def merge_rocprof_perfetto(marker_csv: str | Path,
                           kernel_csv: str | Path,
                           output_json: str | Path) -> dict[str, Any]:
    """Merge rocprof marker/kernel CSV using correlation IDs and trace flows."""
    with Path(marker_csv).open(newline="", encoding="utf-8") as stream:
        markers = list(csv.DictReader(stream))
    with Path(kernel_csv).open(newline="", encoding="utf-8") as stream:
        kernels = list(csv.DictReader(stream))
    if not markers or not kernels:
        raise ValueError("rocprof marker and kernel traces must both be non-empty")
    required_marker = {"Function", "Process_Id", "Thread_Id", "Correlation_Id",
                       "Start_Timestamp", "End_Timestamp"}
    required_kernel = {"Agent_Id", "Queue_Id", "Kernel_Name", "Correlation_Id",
                       "Start_Timestamp", "End_Timestamp"}
    if required_marker - markers[0].keys() or required_kernel - kernels[0].keys():
        raise ValueError("rocprof CSV schema is incompatible")
    starts = [int(row["Start_Timestamp"]) for row in markers + kernels]
    origin = min(starts)
    events: list[dict[str, Any]] = []
    marker_ids = {int(row["Correlation_Id"]) for row in markers}
    kernel_ids = {int(row["Correlation_Id"]) for row in kernels}
    linked = marker_ids & kernel_ids
    for row in markers:
        correlation = int(row["Correlation_Id"])
        start = int(row["Start_Timestamp"])
        events.append({"name": row["Function"], "cat": "roctx", "ph": "X",
                       "ts": (start-origin)/1000.0,
                       "dur": (int(row["End_Timestamp"])-start)/1000.0,
                       "pid": int(row["Process_Id"]),
                       "tid": int(row["Thread_Id"]),
                       "args": {"correlation_id": correlation}})
        if correlation in linked:
            events.append({"name": "ROCTX to HIP", "cat": "correlation",
                           "ph": "s", "ts": (start-origin)/1000.0,
                           "pid": int(row["Process_Id"]),
                           "tid": int(row["Thread_Id"]), "id": correlation})
    for row in kernels:
        correlation = int(row["Correlation_Id"])
        start = int(row["Start_Timestamp"])
        gpu_tid = 1_000_000 + int(row["Queue_Id"])
        events.append({"name": row["Kernel_Name"], "cat": "gpu_kernel", "ph": "X",
                       "ts": (start-origin)/1000.0,
                       "dur": (int(row["End_Timestamp"])-start)/1000.0,
                       "pid": 0, "tid": gpu_tid,
                       "args": {"agent": row["Agent_Id"],
                                "correlation_id": correlation}})
        if correlation in linked:
            events.append({"name": "ROCTX to HIP", "cat": "correlation",
                           "ph": "f", "ts": (start-origin)/1000.0,
                           "pid": 0, "tid": gpu_tid, "id": correlation})
    document = {"traceEvents": events, "displayTimeUnit": "ms",
                "metadata": {"source": "microllm-rocprof-merge",
                             "schema_version": 1}}
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True)+"\n", encoding="utf-8")
    temporary.replace(destination)
    return {"marker_events": len(markers), "kernel_events": len(kernels),
            "correlated_ids": len(linked), "trace_events": len(events),
            "output": str(destination)}


__all__ = ["ProfileScope", "export_perfetto", "merge_rocprof_perfetto",
           "profile", "profile_scope"]
