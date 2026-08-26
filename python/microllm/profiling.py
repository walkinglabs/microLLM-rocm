"""Small schema-versioned Python span profiler for scripts and API calls."""

from __future__ import annotations

import contextvars
import csv
import ctypes
import functools
import inspect
import json
import math
import os
import threading
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar


_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "microllm_profile_depth", default=0)
_LOCK = threading.Lock()
_ROCTX_LOCK = threading.Lock()
_F = TypeVar("_F", bound=Callable[..., Any])


class _RoctxRuntime:
    def __init__(self) -> None:
        self._library: ctypes.CDLL | None = None
        self._push = None
        self._pop = None
        for name in ("librocprofiler-sdk-roctx.so", "libroctx64.so"):
            try:
                library = ctypes.CDLL(name)
                push = library.roctxRangePushA
                pop = library.roctxRangePop
                push.argtypes = [ctypes.c_char_p]
                push.restype = ctypes.c_int
                pop.argtypes = []
                pop.restype = ctypes.c_int
                self._library = library
                self._push = push
                self._pop = pop
                break
            except (AttributeError, OSError):
                continue

    @property
    def available(self) -> bool:
        return self._push is not None and self._pop is not None

    def push(self, name: str) -> bool:
        return bool(self.available and self._push(name.encode("utf-8")) >= 0)

    def pop(self) -> bool:
        return bool(self.available and self._pop() >= 0)


_ROCTX_RUNTIME: _RoctxRuntime | None = None


def _roctx_runtime() -> _RoctxRuntime:
    global _ROCTX_RUNTIME
    if _ROCTX_RUNTIME is None:
        with _ROCTX_LOCK:
            if _ROCTX_RUNTIME is None:
                _ROCTX_RUNTIME = _RoctxRuntime()
    return _ROCTX_RUNTIME


def roctx_available() -> bool:
    """Return whether Python can load a compatible ROCTX range runtime."""
    return _roctx_runtime().available


def _roctx_range_name(name: str, span_id: str) -> str:
    readable = "".join(character if character.isalnum() or character in ".-_"
                       else "_" for character in name)[:64]
    return f"microllm.python.{span_id}.{readable or 'span'}"


class ProfileScope(AbstractContextManager["ProfileScope"]):
    def __init__(self, name: str, *, output: str | Path,
                 phase: str = "python", run_id: str | None = None,
                 metadata: Mapping[str, Any] | None = None,
                 emit_roctx: bool = False) -> None:
        if not name or not phase:
            raise ValueError("profile name and phase must be non-empty")
        self.name = name
        self.output = Path(output)
        self.phase = phase
        self.run_id = run_id or uuid.uuid4().hex
        self.metadata = dict(metadata or {})
        self.emit_roctx = bool(emit_roctx)
        self.span_id = uuid.uuid4().hex
        json.dumps(self.metadata, allow_nan=False)
        self._token: contextvars.Token[int] | None = None
        self._depth = 0
        self._start_ns = 0
        self._roctx_emitted = False
        self._roctx_status = "disabled"
        self._roctx_range_name: str | None = None
        self._roctx_push_before_ns: int | None = None
        self._roctx_push_after_ns: int | None = None
        self._roctx_pop_before_ns: int | None = None
        self._roctx_pop_after_ns: int | None = None

    def __enter__(self) -> "ProfileScope":
        self._depth = _DEPTH.get()
        self._token = _DEPTH.set(self._depth + 1)
        self._start_ns = time.perf_counter_ns()
        if self.emit_roctx:
            runtime = _roctx_runtime()
            if not runtime.available:
                self._roctx_status = "unavailable"
            else:
                self._roctx_range_name = _roctx_range_name(self.name, self.span_id)
                self._roctx_push_before_ns = time.perf_counter_ns()
                try:
                    self._roctx_emitted = runtime.push(self._roctx_range_name)
                except (OSError, ValueError):
                    self._roctx_emitted = False
                self._roctx_push_after_ns = time.perf_counter_ns()
                self._roctx_status = "emitted" if self._roctx_emitted else "push_error"
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if self._token is None:
            raise RuntimeError("profile scope exited before entering")
        if self._roctx_emitted:
            self._roctx_pop_before_ns = time.perf_counter_ns()
            try:
                popped = _roctx_runtime().pop()
            except (OSError, ValueError):
                popped = False
            self._roctx_pop_after_ns = time.perf_counter_ns()
            if not popped:
                self._roctx_status = "pop_error"
        finish_ns = time.perf_counter_ns()
        _DEPTH.reset(self._token)
        record = {
            "schema_version": 1,
            "framework": "microllm-python",
            "record_type": "python_profile_span",
            "run_id": self.run_id,
            "phase": self.phase,
            "kind": "python_span",
            "name": self.name,
            "span_id": self.span_id,
            "depth": self._depth,
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "native_thread_id": threading.get_native_id(),
            "start_ns": self._start_ns,
            "duration_ns": finish_ns - self._start_ns,
            "status": "error" if exception_type is not None else "pass",
            "exception_type": (exception_type.__name__
                               if exception_type is not None else None),
            "metadata": self.metadata,
            "roctx_requested": self.emit_roctx,
            "roctx_emitted": self._roctx_emitted,
            "roctx_status": self._roctx_status,
            "roctx_range_name": self._roctx_range_name,
            "roctx_push_before_ns": self._roctx_push_before_ns,
            "roctx_push_after_ns": self._roctx_push_after_ns,
            "roctx_pop_before_ns": self._roctx_pop_before_ns,
            "roctx_pop_after_ns": self._roctx_pop_after_ns,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with _LOCK, self.output.open("a", encoding="utf-8") as stream:
            stream.write(line)
        return False


def profile_scope(name: str, *, output: str | Path,
                  phase: str = "python", run_id: str | None = None,
                  metadata: Mapping[str, Any] | None = None,
                  emit_roctx: bool = False) -> ProfileScope:
    return ProfileScope(name, output=output, phase=phase,
                        run_id=run_id, metadata=metadata,
                        emit_roctx=emit_roctx)


def profile(function: _F | None = None, *, name: str | None = None,
            output: str | Path, phase: str = "python",
            run_id: str | None = None,
            metadata: Mapping[str, Any] | None = None,
            emit_roctx: bool = False):
    """Decorate a sync or async callable and append one JSONL span per call."""
    def decorate(target: _F) -> _F:
        span_name = name or target.__qualname__
        if inspect.iscoroutinefunction(target):
            @functools.wraps(target)
            async def async_wrapper(*args, **kwargs):
                with profile_scope(span_name, output=output, phase=phase,
                                   run_id=run_id, metadata=metadata,
                                   emit_roctx=emit_roctx):
                    return await target(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            with profile_scope(span_name, output=output, phase=phase,
                               run_id=run_id, metadata=metadata,
                               emit_roctx=emit_roctx):
                return target(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    return decorate if function is None else decorate(function)


def _profile_rows(input_jsonl: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in
            Path(input_jsonl).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("profile JSONL is empty")
    for row in rows:
        if (row.get("schema_version") != 1 or
                row.get("record_type") != "python_profile_span" or
                int(row.get("start_ns", -1)) < 0 or
                int(row.get("duration_ns", -1)) < 0 or
                not isinstance(row.get("metadata"), dict)):
            raise ValueError("profile JSONL contains an incompatible record")
    return rows


def _write_json_atomic(document: Mapping[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)


def export_perfetto(input_jsonl: str | Path,
                    output_json: str | Path) -> dict[str, Any]:
    """Convert microLLM Python span JSONL to Chrome/Perfetto Trace Event JSON."""
    rows = _profile_rows(input_jsonl)
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
    _write_json_atomic(document, destination)
    return {"events": len(events), "output": str(destination),
            "origin_ns": origin}


def _clock_calibration(rows: list[dict[str, Any]],
                       markers: list[dict[str, str]]) -> dict[str, Any]:
    process_ids = {int(marker["Process_Id"]) for marker in markers}
    if len(process_ids) != 1:
        raise ValueError("clock calibration currently requires one rocprof process")
    marker_index: dict[tuple[int, str], list[dict[str, str]]] = {}
    for marker in markers:
        if int(marker["End_Timestamp"]) < int(marker["Start_Timestamp"]):
            raise ValueError("rocprof marker CSV contains a negative duration")
        key = (int(marker["Process_Id"]), marker["Function"])
        marker_index.setdefault(key, []).append(marker)
    points: list[tuple[int, int]] = []
    matched_spans = 0
    max_boundary_width = 0
    for row in rows:
        if not row.get("roctx_emitted"):
            continue
        name = row.get("roctx_range_name")
        process = int(row.get("process_id", -1))
        matches = marker_index.get((process, name), [])
        if len(matches) != 1:
            raise ValueError(
                f"ROCTX range {name!r} for process {process} matched {len(matches)} markers")
        boundary_names = (
            ("roctx_push_before_ns", "roctx_push_after_ns", "Start_Timestamp"),
            ("roctx_pop_before_ns", "roctx_pop_after_ns", "End_Timestamp"),
        )
        for before_name, after_name, marker_name in boundary_names:
            before = int(row.get(before_name, -1))
            after = int(row.get(after_name, -1))
            if before < 0 or after < before:
                raise ValueError("profile JSONL contains an invalid ROCTX boundary")
            max_boundary_width = max(max_boundary_width, after - before)
            points.append(((before + after) // 2,
                           int(matches[0][marker_name])))
        matched_spans += 1
    if matched_spans < 2:
        raise ValueError("clock calibration requires at least two captured ROCTX spans")
    python_anchor, rocprof_anchor = points[0]
    x = [float(source - python_anchor) for source, _ in points]
    y = [float(target - rocprof_anchor) for _, target in points]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0.0:
        raise ValueError("clock calibration boundaries do not span time")
    scale = sum((source - x_mean) * (target - y_mean)
                for source, target in zip(x, y)) / denominator
    if not 0.99 <= scale <= 1.01:
        raise ValueError(f"rocprof clock scale {scale:.9f} is outside the nanosecond gate")
    delta_intercept = y_mean - scale * x_mean
    calibrated_rocprof_anchor = int(round(rocprof_anchor + delta_intercept))
    residuals = [target - (calibrated_rocprof_anchor +
                           scale * (source - python_anchor))
                 for source, target in points]
    max_abs_residual = max(abs(value) for value in residuals)
    rms_residual = math.sqrt(sum(value * value for value in residuals) /
                             len(residuals))
    boundary_width_gate = 100_000
    residual_gate = 50_000
    if max_boundary_width > boundary_width_gate:
        raise ValueError(
            f"ROCTX boundary width {max_boundary_width}ns exceeds "
            f"the {boundary_width_gate}ns calibration gate")
    if max_abs_residual > residual_gate:
        raise ValueError(
            f"clock residual {max_abs_residual:.1f}ns exceeds "
            f"the {residual_gate}ns calibration gate")
    return {
        "schema_version": 1,
        "status": "pass",
        "clock_model": "affine_midpoint",
        "matched_spans": matched_spans,
        "boundary_points": len(points),
        "python_origin_ns": python_anchor,
        "rocprof_origin_ns": calibrated_rocprof_anchor,
        "scale": scale,
        "max_boundary_width_ns": max_boundary_width,
        "max_abs_residual_ns": max_abs_residual,
        "rms_residual_ns": rms_residual,
        "boundary_width_gate_ns": boundary_width_gate,
        "residual_gate_ns": residual_gate,
    }


def calibrate_python_rocprof_clock(profile_jsonl: str | Path,
                                   marker_csv: str | Path,
                                   output_json: str | Path | None = None
                                   ) -> dict[str, Any]:
    """Fit a measured affine map from Python perf-counter to rocprof nanoseconds."""
    rows = _profile_rows(profile_jsonl)
    with Path(marker_csv).open(newline="", encoding="utf-8") as stream:
        markers = list(csv.DictReader(stream))
    required = {"Function", "Process_Id", "Start_Timestamp", "End_Timestamp"}
    if not markers or required - markers[0].keys():
        raise ValueError("rocprof marker CSV schema is incompatible")
    report = _clock_calibration(rows, markers)
    if output_json is not None:
        _write_json_atomic(report, output_json)
        report = {**report, "output": str(Path(output_json))}
    return report


def merge_rocprof_perfetto(marker_csv: str | Path,
                           kernel_csv: str | Path,
                           output_json: str | Path, *,
                           python_jsonl: str | Path | None = None) -> dict[str, Any]:
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
    if len({int(row["Process_Id"]) for row in markers}) != 1:
        raise ValueError("Perfetto merge currently requires one rocprof process")
    if len({int(row["Correlation_Id"]) for row in markers}) != len(markers):
        raise ValueError("rocprof marker correlations must be unique in one process")
    for row in markers + kernels:
        if int(row["End_Timestamp"]) < int(row["Start_Timestamp"]):
            raise ValueError("rocprof CSV contains a negative duration")
    python_rows: list[dict[str, Any]] = []
    calibration: dict[str, Any] | None = None
    if python_jsonl is not None:
        python_rows = _profile_rows(python_jsonl)
        calibration = _clock_calibration(python_rows, markers)

    def python_to_rocprof(timestamp: int) -> int:
        assert calibration is not None
        return int(round(int(calibration["rocprof_origin_ns"]) +
                         float(calibration["scale"]) *
                         (timestamp - int(calibration["python_origin_ns"]))))

    starts = [int(row["Start_Timestamp"]) for row in markers + kernels]
    starts.extend(python_to_rocprof(int(row["start_ns"]))
                  for row in python_rows)
    origin = min(starts)
    events: list[dict[str, Any]] = []
    marker_ids = {int(row["Correlation_Id"]) for row in markers}
    kernel_ids = {int(row["Correlation_Id"]) for row in kernels}
    linked = marker_ids & kernel_ids
    correlated_kernel_indices: dict[int, list[int]] = {}
    for index, row in enumerate(kernels):
        correlation = int(row["Correlation_Id"])
        if correlation in linked:
            correlated_kernel_indices.setdefault(correlation, []).append(index)
    for row in markers:
        correlation = int(row["Correlation_Id"])
        start = int(row["Start_Timestamp"])
        events.append({"name": row["Function"], "cat": "roctx", "ph": "X",
                       "ts": (start-origin)/1000.0,
                       "dur": (int(row["End_Timestamp"])-start)/1000.0,
                       "pid": int(row["Process_Id"]),
                       "tid": int(row["Thread_Id"]),
                       "args": {"correlation_id": correlation}})
        for kernel_index in correlated_kernel_indices.get(correlation, []):
            events.append({"name": "ROCTX to HIP", "cat": "correlation",
                           "ph": "s", "ts": (start-origin)/1000.0,
                           "pid": int(row["Process_Id"]),
                           "tid": int(row["Thread_Id"]),
                           "id": f"{correlation}:{kernel_index}"})
    for kernel_index, row in enumerate(kernels):
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
                           "pid": 0, "tid": gpu_tid,
                           "id": f"{correlation}:{kernel_index}"})
    for row in python_rows:
        start = python_to_rocprof(int(row["start_ns"]))
        end = python_to_rocprof(int(row["start_ns"]) +
                                int(row["duration_ns"]))
        events.append({
            "name": row["name"], "cat": row["phase"], "ph": "X",
            "ts": (start-origin)/1000.0, "dur": (end-start)/1000.0,
            "pid": int(row.get("process_id", 0)),
            "tid": int(row.get("native_thread_id", row.get("thread_id", 0))),
            "args": {"status": row["status"], "depth": row["depth"],
                     "run_id": row["run_id"], "span_id": row.get("span_id"),
                     "roctx_emitted": bool(row.get("roctx_emitted")),
                     **row["metadata"]},
        })
    document = {"traceEvents": events, "displayTimeUnit": "ms",
                "metadata": {"source": "microllm-rocprof-merge",
                             "schema_version": 1,
                             "python_clock_calibration": calibration}}
    destination = Path(output_json)
    _write_json_atomic(document, destination)
    return {"marker_events": len(markers), "kernel_events": len(kernels),
            "correlated_ids": len(linked), "trace_events": len(events),
            "correlated_pairs": sum(len(indices) for indices in
                                    correlated_kernel_indices.values()),
            "python_events": len(python_rows),
            "python_clock_calibration": calibration,
            "output": str(destination)}


__all__ = ["ProfileScope", "calibrate_python_rocprof_clock", "export_perfetto",
           "merge_rocprof_perfetto", "profile", "profile_scope",
           "roctx_available"]
