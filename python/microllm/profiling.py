"""Small schema-versioned Python span profiler for scripts and API calls."""

from __future__ import annotations

import contextvars
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


__all__ = ["ProfileScope", "profile", "profile_scope"]
