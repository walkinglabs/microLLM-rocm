#!/usr/bin/env python3
"""Measure Python HIP Event completion without device-wide synchronization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--host-work", type=int, default=100000)
    parser.add_argument("--run-id", default="python-hip-event")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if (arguments.device < 0 or arguments.rows <= 0 or arguments.width <= 0 or
            arguments.host_work <= 0):
        raise ValueError("device must be non-negative and work dimensions positive")

    from microllm import Tensor, hip_device_count, softmax
    from microllm.profiling import (hip_event_profile_scope, profile_scope,
                                    roctx_available)

    output = Path(arguments.output)
    report_path = Path(arguments.report)
    for path in (output, report_path):
        if path.exists() and not arguments.overwrite:
            raise FileExistsError(f"refusing to replace {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    if hip_device_count() <= arguments.device:
        raise RuntimeError(f"HIP device {arguments.device} is unavailable")
    if not roctx_available():
        raise RuntimeError("ROCTX runtime is unavailable")

    elements = arguments.rows * arguments.width
    tensor = Tensor.from_f32([1.0] * elements,
                             (arguments.rows, arguments.width)).to(
                                 f"hip:{arguments.device}")
    warmup = softmax(tensor)
    expected = 1.0 / arguments.width
    if abs(warmup.tolist()[0] - expected) > 1.0e-7:
        raise RuntimeError("softmax warm-up failed")

    roctx_warmup = output.with_name(output.stem + "-roctx-warmup.jsonl")
    event_warmup = output.with_name(output.stem + "-event-warmup.jsonl")
    roctx_warmup.write_text("", encoding="utf-8")
    event_warmup.write_text("", encoding="utf-8")
    with profile_scope("roctx.warmup", output=roctx_warmup,
                       run_id=arguments.run_id, emit_roctx=True):
        pass
    with hip_event_profile_scope(
            "event.warmup", output=event_warmup,
            device=f"hip:{arguments.device}", run_id=arguments.run_id) as warm:
        warm_result = softmax(tensor)
    warm.wait()
    if abs(warm_result.tolist()[0] - expected) > 1.0e-7:
        raise RuntimeError("Event warm-up failed")

    with hip_event_profile_scope(
            "softmax.async", output=output, device=f"hip:{arguments.device}",
            run_id=arguments.run_id,
            metadata={"rows": arguments.rows, "width": arguments.width},
            emit_roctx=True) as completion:
        result = softmax(tensor)
    ready_after_submit = completion.ready()
    future = completion.observe_async()
    host_start_ns = time.perf_counter_ns()
    host_value = sum(index * index for index in range(arguments.host_work))
    host_finish_ns = time.perf_counter_ns()
    record = future.result(timeout=30.0)
    completion.close()
    values = result.tolist()
    maximum_error = max(abs(values[0] - expected),
                        abs(values[-1] - expected))
    completion_observed_ns = int(record["completion_observed_ns"])
    host_before_observation_ns = max(
        0, min(host_finish_ns, completion_observed_ns) - host_start_ns)
    report = {
        "schema_version": 1,
        "status": "pass",
        "device": arguments.device,
        "rows": arguments.rows,
        "width": arguments.width,
        "elements": elements,
        "event_ready_at_submit": ready_after_submit,
        "submission_duration_ns": int(record["submission_duration_ns"]),
        "completion_duration_ns": int(record["duration_ns"]),
        "device_elapsed_ns": int(record["device_elapsed_ns"]),
        "host_work_ns": host_finish_ns - host_start_ns,
        "host_work_before_completion_observed_ns": host_before_observation_ns,
        "observer_thread_is_distinct": (
            int(record["completion_observer_native_thread_id"]) !=
            int(record["native_thread_id"])),
        "synchronization_scope": record["synchronization_scope"],
        "maximum_output_error": maximum_error,
        "host_work_sentinel": host_value,
        "profile_output": str(output),
    }
    if (ready_after_submit or host_before_observation_ns <= 0 or
            maximum_error > 1.0e-7 or
            not report["observer_thread_is_distinct"] or
            report["synchronization_scope"] != "hip_event_default_stream"):
        raise RuntimeError("asynchronous Event completion gate failed")
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
