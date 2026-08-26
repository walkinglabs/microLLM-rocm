#!/usr/bin/env python3
"""Prove that waiting one Python HIP Event does not drain another Stream."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--busy-iterations", type=int, default=64)
    parser.add_argument("--run-id", default="python-stream-isolation")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.device < 0 or arguments.size <= 0 or arguments.busy_iterations <= 1:
        raise ValueError("device must be non-negative and workload dimensions positive")

    from microllm import (Event, Stream, Tensor, hip_device_count, matmul,
                          matmul_out)
    from microllm.profiling import hip_event_profile_scope, profile_scope

    output = Path(arguments.output)
    report_path = Path(arguments.report)
    for path in (output, report_path):
        if path.exists() and not arguments.overwrite:
            raise FileExistsError(f"refusing to replace {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    if hip_device_count() <= arguments.device:
        raise RuntimeError(f"HIP device {arguments.device} is unavailable")

    size = arguments.size
    elements = size * size
    device = f"hip:{arguments.device}"
    left = Tensor.from_f32([0.001] * elements, (size, size)).to(device)
    right = Tensor.from_f32([0.002] * elements, (size, size)).to(device)
    output_a = matmul(left, right)
    output_b = matmul(left, right)
    expected = size * 0.001 * 0.002
    if abs(output_a.tolist()[0] - expected) > 1.0e-3:
        raise RuntimeError("matmul allocation warm-up failed")
    if abs(output_b.tolist()[0] - expected) > 1.0e-3:
        raise RuntimeError("second matmul allocation warm-up failed")

    stream_a = Stream(device)
    stream_b = Stream(device)
    busy_finish = Event(device, enable_timing=False)
    roctx_warmup = output.with_name(output.stem + "-roctx-warmup.jsonl")
    event_warmup = output.with_name(output.stem + "-event-warmup.jsonl")
    roctx_warmup.write_text("", encoding="utf-8")
    event_warmup.write_text("", encoding="utf-8")
    with profile_scope("roctx.warmup", output=roctx_warmup,
                       run_id=arguments.run_id, emit_roctx=True):
        pass
    with hip_event_profile_scope(
            "stream.event.warmup", output=event_warmup, device=device,
            stream=stream_a, run_id=arguments.run_id) as warm:
        matmul_out(output_a, left, right, stream=stream_a)
    warm.wait()
    warm.close()

    with hip_event_profile_scope(
            "stream.a.matmul", output=output, device=device, stream=stream_a,
            run_id=arguments.run_id,
            metadata={"size": size, "busy_iterations": arguments.busy_iterations},
            emit_roctx=True) as completion:
        matmul_out(output_a, left, right, stream=stream_a)
    target_pending_at_submit = not completion.ready()

    for _ in range(arguments.busy_iterations):
        matmul_out(output_b, left, right, stream=stream_b)
    busy_finish.record(stream_b)
    busy_pending_before_target_wait = not busy_finish.ready()
    target_record = completion.observe_async().result(timeout=30.0)
    completion.close()
    busy_pending_after_target_wait = not busy_finish.ready()
    busy_finish.synchronize()
    target_values = output_a.tolist()
    busy_values = output_b.tolist()
    maximum_error = max(abs(target_values[0] - expected),
                        abs(target_values[-1] - expected),
                        abs(busy_values[0] - expected),
                        abs(busy_values[-1] - expected))
    report = {
        "schema_version": 1,
        "status": "pass",
        "device": arguments.device,
        "size": size,
        "busy_iterations": arguments.busy_iterations,
        "target_pending_at_submit": target_pending_at_submit,
        "busy_pending_before_target_wait": busy_pending_before_target_wait,
        "busy_pending_after_target_wait": busy_pending_after_target_wait,
        "target_device_elapsed_ns": int(target_record["device_elapsed_ns"]),
        "target_submission_duration_ns": int(
            target_record["submission_duration_ns"]),
        "synchronization_scope": target_record["synchronization_scope"],
        "observer_thread_is_distinct": (
            int(target_record["completion_observer_native_thread_id"]) !=
            int(target_record["native_thread_id"])),
        "maximum_output_error": maximum_error,
    }
    if (not target_pending_at_submit or
            not busy_pending_before_target_wait or
            not busy_pending_after_target_wait or
            report["synchronization_scope"] != "hip_event_explicit_stream" or
            not report["observer_thread_is_distinct"] or
            maximum_error > 1.0e-3):
        raise RuntimeError("explicit Stream isolation gate failed")
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    busy_finish.close()
    stream_a.close()
    stream_b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
