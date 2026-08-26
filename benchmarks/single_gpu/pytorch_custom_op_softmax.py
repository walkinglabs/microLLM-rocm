#!/usr/bin/env python3
"""PyTorch ROCm native versus C++ microLLM Custom Op Softmax matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--worker-order", choices=("native-first", "custom-first"))
    parser.add_argument("--worker-run", type=int, default=0)
    return parser.parse_args()


def errors(actual, expected) -> tuple[float, float]:
    import torch
    difference = actual.float() - expected.float()
    return float(difference.abs().max()), float(
        torch.sqrt(torch.mean(difference.square())))


def time_policy(function, stream, warmup: int, repetitions: int) -> dict:
    import torch
    output = None
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            output = function()
    stream.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(repetitions):
            output = function()
        finish.record(stream)
    finish.synchronize()
    wall_finish = time.perf_counter_ns()
    return {
        "event_ms": start.elapsed_time(finish) / repetitions,
        "wall_ms": (wall_finish - wall_start) / 1_000_000 / repetitions,
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - before,
        "output": output,
    }


def worker(options: argparse.Namespace) -> int:
    import os
    os.environ["MICROLLM_TORCH_OP_LIBRARY"] = str(options.library)
    import torch
    from microllm import torch_ops
    torch_ops.load_library()

    stream = torch.cuda.Stream(device=0)
    order = ("native", "custom") if options.worker_order == "native-first" \
        else ("custom", "native")
    cases = (("scalar", 1, 1), ("tail", 3, 17), ("small", 32, 128),
             ("model", 64, 1024), ("wide", 8, 4096))
    records = []
    for dtype_name, dtype, tolerance in (
            ("fp16", torch.float16, 5.0e-4),
            ("bf16", torch.bfloat16, 4.0e-3)):
        for shape_name, rows, width in cases:
            values = torch.arange(rows * width, device="cuda", dtype=torch.float32)
            owner_input = (((values % 251) * 0.03125 - 2)
                           .reshape(rows, width).to(dtype))

            def custom():
                return torch_ops.softmax(owner_input)

            def native():
                return torch.softmax(owner_input, dim=-1)

            timed = {}
            for policy in order:
                timed[policy] = time_policy(
                    native if policy == "native" else custom,
                    stream, options.warmup, options.repetitions)
            maximum, rms = errors(timed["custom"]["output"],
                                  timed["native"]["output"])
            records.append({
                "dtype": dtype_name, "shape": shape_name,
                "rows": rows, "width": width,
                "maximum_error": maximum, "rms_error": rms,
                "tolerance": tolerance,
                "output_distinct": (
                    timed["custom"]["output"].data_ptr() != owner_input.data_ptr()),
                "native_event_ms": timed["native"]["event_ms"],
                "custom_event_ms": timed["custom"]["event_ms"],
                "native_wall_ms": timed["native"]["wall_ms"],
                "custom_wall_ms": timed["custom"]["wall_ms"],
                "native_peak_extra_bytes": timed["native"]["peak_extra_bytes"],
                "custom_peak_extra_bytes": timed["custom"]["peak_extra_bytes"],
            })
    passed = all(row["maximum_error"] <= row["tolerance"] and
                 row["output_distinct"] for row in records)
    report = {
        "schema_version": 1,
        "record_type": "pytorch_rocm_custom_op_softmax_worker",
        "status": "pass" if passed else "fail",
        "run": options.worker_run,
        "order": options.worker_order,
        "architecture": torch.cuda.get_arch_list()[0],
        "records": records,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


def orchestrate(options: argparse.Namespace) -> int:
    if options.output is None:
        raise ValueError("--output is required")
    options.output.mkdir(parents=True, exist_ok=True)
    workers = []
    for run in range(1, options.runs + 1):
        for order in ("native-first", "custom-first"):
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--library", str(options.library), "--worker-order", order,
                "--worker-run", str(run), "--warmup", str(options.warmup),
                "--repetitions", str(options.repetitions),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker failed: {completed.stderr}\n{completed.stdout}")
            workers.append(json.loads(completed.stdout))
    (options.output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in workers),
        encoding="utf-8")
    keys = sorted({(row["dtype"], row["shape"], row["rows"], row["width"])
                   for worker in workers for row in worker["records"]})
    groups = []
    for key in keys:
        selected = [row for worker in workers for row in worker["records"]
                    if (row["dtype"], row["shape"], row["rows"], row["width"]) == key]
        groups.append({
            "dtype": key[0], "shape": key[1], "rows": key[2], "width": key[3],
            "processes": len(selected),
            "maximum_error": max(row["maximum_error"] for row in selected),
            "maximum_rms_error": max(row["rms_error"] for row in selected),
            "tolerance": selected[0]["tolerance"],
            "all_outputs_distinct": all(row["output_distinct"] for row in selected),
            "event_speedup_median": statistics.median(
                row["native_event_ms"] / row["custom_event_ms"] for row in selected),
            "wall_speedup_median": statistics.median(
                row["native_wall_ms"] / row["custom_wall_ms"] for row in selected),
            "native_event_ms_median": statistics.median(
                row["native_event_ms"] for row in selected),
            "custom_event_ms_median": statistics.median(
                row["custom_event_ms"] for row in selected),
            "native_peak_extra_bytes_median": statistics.median(
                row["native_peak_extra_bytes"] for row in selected),
            "custom_peak_extra_bytes_median": statistics.median(
                row["custom_peak_extra_bytes"] for row in selected),
        })
    passed = all(row["maximum_error"] <= row["tolerance"] and
                 row["all_outputs_distinct"] for row in groups)
    summary = {
        "schema_version": 1,
        "record_type": "pytorch_rocm_custom_op_softmax_matrix",
        "status": "pass" if passed else "fail",
        "correctness_pass": passed,
        "worker_processes": len(workers),
        "case_count": len(groups),
        "groups": groups,
    }
    (options.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    options = arguments()
    return worker(options) if options.worker_order else orchestrate(options)


if __name__ == "__main__":
    raise SystemExit(main())
