#!/usr/bin/env python3
"""PyTorch ROCm oracle/performance matrix for caller-owned FP16/BF16 Softmax."""

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
    parser.add_argument("--worker-order", choices=("torch-first", "microllm-first"))
    parser.add_argument("--worker-run", type=int, default=0)
    return parser.parse_args()


def wrap(tensor, device: str, dtype):
    from microllm import Tensor
    return Tensor.from_external(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(),
        tuple(tensor.shape), tuple(tensor.stride()), dtype=dtype,
        device=device, owner=tensor)


def errors(actual, expected) -> tuple[float, float]:
    import torch
    difference = actual.float() - expected.float()
    return float(difference.abs().max()), float(torch.sqrt(torch.mean(difference.square())))


def time_policy(function, stream_owner, warmup: int, repetitions: int) -> dict:
    import torch
    output = None
    with torch.cuda.stream(stream_owner):
        for _ in range(warmup):
            output = function()
    stream_owner.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    with torch.cuda.stream(stream_owner):
        start.record(stream_owner)
        for _ in range(repetitions):
            output = function()
        finish.record(stream_owner)
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
    os.environ["MICROLLM_LIBRARY"] = str(options.library)
    import torch
    from microllm import DType, Stream, softmax_out

    stream_owner = torch.cuda.Stream(device=0)
    stream = Stream.from_external(int(stream_owner.cuda_stream), device="hip:0")
    order = ("torch", "microllm") if options.worker_order == "torch-first" \
        else ("microllm", "torch")
    records = []
    cases = (("scalar", 1, 1), ("tail", 3, 17), ("small", 32, 128),
             ("model", 64, 1024), ("wide", 8, 4096))
    for dtype_name, torch_dtype, micro_dtype, tolerance in (
            ("fp16", torch.float16, DType.FLOAT16, 5.0e-4),
            ("bf16", torch.bfloat16, DType.BFLOAT16, 4.0e-3)):
        for shape_name, rows, width in cases:
            values = torch.arange(rows * width, device="cuda", dtype=torch.float32)
            owner_input = (((values % 251) * 0.03125 - 2)
                           .reshape(rows, width).to(torch_dtype))
            owner_output = torch.empty_like(owner_input)
            input_view = wrap(owner_input, "hip:0", micro_dtype)
            output_view = wrap(owner_output, "hip:0", micro_dtype)

            def micro():
                softmax_out(output_view, input_view, stream=stream)
                return owner_output

            def native():
                return torch.softmax(owner_input, dim=-1)

            timed = {}
            for policy in order:
                timed[policy] = time_policy(
                    native if policy == "torch" else micro,
                    stream_owner, options.warmup, options.repetitions)
            maximum, rms = errors(timed["microllm"]["output"],
                                  timed["torch"]["output"])
            records.append({
                "dtype": dtype_name, "shape": shape_name,
                "rows": rows, "width": width,
                "maximum_error": maximum, "rms_error": rms,
                "tolerance": tolerance,
                "input_pointer_matches": input_view.data_ptr == owner_input.data_ptr(),
                "output_pointer_matches": output_view.data_ptr == owner_output.data_ptr(),
                "wrappers_non_owning": not input_view.owning and not output_view.owning,
                "torch_event_ms": timed["torch"]["event_ms"],
                "microllm_event_ms": timed["microllm"]["event_ms"],
                "torch_wall_ms": timed["torch"]["wall_ms"],
                "microllm_wall_ms": timed["microllm"]["wall_ms"],
                "torch_peak_extra_bytes": timed["torch"]["peak_extra_bytes"],
                "microllm_peak_extra_bytes": timed["microllm"]["peak_extra_bytes"],
            })
            input_view.close(); output_view.close()
    stream.close()
    passed = all(row["maximum_error"] <= row["tolerance"] and
                 row["input_pointer_matches"] and row["output_pointer_matches"] and
                 row["wrappers_non_owning"] for row in records)
    report = {
        "schema_version": 1, "status": "pass" if passed else "fail",
        "record_type": "pytorch_rocm_typed_softmax_worker",
        "run": options.worker_run, "order": options.worker_order,
        "architecture": torch.cuda.get_arch_list()[0], "records": records,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


def orchestrate(options: argparse.Namespace) -> int:
    if options.output is None:
        raise ValueError("--output is required")
    options.output.mkdir(parents=True, exist_ok=True)
    workers = []
    for run in range(1, options.runs + 1):
        for order in ("torch-first", "microllm-first"):
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--library", str(options.library), "--worker-order", order,
                       "--worker-run", str(run), "--warmup", str(options.warmup),
                       "--repetitions", str(options.repetitions)]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(f"worker failed: {completed.stderr}\n{completed.stdout}")
            workers.append(json.loads(completed.stdout))
    (options.output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in workers),
        encoding="utf-8")
    keys = sorted({(row["dtype"], row["shape"], row["rows"], row["width"])
                   for worker in workers for row in worker["records"]})
    groups = []
    for item in keys:
        selected = [row for worker in workers for row in worker["records"]
                    if (row["dtype"], row["shape"], row["rows"], row["width"]) == item]
        groups.append({
            "dtype": item[0], "shape": item[1], "rows": item[2], "width": item[3],
            "processes": len(selected),
            "maximum_error": max(row["maximum_error"] for row in selected),
            "maximum_rms_error": max(row["rms_error"] for row in selected),
            "tolerance": selected[0]["tolerance"],
            "all_pointers_match": all(row["input_pointer_matches"] and
                                      row["output_pointer_matches"] for row in selected),
            "all_wrappers_non_owning": all(row["wrappers_non_owning"] for row in selected),
            "event_speedup_median": statistics.median(
                row["torch_event_ms"] / row["microllm_event_ms"] for row in selected),
            "wall_speedup_median": statistics.median(
                row["torch_wall_ms"] / row["microllm_wall_ms"] for row in selected),
            "torch_peak_extra_bytes_median": statistics.median(
                row["torch_peak_extra_bytes"] for row in selected),
            "microllm_peak_extra_bytes_median": statistics.median(
                row["microllm_peak_extra_bytes"] for row in selected),
        })
    correctness = all(row["maximum_error"] <= row["tolerance"] and
                      row["all_pointers_match"] and row["all_wrappers_non_owning"]
                      for row in groups)
    summary = {
        "schema_version": 1, "status": "pass" if correctness else "fail",
        "record_type": "pytorch_rocm_typed_softmax_matrix",
        "correctness_pass": correctness, "worker_processes": len(workers),
        "case_count": len(groups), "groups": groups,
    }
    (options.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if correctness else 2


def main() -> int:
    options = arguments()
    return worker(options) if options.worker_order else orchestrate(options)


if __name__ == "__main__":
    raise SystemExit(main())
