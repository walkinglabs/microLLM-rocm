#!/usr/bin/env python3
"""Fresh-process PyTorch ROCm Custom Op correctness and performance matrix."""

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


def synchronize() -> None:
    import torch
    torch.cuda.synchronize()


def time_forward(function, left, right, warmup: int, repetitions: int) -> dict:
    import torch
    output = None
    for _ in range(warmup):
        output = function(left, right)
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(repetitions):
        output = function(left, right)
    finish.record()
    finish.synchronize()
    wall_finish = time.perf_counter_ns()
    assert output is not None
    return {
        "event_ms": start.elapsed_time(finish) / repetitions,
        "wall_ms": (wall_finish - wall_start) / 1_000_000 / repetitions,
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - memory_before,
        "output": output,
    }


def time_backward(add, multiply, left_seed, right_seed,
                  warmup: int, repetitions: int) -> dict:
    import torch
    left = left_seed.detach().clone().requires_grad_()
    right = right_seed.detach().clone().requires_grad_()

    def execute():
        left.grad = None
        right.grad = None
        loss = (add(left, right) + multiply(left, right)).sum()
        loss.backward()
        return loss

    loss = None
    for _ in range(warmup):
        loss = execute()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(repetitions):
        loss = execute()
    finish.record()
    finish.synchronize()
    wall_finish = time.perf_counter_ns()
    assert loss is not None and left.grad is not None and right.grad is not None
    return {
        "event_ms": start.elapsed_time(finish) / repetitions,
        "wall_ms": (wall_finish - wall_start) / 1_000_000 / repetitions,
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - memory_before,
        "loss": float(loss.detach()),
        "left_gradient": left.grad.detach().clone(),
        "right_gradient": right.grad.detach().clone(),
    }


def error(actual, expected) -> tuple[float, float]:
    import torch
    difference = (actual.float() - expected.float()).abs()
    if difference.numel() == 0:
        return 0.0, 0.0
    return float(difference.max()), float(torch.sqrt(torch.mean(difference.square())))


def worker(args: argparse.Namespace) -> int:
    import torch
    from microllm import torch_ops

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm and one visible GPU are required")
    torch_ops.load_library(str(args.library))
    device = torch.device("cuda:0")
    dtype_rows = (("fp32", torch.float32),
                  ("fp16", torch.float16),
                  ("bf16", torch.bfloat16))
    shapes = (("launch", 4096), ("medium", 1 << 20), ("bandwidth", 1 << 24))
    custom = {"add": torch_ops.add, "multiply": torch_ops.multiply}
    reference = {"add": torch.add, "multiply": torch.multiply}
    order = ("torch", "microllm") if args.worker_order == "torch-first" \
        else ("microllm", "torch")
    records: list[dict] = []
    for dtype_name, dtype in dtype_rows:
        for shape_name, elements in shapes:
            values = torch.arange(elements, device=device, dtype=torch.float32)
            left = ((values % 251) * 0.00390625 - 0.5).to(dtype)
            right = ((values.flip(0) % 127) * -0.0078125 + 0.25).to(dtype)
            for operation in ("add", "multiply"):
                timed = {}
                for policy in order:
                    function = (reference if policy == "torch" else custom)[operation]
                    timed[policy] = time_forward(
                        function, left, right, args.warmup, args.repetitions)
                maximum, rms = error(timed["microllm"]["output"],
                                     timed["torch"]["output"])
                records.append({
                    "kind": "forward",
                    "operation": operation,
                    "dtype": dtype_name,
                    "shape": shape_name,
                    "elements": elements,
                    "maximum_error": maximum,
                    "rms_error": rms,
                    "torch_event_ms": timed["torch"]["event_ms"],
                    "microllm_event_ms": timed["microllm"]["event_ms"],
                    "torch_wall_ms": timed["torch"]["wall_ms"],
                    "microllm_wall_ms": timed["microllm"]["wall_ms"],
                    "torch_peak_extra_bytes": timed["torch"]["peak_extra_bytes"],
                    "microllm_peak_extra_bytes": timed["microllm"]["peak_extra_bytes"],
                })
            del values, left, right

    for shape_name, elements in (("medium", 1 << 16), ("large", 1 << 20)):
        values = torch.arange(elements, device=device, dtype=torch.float32)
        left = (values % 251) * 0.00390625 - 0.5
        right = (values.flip(0) % 127) * -0.0078125 + 0.25
        timed = {}
        for policy in order:
            add = torch.add if policy == "torch" else torch_ops.add
            multiply = torch.multiply if policy == "torch" else torch_ops.multiply
            timed[policy] = time_backward(
                add, multiply, left, right, args.warmup, args.repetitions)
        left_max, left_rms = error(
            timed["microllm"]["left_gradient"], timed["torch"]["left_gradient"])
        right_max, right_rms = error(
            timed["microllm"]["right_gradient"], timed["torch"]["right_gradient"])
        records.append({
            "kind": "forward_backward",
            "operation": "add_multiply_branch",
            "dtype": "fp32",
            "shape": shape_name,
            "elements": elements,
            "maximum_error": max(left_max, right_max),
            "rms_error": max(left_rms, right_rms),
            "loss_error": abs(timed["microllm"]["loss"] - timed["torch"]["loss"]),
            "torch_event_ms": timed["torch"]["event_ms"],
            "microllm_event_ms": timed["microllm"]["event_ms"],
            "torch_wall_ms": timed["torch"]["wall_ms"],
            "microllm_wall_ms": timed["microllm"]["wall_ms"],
            "torch_peak_extra_bytes": timed["torch"]["peak_extra_bytes"],
            "microllm_peak_extra_bytes": timed["microllm"]["peak_extra_bytes"],
        })
    info = {
        "schema_version": 1,
        "status": "pass" if all(row["maximum_error"] == 0.0 and
                                    row["rms_error"] == 0.0 and
                                    row.get("loss_error", 0.0) == 0.0
                                    for row in records) else "fail",
        "record_type": "pytorch_rocm_custom_op_worker",
        "run": args.worker_run,
        "order": args.worker_order,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.device_count() > 0
                        else "PyTorch ROCm default device"),
        "architecture": (torch.cuda.get_arch_list()[0]
                         if torch.cuda.get_arch_list() else "unknown"),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "records": records,
    }
    print(json.dumps(info, sort_keys=True))
    return 0 if info["status"] == "pass" else 2


def orchestrate(args: argparse.Namespace) -> int:
    if args.output is None:
        raise ValueError("--output is required in orchestrator mode")
    if args.runs < 1 or args.warmup < 1 or args.repetitions < 1:
        raise ValueError("runs, warmup and repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    workers = []
    for run in range(1, args.runs + 1):
        for order in ("torch-first", "microllm-first"):
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--library", str(args.library),
                "--worker-order", order, "--worker-run", str(run),
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker failed: {' '.join(command)}\n"
                    f"stdout={completed.stdout}\nstderr={completed.stderr}")
            workers.append(json.loads(completed.stdout))
    raw = args.output / "raw.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                               for row in workers), encoding="utf-8")

    keys = sorted({(row["kind"], row["operation"], row["dtype"],
                    row["shape"], row["elements"])
                   for worker_row in workers for row in worker_row["records"]})
    groups = []
    for key in keys:
        selected = [row for worker_row in workers for row in worker_row["records"]
                    if (row["kind"], row["operation"], row["dtype"],
                        row["shape"], row["elements"]) == key]
        event_speedups = [row["torch_event_ms"] / row["microllm_event_ms"]
                          for row in selected]
        wall_speedups = [row["torch_wall_ms"] / row["microllm_wall_ms"]
                         for row in selected]
        groups.append({
            "kind": key[0], "operation": key[1], "dtype": key[2],
            "shape": key[3], "elements": key[4], "processes": len(selected),
            "maximum_error": max(row["maximum_error"] for row in selected),
            "maximum_rms_error": max(row["rms_error"] for row in selected),
            "maximum_loss_error": max(row.get("loss_error", 0.0) for row in selected),
            "event_speedup_median": statistics.median(event_speedups),
            "event_speedup_minimum": min(event_speedups),
            "event_speedup_maximum": max(event_speedups),
            "wall_speedup_median": statistics.median(wall_speedups),
            "wall_speedup_minimum": min(wall_speedups),
            "wall_speedup_maximum": max(wall_speedups),
            "torch_peak_extra_bytes_median": statistics.median(
                row["torch_peak_extra_bytes"] for row in selected),
            "microllm_peak_extra_bytes_median": statistics.median(
                row["microllm_peak_extra_bytes"] for row in selected),
        })
    correctness = all(group["maximum_error"] == 0.0 and
                      group["maximum_rms_error"] == 0.0 and
                      group["maximum_loss_error"] == 0.0 for group in groups)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness else "fail",
        "record_type": "pytorch_rocm_custom_op_matrix",
        "correctness_pass": correctness,
        "worker_processes": len(workers),
        "case_count": len(groups),
        "orders": sorted({row["order"] for row in workers}),
        "torch_version": workers[0]["torch_version"],
        "torch_hip_version": workers[0]["torch_hip_version"],
        "device_name": workers[0]["device_name"],
        "architecture": workers[0]["architecture"],
        "groups": groups,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if correctness else 2


def main() -> int:
    args = arguments()
    return worker(args) if args.worker_order else orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
