#!/usr/bin/env python3
"""Fresh-process fused SwiGLU PyTorch ROCm Custom Op matrix."""

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
from pytorch_custom_op_rocm_matrix import error, synchronize, time_forward  # noqa: E402


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


def tolerance(dtype_name: str, kind: str) -> float:
    if dtype_name == "fp32":
        return 3.0e-6 if kind == "forward_backward" else 1.0e-6
    if dtype_name == "fp16":
        return 4.0e-3
    return 6.25e-2


def time_backward(function, left_seed, right_seed,
                  warmup: int, repetitions: int) -> dict:
    import torch
    left = left_seed.detach().clone().requires_grad_()
    right = right_seed.detach().clone().requires_grad_()

    def execute():
        left.grad = None
        right.grad = None
        loss = function(left, right).sum()
        loss.backward()
        return loss

    for _ in range(warmup):
        loss = execute()
    del loss
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
    return {
        "event_ms": start.elapsed_time(finish) / repetitions,
        "wall_ms": (wall_finish - wall_start) / 1_000_000 / repetitions,
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - memory_before,
        "loss": float(loss.detach()),
        "left_gradient": left.grad.detach().clone(),
        "right_gradient": right.grad.detach().clone(),
    }


def worker(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from microllm import torch_ops

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm and one visible GPU are required")
    torch_ops.load_library(str(args.library))
    native = lambda gate, up: F.silu(gate) * up
    custom = torch_ops.swiglu
    functions = {"torch": native, "microllm": custom}
    order = ("torch", "microllm") if args.worker_order == "torch-first" \
        else ("microllm", "torch")
    records = []
    for dtype_name, dtype in (("fp32", torch.float32),
                              ("fp16", torch.float16),
                              ("bf16", torch.bfloat16)):
        for shape_name, elements in (("launch", 4096),
                                     ("medium", 1 << 20),
                                     ("bandwidth", 1 << 24)):
            values = torch.arange(elements, device="cuda", dtype=torch.float32)
            gate = ((values % 251) * 0.03125 - 2).to(dtype)
            up = ((values.flip(0) % 127) * -0.015625 + 1).to(dtype)
            timed = {policy: time_forward(
                functions[policy], gate, up, args.warmup, args.repetitions)
                     for policy in order}
            maximum, rms = error(timed["microllm"]["output"],
                                 timed["torch"]["output"])
            records.append({
                "kind": "forward", "dtype": dtype_name,
                "shape": shape_name, "elements": elements,
                "maximum_error": maximum, "rms_error": rms,
                "tolerance": tolerance(dtype_name, "forward"),
                "torch_event_ms": timed["torch"]["event_ms"],
                "microllm_event_ms": timed["microllm"]["event_ms"],
                "torch_wall_ms": timed["torch"]["wall_ms"],
                "microllm_wall_ms": timed["microllm"]["wall_ms"],
                "torch_peak_extra_bytes": timed["torch"]["peak_extra_bytes"],
                "microllm_peak_extra_bytes": timed["microllm"]["peak_extra_bytes"],
            })
            del values, gate, up

        for shape_name, elements in (("medium", 1 << 16), ("large", 1 << 20)):
            values = torch.arange(elements, device="cuda", dtype=torch.float32)
            gate = ((values % 251) * 0.03125 - 2).to(dtype)
            up = ((values.flip(0) % 127) * -0.015625 + 1).to(dtype)
            timed = {policy: time_backward(
                functions[policy], gate, up, args.warmup, args.repetitions)
                     for policy in order}
            left_max, left_rms = error(
                timed["microllm"]["left_gradient"], timed["torch"]["left_gradient"])
            right_max, right_rms = error(
                timed["microllm"]["right_gradient"], timed["torch"]["right_gradient"])
            records.append({
                "kind": "forward_backward", "dtype": dtype_name,
                "shape": shape_name, "elements": elements,
                "maximum_error": max(left_max, right_max),
                "rms_error": max(left_rms, right_rms),
                "loss_error": abs(timed["microllm"]["loss"] - timed["torch"]["loss"]),
                "tolerance": tolerance(dtype_name, "forward_backward"),
                "torch_event_ms": timed["torch"]["event_ms"],
                "microllm_event_ms": timed["microllm"]["event_ms"],
                "torch_wall_ms": timed["torch"]["wall_ms"],
                "microllm_wall_ms": timed["microllm"]["wall_ms"],
                "torch_peak_extra_bytes": timed["torch"]["peak_extra_bytes"],
                "microllm_peak_extra_bytes": timed["microllm"]["peak_extra_bytes"],
            })
            del values, gate, up
    passed = all(row["maximum_error"] <= row["tolerance"] and
                 row["rms_error"] <= row["tolerance"] and
                 row.get("loss_error", 0.0) <= row["tolerance"]
                 for row in records)
    report = {
        "schema_version": 1, "status": "pass" if passed else "fail",
        "record_type": "pytorch_rocm_custom_op_swiglu_worker",
        "run": args.worker_run, "order": args.worker_order,
        "torch_version": torch.__version__, "torch_hip_version": torch.version.hip,
        "architecture": torch.cuda.get_arch_list()[0],
        "warmup": args.warmup, "repetitions": args.repetitions,
        "records": records,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


def orchestrate(args: argparse.Namespace) -> int:
    if args.output is None:
        raise ValueError("--output is required")
    args.output.mkdir(parents=True, exist_ok=True)
    workers = []
    for run in range(1, args.runs + 1):
        for order in ("torch-first", "microllm-first"):
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--library", str(args.library), "--worker-order", order,
                       "--worker-run", str(run), "--warmup", str(args.warmup),
                       "--repetitions", str(args.repetitions)]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(f"worker failed: {completed.stderr}\n{completed.stdout}")
            workers.append(json.loads(completed.stdout))
    (args.output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in workers),
        encoding="utf-8")
    keys = sorted({(row["kind"], row["dtype"], row["shape"], row["elements"])
                   for worker_row in workers for row in worker_row["records"]})
    groups = []
    for item in keys:
        selected = [row for worker_row in workers for row in worker_row["records"]
                    if (row["kind"], row["dtype"], row["shape"], row["elements"]) == item]
        groups.append({
            "kind": item[0], "dtype": item[1], "shape": item[2],
            "elements": item[3], "processes": len(selected),
            "maximum_error": max(row["maximum_error"] for row in selected),
            "maximum_rms_error": max(row["rms_error"] for row in selected),
            "maximum_loss_error": max(row.get("loss_error", 0.0) for row in selected),
            "tolerance": selected[0]["tolerance"],
            "event_speedup_median": statistics.median(
                row["torch_event_ms"] / row["microllm_event_ms"] for row in selected),
            "event_speedup_minimum": min(
                row["torch_event_ms"] / row["microllm_event_ms"] for row in selected),
            "event_speedup_maximum": max(
                row["torch_event_ms"] / row["microllm_event_ms"] for row in selected),
            "wall_speedup_median": statistics.median(
                row["torch_wall_ms"] / row["microllm_wall_ms"] for row in selected),
            "torch_peak_extra_bytes_median": statistics.median(
                row["torch_peak_extra_bytes"] for row in selected),
            "microllm_peak_extra_bytes_median": statistics.median(
                row["microllm_peak_extra_bytes"] for row in selected),
        })
    correctness = all(row["maximum_error"] <= row["tolerance"] and
                      row["maximum_rms_error"] <= row["tolerance"] and
                      row["maximum_loss_error"] <= row["tolerance"] for row in groups)
    summary = {
        "schema_version": 1, "status": "pass" if correctness else "fail",
        "record_type": "pytorch_rocm_custom_op_swiglu_matrix",
        "correctness_pass": correctness, "worker_processes": len(workers),
        "case_count": len(groups), "architecture": workers[0]["architecture"],
        "torch_version": workers[0]["torch_version"],
        "torch_hip_version": workers[0]["torch_hip_version"],
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

