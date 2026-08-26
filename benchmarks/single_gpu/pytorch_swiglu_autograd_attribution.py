#!/usr/bin/env python3
"""Attribute fused SwiGLU F+B cost to kernels, dispatcher, or Autograd."""

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
from pytorch_custom_op_rocm_matrix import error, synchronize  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--worker-first", choices=("native", "custom", "manual"))
    parser.add_argument("--worker-run", type=int, default=0)
    return parser.parse_args()


def time_autograd(function, gate_seed, up_seed, warmup: int, repetitions: int) -> dict:
    import torch
    gate = gate_seed.detach().clone().requires_grad_()
    up = up_seed.detach().clone().requires_grad_()

    def execute():
        gate.grad = None
        up.grad = None
        loss = function(gate, up).sum()
        loss.backward()
        return loss

    for _ in range(warmup):
        loss = execute()
    del loss
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(repetitions):
        loss = execute()
    finish.record(); finish.synchronize()
    wall_finish = time.perf_counter_ns()
    return {
        "event_ms": start.elapsed_time(finish) / repetitions,
        "wall_ms": (wall_finish - wall_start) / 1_000_000 / repetitions,
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - before,
        "loss": float(loss.detach()),
        "gate_gradient": gate.grad.detach().clone(),
        "up_gradient": up.grad.detach().clone(),
    }


def time_manual(custom, gate, up, warmup: int, repetitions: int) -> dict:
    import torch
    seed = torch.ones((), device=gate.device, dtype=gate.dtype)

    def execute():
        output = custom(gate, up)
        loss = output.sum()
        gradients = torch.ops.microllm.swiglu_backward_scalar_seed(
            gate, up, seed)
        return loss, gradients

    with torch.no_grad():
        for _ in range(warmup):
            loss, gradients = execute()
        del loss, gradients
        synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        for _ in range(repetitions):
            loss, gradients = execute()
        finish.record(); finish.synchronize()
        wall_finish = time.perf_counter_ns()
    return {
        "event_ms": start.elapsed_time(finish) / repetitions,
        "wall_ms": (wall_finish - wall_start) / 1_000_000 / repetitions,
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - before,
        "loss": float(loss),
        "gate_gradient": gradients[0].clone(),
        "up_gradient": gradients[1].clone(),
    }


def worker(options: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from microllm import torch_ops

    torch_ops.load_library(str(options.library))
    functions = {
        "native": lambda gate, up: F.silu(gate) * up,
        "custom": torch_ops.swiglu,
    }
    orders = {
        "native": ("native", "custom", "manual"),
        "custom": ("custom", "manual", "native"),
        "manual": ("manual", "native", "custom"),
    }
    records = []
    for shape, elements in (("medium", 65536), ("large", 1 << 20)):
        values = torch.arange(elements, device="cuda", dtype=torch.float32)
        gate = (values % 251) * 0.03125 - 2
        up = (values.flip(0) % 127) * -0.015625 + 1
        timed = {}
        for policy in orders[options.worker_first]:
            if policy == "manual":
                timed[policy] = time_manual(
                    torch_ops.swiglu, gate, up,
                    options.warmup, options.repetitions)
            else:
                timed[policy] = time_autograd(
                    functions[policy], gate, up,
                    options.warmup, options.repetitions)
        custom_gate_max, custom_gate_rms = error(
            timed["custom"]["gate_gradient"], timed["native"]["gate_gradient"])
        custom_up_max, custom_up_rms = error(
            timed["custom"]["up_gradient"], timed["native"]["up_gradient"])
        manual_gate_max, manual_gate_rms = error(
            timed["manual"]["gate_gradient"], timed["native"]["gate_gradient"])
        manual_up_max, manual_up_rms = error(
            timed["manual"]["up_gradient"], timed["native"]["up_gradient"])
        records.append({
            "shape": shape, "elements": elements,
            "custom_maximum_error": max(custom_gate_max, custom_up_max),
            "custom_rms_error": max(custom_gate_rms, custom_up_rms),
            "manual_maximum_error": max(manual_gate_max, manual_up_max),
            "manual_rms_error": max(manual_gate_rms, manual_up_rms),
            "custom_loss_error": abs(timed["custom"]["loss"] - timed["native"]["loss"]),
            "manual_loss_error": abs(timed["manual"]["loss"] - timed["native"]["loss"]),
            **{f"{policy}_{metric}": timed[policy][metric]
               for policy in ("native", "custom", "manual")
               for metric in ("event_ms", "wall_ms", "peak_extra_bytes")},
        })
    passed = all(max(row["custom_maximum_error"], row["manual_maximum_error"],
                         row["custom_loss_error"], row["manual_loss_error"]) <= 3.0e-6
                 for row in records)
    report = {
        "schema_version": 1, "status": "pass" if passed else "fail",
        "record_type": "pytorch_rocm_swiglu_autograd_attribution_worker",
        "run": options.worker_run, "first": options.worker_first,
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
        for first in ("native", "custom", "manual"):
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--library", str(options.library), "--worker-first", first,
                       "--worker-run", str(run), "--warmup", str(options.warmup),
                       "--repetitions", str(options.repetitions)]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(f"worker failed: {completed.stderr}\n{completed.stdout}")
            workers.append(json.loads(completed.stdout))
    (options.output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in workers),
        encoding="utf-8")
    groups = []
    for shape, elements in (("medium", 65536), ("large", 1 << 20)):
        selected = [row for worker_row in workers for row in worker_row["records"]
                    if row["shape"] == shape]
        group = {"shape": shape, "elements": elements, "processes": len(selected)}
        for field in ("custom_maximum_error", "custom_rms_error",
                      "manual_maximum_error", "manual_rms_error",
                      "custom_loss_error", "manual_loss_error"):
            group[field] = max(row[field] for row in selected)
        for policy in ("native", "custom", "manual"):
            for metric in ("event_ms", "wall_ms", "peak_extra_bytes"):
                group[f"{policy}_{metric}_median"] = statistics.median(
                    row[f"{policy}_{metric}"] for row in selected)
        group["manual_vs_custom_event"] = (
            group["custom_event_ms_median"] / group["manual_event_ms_median"])
        group["manual_vs_custom_wall"] = (
            group["custom_wall_ms_median"] / group["manual_wall_ms_median"])
        group["manual_vs_native_event"] = (
            group["native_event_ms_median"] / group["manual_event_ms_median"])
        groups.append(group)
    correctness = all(max(row["custom_maximum_error"], row["manual_maximum_error"],
                          row["custom_loss_error"], row["manual_loss_error"]) <= 3.0e-6
                      for row in groups)
    summary = {
        "schema_version": 1, "status": "pass" if correctness else "fail",
        "record_type": "pytorch_rocm_swiglu_autograd_attribution",
        "correctness_pass": correctness, "worker_processes": len(workers),
        "case_count": len(groups), "groups": groups,
    }
    (options.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if correctness else 2


def main() -> int:
    options = arguments()
    return worker(options) if options.worker_first else orchestrate(options)


if __name__ == "__main__":
    raise SystemExit(main())

