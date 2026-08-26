#!/usr/bin/env python3
"""Compare eager, compiled, native, and manual fused SwiGLU F+B."""

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
from pytorch_swiglu_autograd_attribution import time_manual  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument(
        "--worker-first", choices=("native", "eager", "compiled", "manual"))
    parser.add_argument("--worker-run", type=int, default=0)
    return parser.parse_args()


def time_loss(function, gate_seed, up_seed, warmup: int, repetitions: int) -> dict:
    import torch
    gate = gate_seed.detach().clone().requires_grad_()
    up = up_seed.detach().clone().requires_grad_()

    def execute():
        gate.grad = None; up.grad = None
        loss = function(gate, up)
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
    wall_start = time.perf_counter_ns(); start.record()
    for _ in range(repetitions):
        loss = execute()
    finish.record(); finish.synchronize(); wall_finish = time.perf_counter_ns()
    return {
        "event_ms": start.elapsed_time(finish) / repetitions,
        "wall_ms": (wall_finish - wall_start) / 1_000_000 / repetitions,
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - before,
        "loss": float(loss.detach()),
        "gate_gradient": gate.grad.detach().clone(),
        "up_gradient": up.grad.detach().clone(),
    }


def compile_cold(function, gate_seed, up_seed) -> tuple[object, float]:
    import torch
    compiled = torch.compile(function, fullgraph=True)
    gate = gate_seed.detach().clone().requires_grad_()
    up = up_seed.detach().clone().requires_grad_()
    started = time.perf_counter_ns()
    loss = compiled(gate, up); loss.backward(); synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return compiled, elapsed_ms


def worker(options: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from microllm import torch_ops

    torch_ops.load_library(str(options.library))
    device_count_workaround = "none"
    if (torch.version.hip and torch.cuda.device_count() == 0 and
            torch._C._cuda_getDeviceCount() > 0):
        torch.cuda._device_count_amdsmi = lambda: -1
        device_count_workaround = "amdsmi_zero_fallback_to_hip_runtime"
    native_loss = lambda gate, up: (F.silu(gate) * up).sum()
    eager_loss = lambda gate, up: torch_ops.swiglu(gate, up).sum()
    order_base = ("native", "eager", "compiled", "manual")
    start = order_base.index(options.worker_first)
    order = order_base[start:] + order_base[:start]
    records = []
    for shape, elements in (("medium", 65536), ("large", 1 << 20)):
        values = torch.arange(elements, device="cuda", dtype=torch.float32)
        gate = (values % 251) * 0.03125 - 2
        up = (values.flip(0) % 127) * -0.015625 + 1
        compiled_loss, cold_ms = compile_cold(eager_loss, gate, up)
        timed = {}
        for policy in order:
            if policy == "manual":
                timed[policy] = time_manual(
                    torch_ops.swiglu, gate, up,
                    options.warmup, options.repetitions)
            else:
                timed[policy] = time_loss(
                    {"native": native_loss, "eager": eager_loss,
                     "compiled": compiled_loss}[policy],
                    gate, up, options.warmup, options.repetitions)
        records.append({
            "shape": shape, "elements": elements, "compile_cold_ms": cold_ms,
            **{f"{policy}_{metric}": timed[policy][metric]
               for policy in order_base
               for metric in ("event_ms", "wall_ms", "peak_extra_bytes")},
            **{f"{policy}_loss_error": abs(
                timed[policy]["loss"] - timed["native"]["loss"])
               for policy in ("eager", "compiled", "manual")},
            **{f"{policy}_maximum_error": max(
                error(timed[policy]["gate_gradient"],
                      timed["native"]["gate_gradient"])[0],
                error(timed[policy]["up_gradient"],
                      timed["native"]["up_gradient"])[0])
               for policy in ("eager", "compiled", "manual")},
        })
    passed = all(
        row[f"{policy}_maximum_error"] <= 3.0e-6 and
        row[f"{policy}_loss_error"] <= (4.0e-3 if policy == "compiled" else 3.0e-6)
        for row in records for policy in ("eager", "compiled", "manual"))
    report = {
        "schema_version": 1, "status": "pass" if passed else "fail",
        "record_type": "pytorch_rocm_swiglu_compile_worker",
        "run": options.worker_run, "first": options.worker_first,
        "device_count_workaround": device_count_workaround,
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
        for first in ("native", "eager", "compiled", "manual"):
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
    policies = ("native", "eager", "compiled", "manual")
    for shape, elements in (("medium", 65536), ("large", 1 << 20)):
        selected = [row for worker_row in workers for row in worker_row["records"]
                    if row["shape"] == shape]
        group = {
            "shape": shape, "elements": elements, "processes": len(selected),
            "compile_cold_ms_median": statistics.median(
                row["compile_cold_ms"] for row in selected),
        }
        for policy in policies:
            for metric in ("event_ms", "wall_ms", "peak_extra_bytes"):
                group[f"{policy}_{metric}_median"] = statistics.median(
                    row[f"{policy}_{metric}"] for row in selected)
        for policy in ("eager", "compiled", "manual"):
            group[f"{policy}_maximum_error"] = max(
                row[f"{policy}_maximum_error"] for row in selected)
            group[f"{policy}_loss_error"] = max(
                row[f"{policy}_loss_error"] for row in selected)
        group["compiled_vs_eager_event"] = (
            group["eager_event_ms_median"] / group["compiled_event_ms_median"])
        group["compiled_vs_native_event"] = (
            group["native_event_ms_median"] / group["compiled_event_ms_median"])
        group["manual_vs_compiled_event"] = (
            group["compiled_event_ms_median"] / group["manual_event_ms_median"])
        groups.append(group)
    correctness = all(
        row[f"{policy}_maximum_error"] <= 3.0e-6 and
        row[f"{policy}_loss_error"] <= (4.0e-3 if policy == "compiled" else 3.0e-6)
        for row in groups for policy in ("eager", "compiled", "manual"))
    compiled_gate = all(row["compiled_vs_eager_event"] >= 1.05 for row in groups)
    summary = {
        "schema_version": 1, "status": "pass" if correctness else "fail",
        "record_type": "pytorch_rocm_swiglu_compile_matrix",
        "correctness_pass": correctness, "compiled_gate_pass": compiled_gate,
        "decision": ("recommend_compiled_swiglu" if correctness and compiled_gate
                     else "reject_compiled_swiglu"),
        "worker_processes": len(workers), "case_count": len(groups),
        "groups": groups,
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
