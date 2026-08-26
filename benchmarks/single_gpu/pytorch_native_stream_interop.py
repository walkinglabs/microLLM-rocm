#!/usr/bin/env python3
"""Prove bidirectional ordering on a non-owning PyTorch ROCm Stream."""

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
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--run-id", default="pytorch-native-stream")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.device < 0 or arguments.size <= 0 or arguments.iterations <= 1:
        raise ValueError("device must be non-negative and workload dimensions positive")

    import torch
    from microllm import Event, Stream, Tensor, matmul, matmul_out
    from microllm.profiling import profile_scope

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is unavailable")
    profile = Path(arguments.profile)
    report_path = Path(arguments.report)
    for path in (profile, report_path):
        if path.exists() and not arguments.overwrite:
            raise FileExistsError(f"refusing to replace {path}")
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("", encoding="utf-8")

    size = arguments.size
    iterations = arguments.iterations
    expected = size * 0.001 * 0.002
    torch_stream = torch.cuda.Stream(device=arguments.device)
    external = Stream.from_external(
        int(torch_stream.cuda_stream), device=f"hip:{arguments.device}")
    if external.owning or external.native_handle != int(torch_stream.cuda_stream):
        raise RuntimeError("native Stream wrapper changed ownership or handle")

    torch_left = torch.full((size, size), 0.001, device="cuda", dtype=torch.float32)
    torch_right = torch.full((size, size), 0.002, device="cuda", dtype=torch.float32)
    torch_output = torch.empty_like(torch_left)
    with torch.cuda.stream(torch_stream):
        torch.mm(torch_left, torch_right, out=torch_output)
    torch_warm = torch.cuda.Event()
    torch_warm.record(torch_stream)
    torch_warm.synchronize()

    elements = size * size
    micro_left = Tensor.from_f32([0.001] * elements, (size, size)).to(
        f"hip:{arguments.device}")
    micro_right = Tensor.from_f32([0.002] * elements, (size, size)).to(
        f"hip:{arguments.device}")
    micro_output = matmul(micro_left, micro_right)
    if abs(micro_output.tolist()[0] - expected) > 1.0e-3:
        raise RuntimeError("microLLM matmul warm-up failed")
    matmul_out(micro_output, micro_left, micro_right, stream=external)
    micro_warm = Event(f"hip:{arguments.device}", enable_timing=False)
    micro_warm.record(external)
    micro_warm.synchronize()

    torch_finish_for_micro = Event(f"hip:{arguments.device}", enable_timing=False)
    with profile_scope("torch.to.microllm", output=profile,
                       run_id=arguments.run_id, emit_roctx=True):
        with torch.cuda.stream(torch_stream):
            for _ in range(iterations):
                torch.mm(torch_left, torch_right, out=torch_output)
        torch_finish_for_micro.record(external)
        torch_pending_for_micro = not torch_finish_for_micro.ready()
    wait_start = time.perf_counter_ns()
    torch_finish_for_micro.synchronize()
    torch_to_micro_wait_ns = time.perf_counter_ns() - wait_start

    micro_finish_for_torch = torch.cuda.Event()
    with profile_scope("microllm.to.torch", output=profile,
                       run_id=arguments.run_id, emit_roctx=True):
        for _ in range(iterations):
            matmul_out(micro_output, micro_left, micro_right, stream=external)
        micro_finish_for_torch.record(torch_stream)
        micro_pending_for_torch = not micro_finish_for_torch.query()
    wait_start = time.perf_counter_ns()
    micro_finish_for_torch.synchronize()
    micro_to_torch_wait_ns = time.perf_counter_ns() - wait_start

    torch_values = torch_output.flatten()[[0, -1]].cpu().tolist()
    micro_values = micro_output.tolist()
    maximum_error = max(abs(torch_values[0] - expected),
                        abs(torch_values[-1] - expected),
                        abs(micro_values[0] - expected),
                        abs(micro_values[-1] - expected))
    report = {
        "schema_version": 1,
        "status": "pass",
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "native_stream_handle": int(torch_stream.cuda_stream),
        "wrapper_owning": external.owning,
        "size": size,
        "iterations_per_direction": iterations,
        "torch_pending_for_microllm_event": torch_pending_for_micro,
        "microllm_pending_for_torch_event": micro_pending_for_torch,
        "torch_to_microllm_event_wait_ns": torch_to_micro_wait_ns,
        "microllm_to_torch_event_wait_ns": micro_to_torch_wait_ns,
        "maximum_output_error": maximum_error,
    }
    if (not torch_pending_for_micro or not micro_pending_for_torch or
            report["wrapper_owning"] or maximum_error > 1.0e-3):
        raise RuntimeError("PyTorch native Stream interop gate failed")
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    torch_finish_for_micro.close()
    micro_warm.close()
    external.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
