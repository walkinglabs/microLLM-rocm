#!/usr/bin/env python3
"""Capture Python/ROCTX/HIP spans and merge a measured three-way timeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def capture(arguments: argparse.Namespace) -> int:
    from microllm import Tensor, hip_device_count
    from microllm.profiling import profile_scope, roctx_available

    if (arguments.device < 0 or arguments.elements <= 0 or
            arguments.iterations < 2 or arguments.sleep_ms < 0.0):
        raise ValueError(
            "device/sleep must be non-negative, elements positive, and iterations at least two")
    output = Path(arguments.output)
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"refusing to append to existing profile: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    if not roctx_available():
        raise RuntimeError("ROCTX runtime is unavailable")
    if hip_device_count() <= arguments.device:
        raise RuntimeError(f"HIP device {arguments.device} is unavailable")

    elements = arguments.elements
    left = Tensor.from_f32([1.0] * elements, (elements,)).to(
        f"hip:{arguments.device}")
    right = Tensor.from_f32([2.0] * elements, (elements,)).to(
        f"hip:{arguments.device}")
    warmup = left + right
    if warmup.tolist() != [3.0] * elements:
        raise RuntimeError("HIP warm-up add failed")
    roctx_warmup = output.with_name(output.stem + "-roctx-warmup.jsonl")
    if roctx_warmup.exists() and not arguments.overwrite:
        raise FileExistsError(f"refusing to replace ROCTX warm-up: {roctx_warmup}")
    roctx_warmup.write_text("", encoding="utf-8")
    with profile_scope("roctx.warmup", output=roctx_warmup,
                       run_id=arguments.run_id, emit_roctx=True):
        time.sleep(0.001)

    for iteration in range(arguments.iterations):
        with profile_scope(
                f"iteration.{iteration}", output=output,
                run_id=arguments.run_id, phase="python_outer",
                metadata={"iteration": iteration, "elements": elements},
                emit_roctx=True):
            with profile_scope(
                    f"host.delay.{iteration}", output=output,
                    run_id=arguments.run_id, phase="python_host"):
                time.sleep((arguments.sleep_ms + iteration) / 1000.0)
            with profile_scope(
                    f"hip.add.{iteration}", output=output,
                    run_id=arguments.run_id, phase="python_gpu_submit"):
                result = left + right
                values = result.tolist()
                if values[0] != 3.0 or values[-1] != 3.0:
                    raise RuntimeError("profiled HIP add failed")

    rows = sum(1 for line in output.read_text(encoding="utf-8").splitlines()
               if line)
    print(json.dumps({
        "schema_version": 1,
        "status": "pass",
        "profile_rows": rows,
        "roctx_ranges": arguments.iterations,
        "hip_adds": arguments.iterations,
        "output": str(output),
        "roctx_warmup_output": str(roctx_warmup),
    }, sort_keys=True))
    return 0


def merge(arguments: argparse.Namespace) -> int:
    from microllm.profiling import (calibrate_python_rocprof_clock,
                                    merge_rocprof_perfetto)

    calibration = calibrate_python_rocprof_clock(
        arguments.profile, arguments.marker, arguments.calibration)
    report = merge_rocprof_perfetto(
        arguments.marker, arguments.kernel, arguments.output,
        hip_api_csv=arguments.hip_api, python_jsonl=arguments.profile)
    print(json.dumps({
        "schema_version": 1,
        "status": "pass",
        "calibration": calibration,
        "merge": report,
    }, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--device", type=int, default=0)
    capture_parser.add_argument("--elements", type=int, default=4096)
    capture_parser.add_argument("--iterations", type=int, default=8)
    capture_parser.add_argument("--sleep-ms", type=float, default=5.0)
    capture_parser.add_argument("--run-id", default="python-roctx-hip")
    capture_parser.add_argument("--overwrite", action="store_true")
    capture_parser.set_defaults(handler=capture)

    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--profile", required=True)
    merge_parser.add_argument("--marker", required=True)
    merge_parser.add_argument("--kernel", required=True)
    merge_parser.add_argument("--hip-api", required=True)
    merge_parser.add_argument("--calibration", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(handler=merge)
    return result


def main() -> int:
    arguments = parser().parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
