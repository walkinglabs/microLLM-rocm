#!/usr/bin/env python3
"""Run a reproducible built-in single-GPU model memory/performance matrix."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "tiny": {
        "parameter_count": 5_712,
        "train": {"batch": 1, "context": 8, "steps": 3, "warmup": 1, "new_tokens": 8},
        "generate": {"batch": 1, "context": 8, "steps": 3, "warmup": 1, "new_tokens": 8},
    },
    "model-s": {
        "parameter_count": 15_586_176,
        "train": {"batch": 1, "context": 2, "steps": 1, "warmup": 0, "new_tokens": 2},
        "generate": {"batch": 1, "context": 4, "steps": 1, "warmup": 0, "new_tokens": 2},
    },
    "model-m": {
        "parameter_count": 31_334_912,
        "train": {"batch": 1, "context": 1, "steps": 1, "warmup": 0, "new_tokens": 2},
        "generate": {"batch": 1, "context": 4, "steps": 1, "warmup": 0, "new_tokens": 2},
    },
}

REQUIRED_FIELDS = {
    "schema_version",
    "mode",
    "model",
    "device",
    "device_name",
    "architecture",
    "dtype",
    "parameter_count",
    "fp32_weight_bytes",
    "measured_tokens",
    "measured_wall_seconds",
    "tokens_per_second",
    "milliseconds_per_token",
    "model_construction_seconds",
    "warmup_seconds",
    "wall_seconds_with_setup",
    "tokens_per_second_with_setup",
    "device_current_engine_bytes",
    "device_peak_engine_bytes",
    "device_total_allocated_engine_bytes",
    "output_guard",
}


def comma_list(text: str, allowed: set[str], name: str) -> list[str]:
    values = text.split(",")
    if not values or any(not value or value not in allowed for value in values):
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated subset of {','.join(sorted(allowed))}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} cannot contain duplicates")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "hip"), default="hip")
    parser.add_argument("--profiles", default="tiny,model-s,model-m")
    parser.add_argument("--modes", default="train,generate")
    parser.add_argument("--output", type=Path)
    result = parser.parse_args()
    result.profiles = comma_list(result.profiles, set(PROFILES), "profiles")
    result.modes = comma_list(result.modes, {"train", "generate"}, "modes")
    if not result.benchmark.is_file():
        parser.error(f"benchmark executable does not exist: {result.benchmark}")
    return result


def run_one(binary: Path, device: str, profile: str, mode: str) -> dict:
    settings = PROFILES[profile][mode]
    command = [
        str(binary),
        "--mode", mode,
        "--model", profile,
        "--device", device,
        "--steps", str(settings["steps"]),
        "--warmup", str(settings["warmup"]),
        "--batch", str(settings["batch"]),
        "--context", str(settings["context"]),
        "--new-tokens", str(settings["new_tokens"]),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{profile}/{mode} did not emit one JSON object: {completed.stdout!r}"
        ) from error
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise RuntimeError(f"{profile}/{mode} is missing fields: {sorted(missing)}")
    if record["schema_version"] != 1 or record["model"] != profile:
        raise RuntimeError(f"{profile}/{mode} returned the wrong schema or model")
    if record["mode"] != mode or record["device"] != device:
        raise RuntimeError(f"{profile}/{mode} returned the wrong mode or device")
    expected_parameters = PROFILES[profile]["parameter_count"]
    if record["parameter_count"] != expected_parameters:
        raise RuntimeError(
            f"{profile} parameter count {record['parameter_count']} != {expected_parameters}"
        )
    expected_weight_bytes = expected_parameters * 4
    if record["fp32_weight_bytes"] != expected_weight_bytes:
        raise RuntimeError(f"{profile} reported the wrong FP32 weight byte count")
    positive = (
        "measured_tokens",
        "measured_wall_seconds",
        "tokens_per_second",
        "milliseconds_per_token",
        "wall_seconds_with_setup",
        "tokens_per_second_with_setup",
        "device_current_engine_bytes",
        "device_peak_engine_bytes",
        "device_total_allocated_engine_bytes",
    )
    if any(not math.isfinite(float(record[field])) or float(record[field]) <= 0 for field in positive):
        raise RuntimeError(f"{profile}/{mode} contains a non-positive or non-finite metric")
    if record["device_peak_engine_bytes"] < record["device_current_engine_bytes"]:
        raise RuntimeError(f"{profile}/{mode} peak memory is below current memory")
    if record["device_peak_engine_bytes"] < expected_weight_bytes:
        raise RuntimeError(f"{profile}/{mode} peak memory is below its FP32 weights")
    if not math.isfinite(float(record["output_guard"])):
        raise RuntimeError(f"{profile}/{mode} output guard is not finite")
    if mode == "train" and not math.isfinite(float(record["final_loss"])):
        raise RuntimeError(f"{profile}/train loss is not finite")

    record.update(
        {
            "record_type": "single_gpu_model_measurement",
            "matrix_profile": profile,
            "status": "pass",
            "fp32_weight_gib": record["fp32_weight_bytes"] / (1024**3),
            "device_peak_engine_gib": record["device_peak_engine_bytes"] / (1024**3),
            "peak_to_weight_ratio": (
                record["device_peak_engine_bytes"] / record["fp32_weight_bytes"]
            ),
        }
    )
    return record


def main() -> int:
    args = options()
    records = [
        run_one(args.benchmark, args.device, profile, mode)
        for profile in args.profiles
        for mode in args.modes
    ]
    summary = {
        "schema_version": 1,
        "record_type": "single_gpu_model_matrix_summary",
        "status": "pass",
        "device": args.device,
        "profiles": args.profiles,
        "modes": args.modes,
        "measurement_count": len(records),
        "note": "performance is descriptive; no unstable speed threshold is enforced",
    }
    lines = [*(json.dumps(record, sort_keys=True) for record in records),
             json.dumps(summary, sort_keys=True)]
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
