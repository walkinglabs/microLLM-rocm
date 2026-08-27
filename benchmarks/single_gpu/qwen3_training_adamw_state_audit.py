#!/usr/bin/env python3
"""Compare every official Qwen3 AdamW first/second moment after one step."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

try:
    from .qwen3_training_all_parameter_audit import (
        parameter_family, run_json, shaped_manifest)
except ImportError:
    from qwen3_training_all_parameter_audit import (
        parameter_family, run_json, shaped_manifest)


EXPECTED_TENSORS = 620
EXPECTED_ELEMENTS = 1_192_099_840
LIMITS = {
    "fp32": {"maximum": 1.0e-3, "rms": 1.0e-6},
    "bf16": {"maximum": 1.0e-2, "rms": 1.0e-4},
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--micro-binary", type=Path, required=True)
    parser.add_argument("--pytorch-python", type=Path, required=True)
    parser.add_argument("--pytorch-runner", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    args = parser.parse_args()
    for path in (args.manifest, args.micro_binary, args.pytorch_python,
                 args.pytorch_runner):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def split_name(name: str) -> tuple[str, str]:
    for suffix in ("first_moment", "second_moment"):
        marker = f".adamw.{suffix}"
        if name.endswith(marker):
            return name[:-len(marker)], suffix
    raise RuntimeError(f"invalid AdamW state name: {name}")


def compare(candidate_path: Path, reference_path: Path) -> tuple[list[dict], dict]:
    records = []
    total_elements = 0
    total_squared = 0.0
    maximum = 0.0
    worst = ""
    grouped = {}
    with safe_open(candidate_path, framework="pt", device="cpu") as candidate, \
            safe_open(reference_path, framework="pt", device="cpu") as reference:
        names = set(candidate.keys())
        if names != set(reference.keys()) or len(names) != EXPECTED_TENSORS:
            raise RuntimeError("AdamW exports have incompatible names")
        for name in sorted(names):
            parameter, moment = split_name(name)
            left = candidate.get_tensor(name).float()
            right = reference.get_tensor(name).float()
            if left.shape != right.shape or left.numel() == 0:
                raise RuntimeError(f"AdamW state shape mismatch: {name}")
            difference = left - right
            finite = bool(torch.isfinite(left).all() and
                          torch.isfinite(right).all() and
                          torch.isfinite(difference).all())
            tensor_max = float(difference.abs().max())
            squared = float(torch.sum(difference.double() ** 2))
            tensor_rms = math.sqrt(squared / difference.numel())
            family = parameter_family(parameter)
            records.append({
                "schema_version": 1,
                "record_type": "qwen3_training_adamw_tensor_comparison",
                "status": "pass" if finite else "nonfinite",
                "name": name, "parameter": parameter, "moment": moment,
                "family": family, "shape": list(left.shape),
                "elements": left.numel(),
                "maximum_absolute_difference": tensor_max,
                "rms_difference": tensor_rms, "all_finite": finite,
            })
            key = f"{family}:{moment}"
            group = grouped.setdefault(key, {
                "family": family, "moment": moment, "tensor_count": 0,
                "compared_elements": 0, "maximum_absolute_difference": 0.0,
                "squared": 0.0, "worst_tensor": "", "all_finite": True,
            })
            group["tensor_count"] += 1
            group["compared_elements"] += difference.numel()
            group["squared"] += squared
            group["all_finite"] = group["all_finite"] and finite
            if tensor_max > group["maximum_absolute_difference"]:
                group["maximum_absolute_difference"] = tensor_max
                group["worst_tensor"] = name
            if tensor_max > maximum:
                maximum, worst = tensor_max, name
            total_elements += difference.numel()
            total_squared += squared
    if total_elements != EXPECTED_ELEMENTS or not all(r["all_finite"] for r in records):
        raise RuntimeError("AdamW element/finite contract failed")
    groups = []
    for group in grouped.values():
        group["rms_difference"] = math.sqrt(
            group.pop("squared") / group["compared_elements"])
        groups.append(group)
    groups.sort(key=lambda row: (row["family"], row["moment"]))
    return records, {
        "tensor_count": len(records), "compared_elements": total_elements,
        "maximum_absolute_difference": maximum,
        "rms_difference": math.sqrt(total_squared / total_elements),
        "worst_tensor": worst, "groups": groups,
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    model, manifest = shaped_manifest(
        args.manifest, args.model, args.context, args.learning_rate)
    shaped = args.output_directory / "manifest.json"
    shaped.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    micro_file = args.output_directory / "micro-moments.safetensors"
    torch_file = args.output_directory / "pytorch-moments.safetensors"
    micro_command = [
        str(args.micro_binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens", model["training"]["tokens"],
        "--device", "hip", "--learning-rate", str(args.learning_rate),
        "--warmup", "0", "--steps", "1", "--batch", "1",
        "--linear-precision", args.precision,
        "--bf16-weight-mirrors", "true" if args.precision == "bf16" else "false",
        "--adamw-moment-precision", "fp32",
        "--all-moments-output", str(micro_file),
    ]
    torch_command = [
        str(args.pytorch_python), str(args.pytorch_runner),
        "--manifest", str(shaped), "--device", "cuda",
        "--dtype", "bf16_amp" if args.precision == "bf16" else "fp32",
        "--worker-model", model["name"], "--worker-mode", "train",
        "--all-moments-output", str(torch_file),
    ]
    if args.allow_amdsmi_fallback:
        torch_command.append("--allow-amdsmi-fallback")
    micro = run_json(micro_command, args.timeout_seconds)
    pytorch = run_json(torch_command, args.timeout_seconds)
    for framework, worker in (("microllm", micro), ("pytorch", pytorch)):
        if (worker.get("all_moment_tensors") != EXPECTED_TENSORS or
                worker.get("all_moment_elements") != EXPECTED_ELEMENTS or
                worker.get("all_moment_step") != 1):
            raise RuntimeError(f"{framework} AdamW export contract failed")
    raw, comparison = compare(micro_file, torch_file)
    limits = LIMITS[args.precision]
    gates = {
        "step": micro["all_moment_step"] == pytorch["all_moment_step"] == 1,
        "maximum": comparison["maximum_absolute_difference"] <= limits["maximum"],
        "rms": comparison["rms_difference"] <= limits["rms"],
    }
    export_bytes = micro_file.stat().st_size + torch_file.stat().st_size
    micro_file.unlink()
    torch_file.unlink()
    summary = {
        "schema_version": 1, "record_type": "qwen3_training_adamw_state_audit",
        "status": "pass" if all(gates.values()) else "precision_mismatch",
        "model": model["name"], "revision": model["revision"],
        "precision": args.precision, "context": args.context, "batch": 1,
        "moment_storage": "fp32", "step": 1, "comparison": comparison,
        "limits": limits, "gates": gates,
        "temporary_export_bytes": export_bytes,
        "temporary_exports_removed": not micro_file.exists() and not torch_file.exists(),
        "boundary": "all first/second moments after one AdamW step; parameters and gradients are covered by Experiment 383",
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "workers.json").write_text(
        json.dumps({"microllm": micro, "pytorch": pytorch}, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
