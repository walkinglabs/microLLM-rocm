#!/usr/bin/env python3
"""Compare all official Qwen3 gate/up gradients and updated parameters."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open


EXPECTED_TENSORS = 56
EXPECTED_ELEMENTS = 176_160_768
LIMITS = {
    "fp32": {
        "gradient_max": 5.0e-3, "gradient_rms": 5.0e-5,
        "parameter_max": 1.0e-4, "parameter_rms": 1.0e-6,
    },
    "bf16": {
        "gradient_max": 5.0e-2, "gradient_rms": 1.0e-3,
        "parameter_max": 2.0e-4, "parameter_rms": 2.0e-6,
    },
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
    if (args.context <= 0 or args.learning_rate <= 0 or args.timeout_seconds <= 0):
        parser.error("context, learning rate and timeout must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def shaped_manifest(path: Path, model_name: str, context: int,
                    learning_rate: float) -> tuple[dict, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    selected = [model for model in models or [] if model.get("name") == model_name]
    if len(selected) != 1:
        raise RuntimeError(f"manifest must contain exactly one {model_name}")
    model = json.loads(json.dumps(selected[0]))
    seed = [int(value) for value in model["training"]["tokens"].split(",")]
    if len(seed) < 2 or any(token < 0 for token in seed):
        raise RuntimeError("training token seed is invalid")
    tokens = [seed[index % len(seed)] for index in range(context + 1)]
    model["training"].update({
        "tokens": ",".join(str(token) for token in tokens),
        "learning_rate": learning_rate, "batch": 1, "warmup": 0, "steps": 1,
    })
    return model, {"schema_version": 1, "models": [model]}


def run_json(command: list[str], timeout: int) -> dict:
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"worker emitted {len(lines)} JSON lines")
    return json.loads(lines[0])


def compare_files(candidate_path: Path, reference_path: Path,
                  kind: str) -> tuple[list[dict], dict]:
    records = []
    total_elements = 0
    total_squared = 0.0
    maximum = 0.0
    with safe_open(candidate_path, framework="pt", device="cpu") as candidate, \
            safe_open(reference_path, framework="pt", device="cpu") as reference:
        candidate_names = set(candidate.keys())
        reference_names = set(reference.keys())
        if candidate_names != reference_names or len(candidate_names) != EXPECTED_TENSORS:
            raise RuntimeError("gate/up exports have incompatible Tensor names")
        for name in sorted(candidate_names):
            left = candidate.get_tensor(name).float()
            right = reference.get_tensor(name).float()
            if left.shape != right.shape or left.numel() == 0:
                raise RuntimeError(f"gate/up shape mismatch: {name}")
            difference = left - right
            tensor_max = float(difference.abs().max())
            squared = float(torch.sum(difference.double() * difference.double()))
            tensor_rms = math.sqrt(squared / difference.numel())
            records.append({
                "schema_version": 1,
                "record_type": "qwen3_training_gate_up_tensor_comparison",
                "status": "pass", "kind": kind, "name": name,
                "shape": list(left.shape), "elements": left.numel(),
                "maximum_absolute_difference": tensor_max,
                "rms_difference": tensor_rms,
                "candidate_rms": float(torch.sqrt(torch.mean(left.double() ** 2))),
                "reference_rms": float(torch.sqrt(torch.mean(right.double() ** 2))),
                "all_finite": bool(torch.isfinite(left).all() and
                                   torch.isfinite(right).all()),
            })
            maximum = max(maximum, tensor_max)
            total_elements += difference.numel()
            total_squared += squared
    if total_elements != EXPECTED_ELEMENTS or not all(
            record["all_finite"] for record in records):
        raise RuntimeError("gate/up comparison element/finite contract failed")
    return records, {
        "tensor_count": len(records), "compared_elements": total_elements,
        "maximum_absolute_difference": maximum,
        "rms_difference": math.sqrt(total_squared / total_elements),
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    model, manifest = shaped_manifest(
        args.manifest, args.model, args.context, args.learning_rate)
    shaped_path = args.output_directory / "manifest.json"
    shaped_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    micro_grad = args.output_directory / "micro-gradients.safetensors"
    micro_param = args.output_directory / "micro-parameters.safetensors"
    torch_grad = args.output_directory / "pytorch-gradients.safetensors"
    torch_param = args.output_directory / "pytorch-parameters.safetensors"
    micro_command = [
        str(args.micro_binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens", model["training"]["tokens"],
        "--device", "hip", "--learning-rate", str(args.learning_rate),
        "--warmup", "0", "--steps", "1", "--batch", "1",
        "--linear-precision", args.precision,
        "--bf16-weight-mirrors", "true" if args.precision == "bf16" else "false",
        "--gate-up-gradients-output", str(micro_grad),
        "--gate-up-parameters-output", str(micro_param),
    ]
    pytorch_command = [
        str(args.pytorch_python), str(args.pytorch_runner),
        "--manifest", str(shaped_path), "--device", "cuda",
        "--dtype", "bf16_amp" if args.precision == "bf16" else "fp32",
        "--worker-model", model["name"], "--worker-mode", "train",
        "--gate-up-gradients-output", str(torch_grad),
        "--gate-up-parameters-output", str(torch_param),
    ]
    if args.allow_amdsmi_fallback:
        pytorch_command.append("--allow-amdsmi-fallback")
    micro = run_json(micro_command, args.timeout_seconds)
    pytorch = run_json(pytorch_command, args.timeout_seconds)
    for framework, record in (("microllm", micro), ("pytorch", pytorch)):
        if (record.get("status") != "pass" or
                record.get("gate_up_gradient_tensors") != EXPECTED_TENSORS or
                record.get("gate_up_gradient_elements") != EXPECTED_ELEMENTS or
                record.get("gate_up_parameter_tensors") != EXPECTED_TENSORS or
                record.get("gate_up_parameter_elements") != EXPECTED_ELEMENTS):
            raise RuntimeError(f"{framework} export contract failed")
    gradient_records, gradients = compare_files(micro_grad, torch_grad, "gradient")
    parameter_records, parameters = compare_files(micro_param, torch_param, "parameter")
    limits = LIMITS[args.precision]
    gates = {
        "loss_finite": math.isfinite(float(micro["loss"])) and
                       math.isfinite(float(pytorch["loss"])),
        "gradient_maximum": gradients["maximum_absolute_difference"] <=
                            limits["gradient_max"],
        "gradient_rms": gradients["rms_difference"] <= limits["gradient_rms"],
        "parameter_maximum": parameters["maximum_absolute_difference"] <=
                             limits["parameter_max"],
        "parameter_rms": parameters["rms_difference"] <= limits["parameter_rms"],
    }
    summary = {
        "schema_version": 1,
        "record_type": "qwen3_training_gate_up_audit",
        "status": "pass" if all(gates.values()) else "precision_mismatch",
        "model": model["name"], "revision": model["revision"],
        "precision": args.precision, "batch": 1, "context": args.context,
        "learning_rate": args.learning_rate,
        "microllm_loss": float(micro["loss"]),
        "pytorch_loss": float(pytorch["loss"]),
        "absolute_loss_difference": abs(float(micro["loss"]) -
                                        float(pytorch["loss"])),
        "gradients": gradients, "parameters": parameters,
        "limits": limits, "gates": gates,
        "microllm_peak_bytes": int(micro["engine_peak_bytes"]),
        "pytorch_peak_bytes": int(pytorch["device_peak_allocated_bytes"]),
        "boundary": (
            "all 56 gate/up gradients before AdamW and parameters after one "
            "step; remaining parameter families and multi-step trajectory are separate"),
    }
    raw = [*gradient_records, *parameter_records]
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
