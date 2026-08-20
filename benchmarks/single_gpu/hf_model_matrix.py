#!/usr/bin/env python3
"""Run optional official Hugging Face inference/training memory benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


MODES = {"infer", "train"}
COMMON_FILES = ("config", "weights", "vocab", "merges")


def comma_list(text: str) -> list[str]:
    values = text.split(",")
    if not values or any(value not in MODES for value in values):
        raise argparse.ArgumentTypeError("modes must be infer, train, or infer,train")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("modes cannot contain duplicates")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--infer-binary", required=True, type=Path)
    parser.add_argument("--train-binary", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "hip"), default="hip")
    parser.add_argument("--modes", default="infer,train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-unavailable", action="store_true")
    result = parser.parse_args()
    result.modes = comma_list(result.modes)
    if not result.manifest.is_file():
        parser.error(f"manifest does not exist: {result.manifest}")
    return result


def load_manifest(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), list):
        raise RuntimeError("HF matrix manifest must have schema_version 1 and models list")
    names: set[str] = set()
    for model in document["models"]:
        required = {
            "name", "revision", "parameter_count", "loaded_tensors",
            *COMMON_FILES, "tokenizer_family", "inference", "training",
        }
        missing = required - model.keys()
        if missing:
            raise RuntimeError(f"HF model entry is missing fields: {sorted(missing)}")
        if not isinstance(model["name"], str) or not model["name"] or model["name"] in names:
            raise RuntimeError("HF model names must be non-empty and unique")
        names.add(model["name"])
        if int(model["parameter_count"]) <= 0 or int(model["loaded_tensors"]) <= 0:
            raise RuntimeError(f"{model['name']} has invalid expected counts")
    return document["models"]


def missing_inputs(model: dict) -> list[str]:
    return [field for field in COMMON_FILES if not Path(model[field]).is_file()]


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"command did not emit one JSON object: {completed.stdout!r}") from error


def validate_common(record: dict, model: dict, mode: str, device: str) -> None:
    if record.get("schema_version") != 1 or record.get("status") != "pass":
        raise RuntimeError(f"{model['name']}/{mode} returned a non-pass schema")
    if record.get("device") not in {device, f"{device}:0"}:
        raise RuntimeError(f"{model['name']}/{mode} returned the wrong device")
    if record.get("parameter_count") != model["parameter_count"]:
        raise RuntimeError(f"{model['name']}/{mode} parameter count changed")
    if record.get("loaded_tensors") != model["loaded_tensors"]:
        raise RuntimeError(f"{model['name']}/{mode} loaded Tensor count changed")
    if record.get("compute_dtype") != "float32":
        raise RuntimeError(f"{model['name']}/{mode} must state its current compute dtype")
    expected_weight_bytes = model["parameter_count"] * 4
    if record.get("fp32_weight_bytes") != expected_weight_bytes:
        raise RuntimeError(f"{model['name']}/{mode} FP32 weight bytes changed")
    numeric = ("engine_current_bytes", "engine_peak_bytes", "engine_total_allocated_bytes")
    if any(not math.isfinite(float(record.get(field, 0))) or record.get(field, 0) <= 0
           for field in numeric):
        raise RuntimeError(f"{model['name']}/{mode} has invalid memory metrics")
    if record["engine_peak_bytes"] < record["engine_current_bytes"]:
        raise RuntimeError(f"{model['name']}/{mode} peak memory is below current memory")
    if record["engine_peak_bytes"] < expected_weight_bytes:
        raise RuntimeError(f"{model['name']}/{mode} peak memory is below FP32 weights")


def infer(binary: Path, model: dict, device: str) -> dict:
    inference = model["inference"]
    command = [
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--vocab", model["vocab"], "--merges", model["merges"],
        "--tokenizer-family", model["tokenizer_family"], "--device", device,
        "--new-tokens", str(inference["new_tokens"]),
    ]
    prompt_kind = inference.get("prompt_kind", "text")
    if prompt_kind == "text":
        command.extend(("--text", inference["prompt"]))
    elif prompt_kind == "chat_user":
        command.extend(("--chat-user", inference["prompt"]))
    else:
        raise RuntimeError(f"{model['name']} has unknown prompt_kind {prompt_kind}")
    record = run_json(command)
    validate_common(record, model, "infer", device)
    positive = ("forward_ms", "prefill_tokens_per_second", "generation_ms",
                "decode_tokens_per_second", "decode_milliseconds_per_token")
    if any(not math.isfinite(float(record.get(field, 0))) or record.get(field, 0) <= 0
           for field in positive):
        raise RuntimeError(f"{model['name']}/infer has invalid timing metrics")
    expected_tokens = inference.get("expected_generated_tokens")
    if expected_tokens is not None and record.get("generated_tokens") != expected_tokens:
        raise RuntimeError(f"{model['name']}/infer generated tokens changed")
    return record


def train(binary: Path, model: dict, device: str) -> dict:
    training = model["training"]
    record = run_json([
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", training["tokens"], "--device", device,
        "--learning-rate", str(training["learning_rate"]),
    ])
    validate_common(record, model, "train", device)
    positive = ("step_ms", "tokens_per_second", "milliseconds_per_token")
    if any(not math.isfinite(float(record.get(field, 0))) or record.get(field, 0) <= 0
           for field in positive):
        raise RuntimeError(f"{model['name']}/train has invalid timing metrics")
    if not record.get("parameter_changed") or not math.isfinite(float(record.get("loss", math.nan))):
        raise RuntimeError(f"{model['name']}/train did not produce a finite update")
    if record.get("optimizer_host_to_device_calls") != 0 or \
       record.get("optimizer_device_to_host_calls") != 0:
        raise RuntimeError(f"{model['name']}/train optimizer copied Tensor payloads")
    return record


def unavailable(model: dict, mode: str, fields: list[str]) -> dict:
    return {
        "schema_version": 1,
        "record_type": "single_gpu_hf_model_measurement",
        "model": model["name"],
        "revision": model["revision"],
        "mode": mode,
        "status": "unavailable",
        "missing_inputs": fields,
    }


def main() -> int:
    args = options()
    models = load_manifest(args.manifest)
    records: list[dict] = []
    failures = 0
    unavailable_count = 0
    for model in models:
        missing = missing_inputs(model)
        for mode in args.modes:
            if missing:
                records.append(unavailable(model, mode, missing))
                unavailable_count += 1
                continue
            try:
                record = (infer(args.infer_binary, model, args.device) if mode == "infer"
                          else train(args.train_binary, model, args.device))
                record.update({
                    "record_type": "single_gpu_hf_model_measurement",
                    "model": model["name"],
                    "revision": model["revision"],
                    "mode": mode,
                    "fp32_weight_gib": record["fp32_weight_bytes"] / (1024**3),
                    "engine_peak_gib": record["engine_peak_bytes"] / (1024**3),
                })
                records.append(record)
            except Exception as error:  # Preserve a machine-readable failed row.
                records.append({
                    "schema_version": 1,
                    "record_type": "single_gpu_hf_model_measurement",
                    "model": model["name"],
                    "revision": model["revision"],
                    "mode": mode,
                    "status": "failed",
                    "error": str(error),
                })
                failures += 1
    status = "failed" if failures else "incomplete" if unavailable_count else "pass"
    summary = {
        "schema_version": 1,
        "record_type": "single_gpu_hf_model_matrix_summary",
        "status": status,
        "device": args.device,
        "model_count": len(models),
        "requested_modes": args.modes,
        "measurement_count": len(records) - unavailable_count - failures,
        "unavailable_count": unavailable_count,
        "failed_count": failures,
    }
    lines = [*(json.dumps(record, sort_keys=True) for record in records),
             json.dumps(summary, sort_keys=True)]
    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    sys.stdout.write(output)
    if failures:
        return 1
    if unavailable_count and not args.allow_unavailable:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
