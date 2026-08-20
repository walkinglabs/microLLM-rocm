#!/usr/bin/env python3
"""Run paired official-model training across batch/context shapes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_shapes(text: str) -> list[tuple[int, int]]:
    result = []
    for item in text.split(","):
        fields = item.lower().split("x")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError("shapes must look like 1x32,2x32")
        try:
            batch, context = (int(value) for value in fields)
        except ValueError as error:
            raise argparse.ArgumentTypeError("shape values must be integers") from error
        if batch <= 0 or context <= 0:
            raise argparse.ArgumentTypeError("shape values must be positive")
        pair = (batch, context)
        if pair in result:
            raise argparse.ArgumentTypeError("shapes cannot contain duplicates")
        result.append(pair)
    if not result:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return result


def comma_names(text: str) -> list[str]:
    result = text.split(",")
    if not result or any(not name for name in result) or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("models must be unique comma-separated names")
    return result


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--micro-binary", required=True, type=Path)
    parser.add_argument("--pytorch-python", required=True, type=Path)
    parser.add_argument("--pytorch-runner", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models", type=comma_names)
    parser.add_argument("--shapes", type=parse_shapes, default=parse_shapes("1x32,2x32,1x128"))
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    result = parser.parse_args()
    for path in (result.manifest, result.micro_binary, result.pytorch_python,
                 result.pytorch_runner):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if result.warmup < 0 or result.steps <= 0 or result.runs <= 0 or \
            result.timeout_seconds <= 0:
        parser.error("warmup must be nonnegative; steps/runs/timeout must be positive")
    return result


def load_models(path: Path, selected: list[str] | None = None) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list) or not models:
        raise RuntimeError("manifest must contain a non-empty schema-version-1 models list")
    by_name = {model.get("name"): model for model in models}
    if len(by_name) != len(models) or None in by_name:
        raise RuntimeError("manifest model names must be present and unique")
    names = selected or list(by_name)
    missing = set(names) - set(by_name)
    if missing:
        raise RuntimeError(f"unknown selected models: {sorted(missing)}")
    result = []
    for name in names:
        model = by_name[name]
        required = {"revision", "parameter_count", "loaded_tensors", "config",
                    "weights", "inference", "training"}
        absent = required - model.keys()
        if absent:
            raise RuntimeError(f"{name} is missing fields: {sorted(absent)}")
        seed = [int(value) for value in str(model["training"].get("tokens", "")).split(",")]
        if len(seed) < 2 or any(value < 0 for value in seed):
            raise RuntimeError(f"{name} needs at least two nonnegative training token IDs")
        result.append(model)
    return result


def expanded_tokens(seed_text: str, context: int) -> str:
    seed = [int(value) for value in seed_text.split(",")]
    return ",".join(str(seed[index % len(seed)]) for index in range(context + 1))


def run_json(command: list[str], timeout: int) -> dict:
    completed = subprocess.run(command, check=True, capture_output=True, text=True,
                               timeout=timeout)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"command emitted {len(lines)} non-empty lines instead of one")
    return json.loads(lines[0])


def validate_record(record: dict, model: dict, framework: str, batch: int,
                    context: int, warmup: int, steps: int) -> None:
    expected_tokens = batch * context * steps
    if record.get("status") != "pass" or record.get("parameter_count") != \
            model["parameter_count"] or record.get("batch") != batch or \
            record.get("context") != context or record.get("warmup") != warmup or \
            record.get("steps") != steps or record.get("trained_tokens") != expected_tokens:
        raise RuntimeError(f"{model['name']} {framework} returned a mismatched shape contract")
    numeric = ("tokens_per_second", "mean_step_ms", "loss")
    if any(not math.isfinite(float(record.get(field, math.nan))) for field in numeric) or \
            float(record["tokens_per_second"]) <= 0 or not record.get("parameter_changed"):
        raise RuntimeError(f"{model['name']} {framework} returned invalid training evidence")
    memory_field = "engine_peak_bytes" if framework == "microllm" \
        else "device_peak_allocated_bytes"
    if int(record.get(memory_field, 0)) <= 0:
        raise RuntimeError(f"{model['name']} {framework} returned invalid peak memory")


def micro_command(args: argparse.Namespace, model: dict, batch: int,
                  token_text: str) -> list[str]:
    command = [
        str(args.micro_binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", token_text, "--device", "hip",
        "--learning-rate", str(model["training"]["learning_rate"]),
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--batch", str(batch), "--linear-precision", args.precision,
    ]
    if args.precision == "bf16":
        command.extend(("--bf16-weight-mirrors", "true"))
    return command


def pytorch_command(args: argparse.Namespace, manifest: Path, model: dict) -> list[str]:
    command = [
        str(args.pytorch_python), str(args.pytorch_runner), "--manifest", str(manifest),
        "--device", "cuda", "--dtype", "bf16_amp" if args.precision == "bf16" else "fp32",
        "--worker-model", model["name"], "--worker-mode", "train",
    ]
    if args.allow_amdsmi_fallback:
        command.append("--allow-amdsmi-fallback")
    return command


def median(records: list[dict], field: str) -> float:
    return statistics.median(float(record[field]) for record in records)


def summarize(records: list[dict], models: list[dict], shapes: list[tuple[int, int]],
              precision: str, runs: int) -> dict:
    rows = []
    for model in models:
        for batch, context in shapes:
            selected = [record for record in records if record.get("model") == model["name"] and
                        record.get("batch") == batch and record.get("context") == context and
                        record.get("status") == "pass"]
            micro = [record for record in selected if record["framework"] == "microllm"]
            torch = [record for record in selected if record["framework"] == "pytorch"]
            if len(micro) != runs or len(torch) != runs:
                rows.append({"model": model["name"], "batch": batch, "context": context,
                             "status": "incomplete", "microllm_runs": len(micro),
                             "pytorch_runs": len(torch)})
                continue
            micro_tps = median(micro, "tokens_per_second")
            torch_tps = median(torch, "tokens_per_second")
            micro_peak = median(micro, "peak_bytes")
            torch_peak = median(torch, "peak_bytes")
            rows.append({
                "model": model["name"], "revision": model["revision"],
                "batch": batch, "context": context, "status": "pass",
                "microllm_tokens_per_second": micro_tps,
                "pytorch_tokens_per_second": torch_tps,
                "throughput_ratio_microllm_over_pytorch": micro_tps / torch_tps,
                "microllm_peak_bytes": micro_peak,
                "pytorch_peak_allocated_bytes": torch_peak,
                "peak_memory_ratio": micro_peak / torch_peak,
                "microllm_final_loss": median(micro, "final_loss"),
                "pytorch_final_loss": median(torch, "final_loss"),
            })
    return {"schema_version": 1, "track": "official_training_shape_matrix",
            "precision": precision, "runs_per_framework": runs,
            "pairing": "fresh processes; order alternates each run",
            "status": "pass" if all(row["status"] == "pass" for row in rows) else "incomplete",
            "rows": rows}


def main() -> int:
    args = options()
    models = load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    records = []

    def save(record: dict) -> None:
        records.append(record)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    for model in models:
        for batch, context in args.shapes:
            token_text = expanded_tokens(model["training"]["tokens"], context)
            shaped = copy.deepcopy(model)
            shaped["training"].update({"tokens": token_text, "batch": batch,
                                       "warmup": args.warmup, "steps": args.steps})
            with tempfile.TemporaryDirectory(prefix="microllm-shape-") as directory:
                temporary_manifest = Path(directory) / "manifest.json"
                temporary_manifest.write_text(json.dumps(
                    {"schema_version": 1, "models": [shaped]}), encoding="utf-8")
                for process_run in range(1, args.runs + 1):
                    order = ("microllm", "pytorch") if process_run % 2 else \
                        ("pytorch", "microllm")
                    for framework in order:
                        try:
                            command = (micro_command(args, model, batch, token_text)
                                       if framework == "microllm" else
                                       pytorch_command(args, temporary_manifest, model))
                            record = run_json(command, args.timeout_seconds)
                            validate_record(record, model, framework, batch, context,
                                            args.warmup, args.steps)
                            record.update({
                                "record_type": "official_training_shape_measurement",
                                "framework": framework, "model": model["name"],
                                "revision": model["revision"], "process_run": process_run,
                                "pair_order": list(order), "batch": batch, "context": context,
                                "precision": args.precision,
                                "peak_bytes": record["engine_peak_bytes"] if framework == "microllm"
                                              else record["device_peak_allocated_bytes"],
                            })
                        except Exception as error:
                            record = {"schema_version": 1,
                                      "record_type": "official_training_shape_measurement",
                                      "status": "failed", "framework": framework,
                                      "model": model["name"], "revision": model["revision"],
                                      "process_run": process_run, "pair_order": list(order),
                                      "batch": batch, "context": context,
                                      "precision": args.precision, "error": str(error)}
                        save(record)
    summary = summarize(records, models, args.shapes, args.precision, args.runs)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
