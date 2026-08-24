#!/usr/bin/env python3
"""Compare FP32 and BF16 AdamW moment storage on official-model shapes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


MODELS = (
    ("qwen2.5-0.5b", 494_032_768),
    ("deepseek-r1-distill-qwen-1.5b", 1_777_088_000),
)
POLICIES = ("fp32", "bf16")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--qwen-config", required=True, type=Path)
    parser.add_argument("--qwen-weights", required=True, type=Path)
    parser.add_argument("--deepseek-config", required=True, type=Path)
    parser.add_argument("--deepseek-weights", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--bf16-multi-tensor-threshold", type=int, default=0)
    result = parser.parse_args()
    if result.runs < 3 or result.warmup < 0 or result.steps <= 0:
        parser.error("runs must be at least 3; warmup nonnegative; steps positive")
    if result.batch <= 0 or result.context <= 0:
        parser.error("batch and context must be positive")
    if result.bf16_multi_tensor_threshold < 0:
        parser.error("BF16 multi-tensor threshold must be non-negative")
    for path in (result.binary, result.qwen_config, result.qwen_weights,
                 result.deepseek_config, result.deepseek_weights):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"command did not emit one JSON object: {completed.stdout!r}") from error


def model_paths(args: argparse.Namespace, model: str) -> tuple[Path, Path]:
    if model.startswith("qwen"):
        return args.qwen_config, args.qwen_weights
    return args.deepseek_config, args.deepseek_weights


def command(args: argparse.Namespace, model: str, policy: str) -> list[str]:
    config, weights = model_paths(args, model)
    tokens = ",".join(str(index) for index in range(1, args.context + 2))
    result = [
        str(args.binary),
        "--config", str(config),
        "--weights", str(weights),
        "--tokens", tokens,
        "--device", "hip",
        "--learning-rate", "0.00001",
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--linear-precision", "bf16",
        "--bf16-weight-mirrors", "true",
        "--adamw-implementation", "auto",
        "--adamw-moment-precision", policy,
    ]
    if policy == "bf16":
        result.extend(("--adamw-bf16-multi-tensor-threshold",
                       str(args.bf16_multi_tensor_threshold)))
    return result


def validate_record(record: dict, model: str, parameters: int,
                    policy: str, args: argparse.Namespace) -> None:
    if record.get("schema_version") != 1 or record.get("status") != "pass":
        raise RuntimeError(f"{model}/{policy} returned a non-pass record")
    if record.get("parameter_count") != parameters:
        raise RuntimeError(f"{model}/{policy} parameter count changed")
    if record.get("adamw_moment_precision") != policy:
        raise RuntimeError(f"{model}/{policy} reported the wrong policy")
    expected_bytes = parameters * (4 if policy == "bf16" else 8)
    if record.get("adamw_moment_state_bytes") != expected_bytes:
        raise RuntimeError(f"{model}/{policy} moment-state bytes changed")
    if record.get("optimizer_device_to_host_calls") != 0 or \
       record.get("optimizer_device_to_host_bytes") != 0:
        raise RuntimeError(f"{model}/{policy} copied optimizer state to the host")
    expected_threshold = (args.bf16_multi_tensor_threshold
                          if policy == "bf16" else 0)
    if record.get("adamw_bf16_multi_tensor_threshold") != expected_threshold:
        raise RuntimeError(f"{model}/{policy} reported the wrong threshold")
    if expected_threshold == 0:
        if record.get("adamw_multi_tensor_update") is not False or \
           record.get("optimizer_host_to_device_calls") != 0 or \
           record.get("optimizer_host_to_device_bytes") != 0:
            raise RuntimeError(
                f"{model}/{policy} unexpectedly copied optimizer payloads")
    else:
        if record.get("adamw_multi_tensor_update") is not True or \
           int(record.get("adamw_bf16_multi_tensor_tensors", 0)) <= 0 or \
           int(record.get("adamw_bf16_multi_tensor_elements", 0)) <= 0 or \
           record.get("optimizer_host_to_device_calls") != args.steps:
            raise RuntimeError(f"{model}/{policy} did not use hybrid dispatch")
        metadata_bytes = int(record.get("optimizer_host_to_device_bytes", 0))
        if metadata_bytes <= 0 or metadata_bytes >= parameters:
            raise RuntimeError(f"{model}/{policy} metadata copy is not bounded")
    if record.get("warmup") != args.warmup or record.get("steps") != args.steps:
        raise RuntimeError(f"{model}/{policy} measurement counts changed")
    if record.get("batch") != args.batch or record.get("context") != args.context:
        raise RuntimeError(f"{model}/{policy} workload shape changed")
    required = ("tokens_per_second", "mean_optimizer_ms", "engine_peak_bytes",
                "first_loss", "final_loss")
    if any(not math.isfinite(float(record.get(field, math.nan)))
           for field in required):
        raise RuntimeError(f"{model}/{policy} emitted a non-finite metric")
    if not record.get("parameter_changed"):
        raise RuntimeError(f"{model}/{policy} did not update a parameter")


def median(records: list[dict], field: str) -> float:
    return statistics.median(float(record[field]) for record in records)


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def summarize(records: list[dict], runs: int) -> dict:
    models = []
    required_passed = True
    stretch_passed = True
    for model, parameters in MODELS:
        grouped = {
            policy: [row for row in records
                     if row["model"] == model and row["policy"] == policy]
            for policy in POLICIES
        }
        if any(len(rows) != runs for rows in grouped.values()):
            raise RuntimeError(f"{model} does not have {runs} rows per policy")
        fp32 = grouped["fp32"]
        bf16 = grouped["bf16"]
        throughput_speedup = (median(bf16, "tokens_per_second") /
                              median(fp32, "tokens_per_second"))
        optimizer_speedup = (median(fp32, "mean_optimizer_ms") /
                             median(bf16, "mean_optimizer_ms"))
        peak_ratio = (median(bf16, "engine_peak_bytes") /
                      median(fp32, "engine_peak_bytes"))
        first_loss_difference = relative_difference(
            median(fp32, "first_loss"), median(bf16, "first_loss"))
        final_loss_difference = relative_difference(
            median(fp32, "final_loss"), median(bf16, "final_loss"))
        gates = {
            "throughput_speedup_at_least_1_01": throughput_speedup >= 1.01,
            "optimizer_speedup_at_least_1_10": optimizer_speedup >= 1.10,
            "peak_ratio_at_most_0_90": peak_ratio <= 0.90,
            "first_loss_relative_difference_at_most_0_01":
                first_loss_difference <= 0.01,
            "final_loss_relative_difference_at_most_0_01":
                final_loss_difference <= 0.01,
            "moment_bytes_exactly_half": (
                int(bf16[0]["adamw_moment_state_bytes"]) * 2 ==
                int(fp32[0]["adamw_moment_state_bytes"]) == parameters * 8),
        }
        required_names = (
            "throughput_speedup_at_least_1_01",
            "peak_ratio_at_most_0_90",
            "first_loss_relative_difference_at_most_0_01",
            "final_loss_relative_difference_at_most_0_01",
            "moment_bytes_exactly_half",
        )
        model_required = all(gates[name] for name in required_names)
        model_stretch = gates["optimizer_speedup_at_least_1_10"]
        required_passed = required_passed and model_required
        stretch_passed = stretch_passed and model_stretch
        models.append({
            "model": model,
            "parameter_count": parameters,
            "fp32_tokens_per_second_median": median(fp32, "tokens_per_second"),
            "bf16_tokens_per_second_median": median(bf16, "tokens_per_second"),
            "throughput_speedup": throughput_speedup,
            "fp32_optimizer_ms_median": median(fp32, "mean_optimizer_ms"),
            "bf16_optimizer_ms_median": median(bf16, "mean_optimizer_ms"),
            "optimizer_speedup": optimizer_speedup,
            "fp32_peak_bytes_median": median(fp32, "engine_peak_bytes"),
            "bf16_peak_bytes_median": median(bf16, "engine_peak_bytes"),
            "peak_ratio": peak_ratio,
            "first_loss_relative_difference": first_loss_difference,
            "final_loss_relative_difference": final_loss_difference,
            "gates": gates,
            "required_gates_passed": model_required,
            "optimizer_stretch_gate_passed": model_stretch,
        })
    status = ("pass" if required_passed and stretch_passed else
              "partial_keep" if required_passed else "fail")
    return {
        "schema_version": 1,
        "experiment": "bf16_adamw_moments",
        "status": status,
        "required_gates_passed": required_passed,
        "optimizer_stretch_gates_passed": stretch_passed,
        "runs_per_policy": runs,
        "bf16_multi_tensor_threshold": int(
            records[0].get("matrix_bf16_multi_tensor_threshold", 0)),
        "models": models,
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    # Alternate policy order on each run so slow drift cannot favor one side.
    for process_run in range(1, args.runs + 1):
        order = POLICIES if process_run % 2 else tuple(reversed(POLICIES))
        for model, parameters in MODELS:
            for policy in order:
                record = run_json(command(args, model, policy))
                validate_record(record, model, parameters, policy, args)
                record.update({
                    "record_type": "bf16_adamw_moment_training",
                    "model": model,
                    "policy": policy,
                    "process_run": process_run,
                    "matrix_bf16_multi_tensor_threshold":
                        args.bf16_multi_tensor_threshold,
                })
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    summary = summarize(records, args.runs)
    (args.output_directory / "training.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] in {"pass", "partial_keep"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
