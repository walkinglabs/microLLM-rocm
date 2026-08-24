#!/usr/bin/env python3
"""Gate exact FP32 gate/up weight-gradient solutions on official models."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


MODELS = {
    "qwen2.5-0.5b": {
        "short": "qwen", "parameters": 494_032_768, "layers": 24,
        "solution": 289155,
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "short": "deepseek", "parameters": 1_777_088_000, "layers": 28,
        "solution": 284846,
    },
}
POLICIES = ("baseline", "candidate")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--qwen-config", required=True, type=Path)
    parser.add_argument("--qwen-weights", required=True, type=Path)
    parser.add_argument("--deepseek-config", required=True, type=Path)
    parser.add_argument("--deepseek-weights", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=512)
    result = parser.parse_args()
    if result.runs < 3 or result.warmup < 0 or result.steps <= 0 or \
       result.batch <= 0 or result.context <= 0:
        parser.error("comparison options are invalid")
    for path in (result.binary, result.qwen_config, result.qwen_weights,
                 result.deepseek_config, result.deepseek_weights):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def model_paths(args: argparse.Namespace, name: str) -> tuple[Path, Path]:
    if name.startswith("qwen"):
        return args.qwen_config, args.qwen_weights
    return args.deepseek_config, args.deepseek_weights


def run(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def main() -> int:
    args = options()
    tokens = ",".join(str(index) for index in range(1, args.context + 2))
    records = []
    for process_run in range(1, args.runs + 1):
        policies = POLICIES if process_run % 2 else tuple(reversed(POLICIES))
        for name, metadata in MODELS.items():
            config, weights = model_paths(args, name)
            for policy in policies:
                command = [
                    str(args.binary), "--config", str(config),
                    "--weights", str(weights), "--tokens", tokens,
                    "--device", "hip", "--learning-rate", "0.00001",
                    "--warmup", str(args.warmup), "--steps", str(args.steps),
                    "--batch", str(args.batch), "--linear-precision", "bf16",
                    "--bf16-weight-mirrors", "true",
                    "--adamw-moment-precision", "bf16",
                    "--adamw-bf16-multi-tensor-threshold", "auto",
                ]
                if policy == "candidate":
                    command.extend((
                        "--fp32-gate-up-weight-gradient-solution-index",
                        str(metadata["solution"])))
                record = run(command)
                expected_dispatches = ((args.warmup + args.steps) *
                                       int(metadata["layers"]) * 2)
                if record.get("status") != "pass" or \
                   record.get("parameter_count") != metadata["parameters"] or \
                   record.get("warmup") != args.warmup or \
                   record.get("steps") != args.steps or \
                   record.get("batch") != args.batch or \
                   record.get("context") != args.context or \
                   not record.get("parameter_changed"):
                    raise RuntimeError(f"invalid model row: {name}/{policy}")
                if policy == "candidate":
                    if record.get("fp32_solution_registered_entries") != 1 or \
                       record.get("fp32_solution_registry_hits") != expected_dispatches or \
                       record.get("fp32_solution_dispatches") != expected_dispatches:
                        raise RuntimeError(
                            f"solution did not hit exact gate/up gradients: {name}")
                elif record.get("fp32_solution_registered_entries") != 0 or \
                     record.get("fp32_solution_registry_hits") != 0 or \
                     record.get("fp32_solution_dispatches") != 0:
                    raise RuntimeError(f"baseline unexpectedly used solution: {name}")
                record.update({
                    "record_type": "fp32_weight_gradient_model_measurement",
                    "model": name, "policy": policy,
                    "process_run": process_run,
                })
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    comparisons = []
    passed = True
    for name in MODELS:
        baseline = [row for row in records
                    if row["model"] == name and row["policy"] == "baseline"]
        candidate = [row for row in records
                     if row["model"] == name and row["policy"] == "candidate"]
        throughput = (median(candidate, "tokens_per_second") /
                      median(baseline, "tokens_per_second"))
        peak_ratio = (median(candidate, "engine_peak_bytes") /
                      median(baseline, "engine_peak_bytes"))
        first_loss = relative(median(candidate, "first_loss"),
                              median(baseline, "first_loss"))
        final_loss = relative(median(candidate, "final_loss"),
                              median(baseline, "final_loss"))
        gates = {
            "throughput_at_least_1_01": throughput >= 1.01,
            "peak_at_most_1_01": peak_ratio <= 1.01,
            "first_loss_difference_at_most_0_01": first_loss <= 0.01,
            "final_loss_difference_at_most_0_01": final_loss <= 0.01,
        }
        passed = passed and all(gates.values())
        comparisons.append({
            "model": name,
            "solution_index": MODELS[name]["solution"],
            "runs_per_policy": len(baseline),
            "baseline_tokens_per_second_median":
                median(baseline, "tokens_per_second"),
            "candidate_tokens_per_second_median":
                median(candidate, "tokens_per_second"),
            "throughput_speedup": throughput,
            "peak_ratio": peak_ratio,
            "first_loss_relative_difference": first_loss,
            "final_loss_relative_difference": final_loss,
            "gates": gates,
        })
    summary = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "experiment": "fp32_weight_gradient_solution_model_gate",
        "processes": len(records),
        "comparisons": comparisons,
        "decision": ("keep explicit exact solutions" if passed
                     else "reject exact solution model route"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "training.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
