#!/usr/bin/env python3
"""Longer stepwise and complete-parameter gate for BF16 gate/up gradients."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_NAMES = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-binary", required=True, type=Path)
    parser.add_argument("--compare-binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.runs <= 0 or args.context < 2 or args.warmup < 0 or args.steps < 3:
        parser.error("trajectory counts are invalid")
    for path in (args.manifest, args.train_binary, args.compare_binary):
        if not path.is_file():
            parser.error(f"trajectory input is unavailable: {path}")
    return args


def tokens(model: dict, context: int) -> str:
    seed = [int(value) for value in model["training"]["tokens"].split(",")]
    return ",".join(str(seed[index % len(seed)]) for index in range(context + 1))


def command(args: argparse.Namespace, model: dict, candidate: bool,
            loss_path: Path, parameter_path: Path | None) -> list[str]:
    result = [
        str(args.train_binary), "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", tokens(model, args.context), "--device", "hip",
        "--batch", "1", "--learning-rate", str(model["training"]["learning_rate"]),
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--linear-precision", "bf16", "--bf16-weight-mirrors", "true",
        "--adamw-implementation", "auto", "--adamw-moment-precision", "bf16",
        "--adamw-bf16-multi-tensor-threshold", "1048576",
        "--tied-embedding-sparse-add", "true",
        "--unique-gradient-inplace-add", "false",
        "--attention-rope-layout-fusion", "true",
        "--attention-context-layout-fusion", "true",
        "--attention-layout-plan-cache", "false",
        "--attention-gemm-scale-fusion", "false",
        "--attention-paired-gqa-repeat", "false",
        "--attention-gqa-value-broadcast", "false",
        "--attention-gqa-forward-value-broadcast", "false",
        "--bf16-gate-up-weight-gradient", "true" if candidate else "false",
        "--loss-trajectory-output", str(loss_path),
    ]
    if parameter_path:
        result.extend(("--gate-up-parameters-output", str(parameter_path)))
    return result


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("trajectory worker emitted no JSON")


def execute(args: argparse.Namespace, model: dict, candidate: bool,
            loss_path: Path, parameter_path: Path | None) -> dict:
    completed = subprocess.run(
        command(args, model, candidate, loss_path, parameter_path),
        text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    record = last_json(completed.stdout)
    trajectory = json.loads(loss_path.read_text(encoding="utf-8"))
    loss_path.unlink()
    losses = trajectory.get("losses", [])
    if (record.get("status") != "pass" or
            record.get("bf16_gate_up_weight_gradient") is not candidate or
            record.get("loss_trajectory_output_written") is not True or
            record.get("loss_trajectory_steps") != args.steps or
            len(losses) != args.steps or
            any(not math.isfinite(float(loss)) for loss in losses) or
            record.get("gate_up_parameters_output_written") is not
                (parameter_path is not None)):
        raise RuntimeError(f"trajectory contract failed for {model['name']}")
    if parameter_path and (record.get("gate_up_parameter_tensors", 0) <= 0 or
                           record.get("gate_up_parameter_elements", 0) <= 0 or
                           not parameter_path.is_file()):
        raise RuntimeError(f"parameter export failed for {model['name']}")
    record.update({
        "record_type": "bf16_weight_gradient_trajectory_measurement",
        "model": model["name"],
        "revision": model.get("revision", "unknown"),
        "policy": "candidate" if candidate else "baseline",
        "losses": losses,
    })
    return record


def compare_parameters(binary: Path, baseline: Path, candidate: Path) -> dict:
    completed = subprocess.run(
        [str(binary), str(baseline), str(candidate)], text=True,
        capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    result = last_json(completed.stdout)
    if (result.get("status") != "pass" or
            result.get("record_type") != "safetensors_complete_comparison" or
            result.get("all_finite") is not True or
            result.get("tensor_count", 0) <= 0 or
            result.get("compared_elements", 0) <= 0):
        raise RuntimeError("complete gate/up parameter comparison failed")
    baseline.unlink()
    candidate.unlink()
    return result


def relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), 1.0e-12)


def main() -> int:
    args = options()
    models = json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
    if {model.get("name") for model in models} != MODEL_NAMES:
        raise RuntimeError("trajectory gate requires both pinned models")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    parameter_comparisons = {}
    with tempfile.TemporaryDirectory(prefix="microllm-bf16-wgrad-trajectory-") as temp:
        scratch = Path(temp)
        for process_run in range(1, args.runs + 1):
            for model in models:
                order = (False, True) if process_run % 2 else (True, False)
                parameter_paths = {}
                for candidate in order:
                    policy = "candidate" if candidate else "baseline"
                    loss_path = scratch / f"{model['name']}-{process_run}-{policy}-loss.json"
                    parameter_path = (scratch / f"{model['name']}-{policy}.safetensors"
                                      if process_run == 1 else None)
                    record = execute(
                        args, model, candidate, loss_path, parameter_path)
                    record["process_run"] = process_run
                    records.append(record)
                    (args.output_directory / "trajectory.jsonl").write_text(
                        "".join(json.dumps(row, sort_keys=True) + "\n"
                                for row in records), encoding="utf-8")
                    if parameter_path:
                        parameter_paths[policy] = parameter_path
                if process_run == 1:
                    parameter_comparisons[model["name"]] = compare_parameters(
                        args.compare_binary, parameter_paths["baseline"],
                        parameter_paths["candidate"])
    comparisons = []
    for model in models:
        selected = [row for row in records if row["model"] == model["name"]]
        policies = {policy: [row for row in selected if row["policy"] == policy]
                    for policy in ("baseline", "candidate")}
        if any(len(rows) != args.runs for rows in policies.values()):
            raise RuntimeError(f"incomplete trajectory matrix for {model['name']}")
        paired_relative = []
        for process_run in range(1, args.runs + 1):
            baseline = next(row for row in policies["baseline"]
                            if row["process_run"] == process_run)
            candidate = next(row for row in policies["candidate"]
                             if row["process_run"] == process_run)
            paired_relative.extend(
                relative(float(left), float(right))
                for left, right in zip(baseline["losses"], candidate["losses"]))
        baseline_tps = statistics.median(
            float(row["tokens_per_second"]) for row in policies["baseline"])
        candidate_tps = statistics.median(
            float(row["tokens_per_second"]) for row in policies["candidate"])
        baseline_peak = statistics.median(
            float(row["engine_peak_bytes"]) for row in policies["baseline"])
        candidate_peak = statistics.median(
            float(row["engine_peak_bytes"]) for row in policies["candidate"])
        comparisons.append({
            "model": model["name"],
            "throughput_speedup": candidate_tps / baseline_tps,
            "baseline_tokens_per_second": baseline_tps,
            "candidate_tokens_per_second": candidate_tps,
            "peak_ratio": candidate_peak / baseline_peak,
            "loss_relative_difference_maximum": max(paired_relative),
            "loss_relative_difference_rms": math.sqrt(statistics.mean(
                value * value for value in paired_relative)),
            "loss_values_compared": len(paired_relative),
            "parameter_comparison": parameter_comparisons[model["name"]],
        })
    gates = {
        "throughput": all(row["throughput_speedup"] >= 1.01
                          for row in comparisons),
        "peak_memory": all(row["peak_ratio"] <= 1.01 for row in comparisons),
        "loss_trajectory": all(row["loss_relative_difference_maximum"] <= 0.005
                               for row in comparisons),
        "parameter_maximum": all(
            row["parameter_comparison"]["maximum_absolute_difference"] <= 5.0e-5
            for row in comparisons),
        "parameter_rms": all(
            row["parameter_comparison"]["rms_difference"] <= 1.0e-6
            for row in comparisons),
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "bf16_weight_gradient_trajectory_summary",
        "context": args.context,
        "warmup": args.warmup,
        "steps": args.steps,
        "runs_per_model_policy": args.runs,
        "raw_processes": len(records),
        "comparisons": comparisons,
        "gates": {
            "throughput_speedup_minimum": 1.01,
            "peak_ratio_maximum": 1.01,
            "loss_relative_difference_maximum": 0.005,
            "parameter_maximum_absolute_difference": 5.0e-5,
            "parameter_rms_difference": 1.0e-6,
        },
        "gate_results": gates,
        "decision": ("admit default gate/up BF16 weight gradients"
                     if all(gates.values()) else
                     "keep explicit; longer trajectory gate failed"),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError,
            json.JSONDecodeError, StopIteration) as error:
        print(f"bf16_weight_gradient_trajectory: {error}", file=sys.stderr)
        raise SystemExit(2)

