#!/usr/bin/env python3
"""Same-binary model gate for gate/up-only BF16 weight gradients."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


EXPECTED_ROUTE_CALLS = {
    "qwen2.5-0.5b": 48,
    "deepseek-r1-distill-qwen-1.5b": 56,
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.runs <= 0 or args.context < 2 or args.warmup < 0 or args.steps <= 0:
        parser.error("model gate counts are invalid")
    if not args.manifest.is_file() or not args.binary.is_file():
        parser.error("model gate inputs are unavailable")
    return args


def expanded_tokens(model: dict, context: int) -> str:
    seed = [int(value) for value in model["training"]["tokens"].split(",")]
    count = context + 1
    return ",".join(str(seed[index % len(seed)]) for index in range(count))


def command(args: argparse.Namespace, model: dict, enabled: bool,
            diagnostics: Path | None = None) -> list[str]:
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", expanded_tokens(model, args.context),
        "--device", "hip", "--batch", "1",
        "--learning-rate", str(model["training"]["learning_rate"]),
        "--warmup", str(0 if diagnostics else args.warmup),
        "--steps", str(1 if diagnostics else args.steps),
        "--linear-precision", "bf16", "--bf16-weight-mirrors", "true",
        "--adamw-implementation", "auto",
        "--adamw-moment-precision", "bf16",
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
        "--bf16-gate-up-weight-gradient", "true" if enabled else "false",
    ]
    if diagnostics:
        result.extend(("--diagnostics-output", str(diagnostics)))
    return result


def execute(args: argparse.Namespace, model: dict, enabled: bool,
            diagnostics: Path | None = None) -> dict:
    completed = subprocess.run(
        command(args, model, enabled, diagnostics), text=True,
        capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    record = json.loads(completed.stdout)
    if (record.get("status") != "pass" or
            record.get("bf16_gate_up_weight_gradient") is not enabled or
            record.get("adamw_moment_precision") != "bf16" or
            record.get("adamw_bf16_multi_tensor_threshold") != 1048576 or
            record.get("parameter_changed") is not True or
            not math.isfinite(float(record.get("final_loss", math.nan)))):
        raise RuntimeError(f"model route contract failed for {model['name']}")
    return record


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def route_calls(diagnostics: dict) -> int:
    return sum(int(row.get("first_assignments", 0))
               for row in diagnostics["gradient_accumulation"]["records"]
               if row.get("first_source") == "bf16_gate_up_weight_gradient")


def main() -> int:
    args = options()
    models = json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
    if {model.get("name") for model in models} != set(EXPECTED_ROUTE_CALLS):
        raise RuntimeError("BF16 weight-gradient model gate requires pinned models")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        for model in models:
            policies = (False, True) if process_run % 2 else (True, False)
            for enabled in policies:
                record = execute(args, model, enabled)
                record.update({
                    "record_type": "bf16_weight_gradient_model_measurement",
                    "model": model["name"],
                    "revision": model.get("revision", "unknown"),
                    "process_run": process_run,
                    "policy": "candidate" if enabled else "baseline",
                })
                records.append(record)
                (args.output_directory / "training.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n"
                            for row in records),
                    encoding="utf-8")
                print(json.dumps(record, sort_keys=True), flush=True)
    diagnostics = {}
    for model in models:
        path = args.output_directory / f"{model['name']}-candidate-diagnostics.json"
        execute(args, model, True, path)
        document = json.loads(path.read_text(encoding="utf-8"))
        calls = route_calls(document)
        if calls != EXPECTED_ROUTE_CALLS[model["name"]]:
            raise RuntimeError(
                f"{model['name']} routed {calls} BF16 gate/up gradients")
        diagnostics[model["name"]] = {
            "bf16_gate_up_weight_gradient_assignments": calls,
            "strided_copy_calls": document["strided_copy"]["calls"],
            "strided_copy_bytes": document["strided_copy"]["bytes"],
        }
    comparisons = []
    for model in models:
        grouped = {policy: [row for row in records
                            if row["model"] == model["name"] and
                            row["policy"] == policy]
                   for policy in ("baseline", "candidate")}
        if any(len(rows) != args.runs for rows in grouped.values()):
            raise RuntimeError(f"incomplete A/B for {model['name']}")
        policies = {}
        for policy, rows in grouped.items():
            policies[policy] = {
                "tokens_per_second": median(rows, "tokens_per_second"),
                "mean_step_ms": median(rows, "mean_step_ms"),
                "mean_optimizer_ms": median(rows, "mean_optimizer_ms"),
                "engine_peak_bytes": median(rows, "engine_peak_bytes"),
                "engine_allocation_calls": median(rows, "engine_allocation_calls"),
                "first_loss": median(rows, "first_loss"),
                "final_loss": median(rows, "final_loss"),
                "observed_parameter_after": median(rows, "observed_parameter_after"),
            }
        baseline, candidate = policies["baseline"], policies["candidate"]
        comparisons.append({
            "model": model["name"],
            "policies": policies,
            "throughput_speedup": candidate["tokens_per_second"] /
                baseline["tokens_per_second"],
            "peak_ratio": candidate["engine_peak_bytes"] /
                baseline["engine_peak_bytes"],
            "allocation_calls_delta": candidate["engine_allocation_calls"] -
                baseline["engine_allocation_calls"],
            "first_loss_relative_difference": abs(
                candidate["first_loss"] - baseline["first_loss"]) /
                max(abs(baseline["first_loss"]), 1.0e-12),
            "final_loss_relative_difference": abs(
                candidate["final_loss"] - baseline["final_loss"]) /
                max(abs(baseline["final_loss"]), 1.0e-12),
            "observed_parameter_relative_difference": abs(
                candidate["observed_parameter_after"] -
                baseline["observed_parameter_after"]) /
                max(abs(baseline["observed_parameter_after"]), 1.0e-12),
            "diagnostics": diagnostics[model["name"]],
        })
    gates = {
        "throughput": all(row["throughput_speedup"] >= 1.01
                          for row in comparisons),
        "peak_memory": all(row["peak_ratio"] <= 1.01 for row in comparisons),
        "first_loss": all(row["first_loss_relative_difference"] <= 0.005
                          for row in comparisons),
        "two_step_loss": all(row["final_loss_relative_difference"] <= 0.005
                             for row in comparisons),
        "observed_parameter": all(
            row["observed_parameter_relative_difference"] <= 5.0e-4
            for row in comparisons),
        "route_count": all(
            row["diagnostics"]["bf16_gate_up_weight_gradient_assignments"] ==
            EXPECTED_ROUTE_CALLS[row["model"]] for row in comparisons),
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "bf16_weight_gradient_model_gate_summary",
        "context": args.context,
        "warmup": args.warmup,
        "steps": args.steps,
        "runs_per_model_policy": args.runs,
        "raw_processes": len(records),
        "aggregation": "median of fresh processes with alternating policy order",
        "comparisons": comparisons,
        "keep_gate": {
            "throughput_speedup_minimum": 1.01,
            "peak_ratio_maximum": 1.01,
            "first_loss_relative_difference_maximum": 0.005,
            "final_loss_relative_difference_maximum": 0.005,
            "observed_parameter_relative_difference_maximum": 5.0e-4,
        },
        "gate_results": gates,
        "decision": ("keep explicit candidate for longer training validation"
                     if all(gates.values()) else
                     "reject model route; keep operator evidence only"),
    }
    (args.output_directory / "training.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"compare_bf16_weight_gradient_models: {error}", file=sys.stderr)
        raise SystemExit(2)
