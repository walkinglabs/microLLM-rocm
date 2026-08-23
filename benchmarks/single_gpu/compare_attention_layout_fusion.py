#!/usr/bin/env python3
"""Run same-binary official-model A/B for one Attention layout fusion."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--collect-diagnostics", action="store_true")
    parser.add_argument(
        "--policy", choices=("rope", "context", "plan"), default="rope",
        help="the one layout policy changed by the A/B")
    result = parser.parse_args()
    if result.runs <= 0 or result.context < 2 or result.warmup < 0 or result.steps <= 0:
        parser.error("runs/steps must be positive, context >= 2 and warmup nonnegative")
    for path in (result.manifest, result.binary):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def expanded_tokens(model: dict, context: int) -> str:
    source = [int(value) for value in model["training"]["tokens"].split(",")]
    if not source:
        raise ValueError(f"{model['name']} has no training tokens")
    # The training CLI shifts one token into the target. Supplying T+1 tokens
    # therefore measures exactly T input/target positions.
    tokens = context + 1
    values = (source * ((tokens + len(source) - 1) // len(source)))[:tokens]
    return ",".join(str(value) for value in values)


def command(args: argparse.Namespace, model: dict, enabled: bool,
            diagnostics: pathlib.Path | None = None) -> list[str]:
    training = model["training"]
    rope_enabled = enabled if args.policy == "rope" else True
    context_enabled = enabled if args.policy == "context" else True
    plan_cache_enabled = enabled if args.policy == "plan" else True
    result = [
        str(args.binary),
        "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", expanded_tokens(model, args.context),
        "--device", "hip",
        "--learning-rate", str(training["learning_rate"]),
        "--warmup", str(0 if diagnostics else args.warmup),
        "--steps", str(1 if diagnostics else args.steps),
        "--batch", "1",
        "--linear-precision", "bf16",
        "--bf16-weight-mirrors", "true",
        "--tied-embedding-sparse-add", "true",
        "--attention-rope-layout-fusion", "true" if rope_enabled else "false",
        "--attention-context-layout-fusion", "true" if context_enabled else "false",
        "--attention-layout-plan-cache", "true" if plan_cache_enabled else "false",
    ]
    if diagnostics:
        result.extend(("--diagnostics-output", str(diagnostics)))
    return result


def execute(args: argparse.Namespace, model: dict, enabled: bool,
            diagnostics: pathlib.Path | None = None) -> dict:
    completed = subprocess.run(
        command(args, model, enabled, diagnostics), check=True, text=True,
        capture_output=True)
    record = json.loads(completed.stdout)
    if record.get("status") != "pass" or not record.get("parameter_changed"):
        raise RuntimeError(f"{model['name']} did not complete a parameter update")
    if not math.isfinite(float(record["final_loss"])):
        raise RuntimeError(f"{model['name']} produced non-finite loss")
    field = {
        "rope": "attention_rope_layout_fusion",
        "context": "attention_context_layout_fusion",
        "plan": "attention_layout_plan_cache",
    }[args.policy]
    if record.get(field) is not enabled:
        raise RuntimeError(f"{model['name']} reported the wrong layout policy")
    return record


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    models = json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for process_run in range(1, args.runs + 1):
        for model in models:
            # Reverse every second pair so a slow drift is not assigned to one policy.
            policies = (False, True) if process_run % 2 else (True, False)
            for enabled in policies:
                record = execute(args, model, enabled)
                record.update({
                    "model": model["name"],
                    "revision": model.get("revision", "unknown"),
                    "process_run": process_run,
                    "policy": "fused" if enabled else "materialized",
                })
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    diagnostics = {}
    if args.collect_diagnostics:
        for model in models:
            diagnostics[model["name"]] = {}
            for enabled, policy in ((False, "materialized"), (True, "fused")):
                path = args.output_directory / f"{model['name']}-{policy}-diagnostics.json"
                execute(args, model, enabled, path)
                diagnostics[model["name"]][policy] = json.loads(
                    path.read_text(encoding="utf-8"))

    comparisons = []
    for model in models:
        selected = [row for row in records if row["model"] == model["name"]]
        policies = {}
        for policy in ("materialized", "fused"):
            group = [row for row in selected if row["policy"] == policy]
            if len(group) != args.runs:
                raise RuntimeError(f"{model['name']}/{policy} has incomplete runs")
            policies[policy] = {
                "tokens_per_second": median(group, "tokens_per_second"),
                "mean_step_ms": median(group, "mean_step_ms"),
                "mean_optimizer_ms": median(group, "mean_optimizer_ms"),
                "engine_peak_bytes": median(group, "engine_peak_bytes"),
                "engine_allocation_calls": median(group, "engine_allocation_calls"),
                "final_loss": median(group, "final_loss"),
                "observed_parameter_after": median(group, "observed_parameter_after"),
            }
        baseline = policies["materialized"]
        candidate = policies["fused"]
        item = {
            "model": model["name"],
            "policies": policies,
            "throughput_speedup": candidate["tokens_per_second"] /
                baseline["tokens_per_second"],
            "peak_ratio": candidate["engine_peak_bytes"] /
                baseline["engine_peak_bytes"],
            "peak_bytes_saved": baseline["engine_peak_bytes"] -
                candidate["engine_peak_bytes"],
            "allocation_calls_saved": baseline["engine_allocation_calls"] -
                candidate["engine_allocation_calls"],
            "loss_relative_difference": abs(
                candidate["final_loss"] - baseline["final_loss"]) /
                max(abs(baseline["final_loss"]), 1.0e-12),
            "observed_parameter_after_equal":
                candidate["observed_parameter_after"] ==
                baseline["observed_parameter_after"],
        }
        if args.collect_diagnostics:
            item["strided_copy"] = {
                policy: {
                    field: diagnostics[model["name"]][policy]["strided_copy"][field]
                    for field in ("calls", "elements", "bytes")
                }
                for policy in ("materialized", "fused")
            }
        comparisons.append(item)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "track": ("attention_layout_plan_cache" if args.policy == "plan" else
                  f"attention_{args.policy}_layout_fusion"),
        "policy": args.policy,
        "context": args.context,
        "warmup": args.warmup,
        "steps": args.steps,
        "runs_per_model_policy": args.runs,
        "aggregation": "median of fresh processes with alternating policy order",
        "comparisons": comparisons,
        "keep_gate": {
            "throughput_ratio_minimum": 1.01 if args.policy == "plan" else 0.98,
            "loss_relative_difference_maximum": 0.005,
            "observed_parameter_after_equal": True,
        },
    }
    summary["gate_results"] = {
        "throughput": all(
            row["throughput_speedup"] >=
            (1.01 if args.policy == "plan" else 0.98)
            for row in comparisons),
        "loss": all(
            row["loss_relative_difference"] <= 0.005 for row in comparisons),
        "parameter": all(
            row["observed_parameter_after_equal"] for row in comparisons),
        ("strided_copy_unchanged_zero" if args.policy == "plan" else
         "strided_copy_reduced"): all(
            (row.get("strided_copy", {}).get("fused", {}).get("bytes") == 0 and
             row.get("strided_copy", {}).get("materialized", {}).get("bytes") == 0)
            if args.policy == "plan" else
            row.get("strided_copy", {}).get("fused", {}).get("bytes", math.inf) <
            row.get("strided_copy", {}).get("materialized", {}).get("bytes", -math.inf)
            for row in comparisons) if args.collect_diagnostics else None,
    }
    if not args.collect_diagnostics:
        summary["decision"] = "incomplete evidence: diagnostics were not collected"
    else:
        accepted = all(summary["gate_results"].values())
        subject = "plan cache" if args.policy == "plan" else "layout fusion"
        summary["decision"] = f"{'keep' if accepted else 'reject'} {subject}"
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
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            subprocess.CalledProcessError, RuntimeError) as error:
        print(f"compare_attention_layout_fusion: {error}", file=sys.stderr)
        raise SystemExit(2)
