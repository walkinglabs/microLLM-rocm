#!/usr/bin/env python3
"""Gate two-node AdamW Graph on graph-safe model/context cases."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys


CASES = (("qwen", 8), ("qwen", 512), ("deepseek", 8))
MODES = ("eager", "graph")
EXPECTED = {
    "qwen": (494032768, 290),
    "deepseek": (1777088000, 339),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--qwen-config", required=True, type=pathlib.Path)
    parser.add_argument("--deepseek-config", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    result = parser.parse_args()
    for path in (result.binary, result.qwen_config, result.deepseek_config):
        if not path.is_file(): parser.error(f"input does not exist: {path}")
    if result.runs < 3 or result.steps < 2:
        parser.error("model Graph matrix counts are invalid")
    return result


def command(args: argparse.Namespace, model: str, context: int,
            mode: str) -> list[str]:
    config = args.qwen_config if model == "qwen" else args.deepseek_config
    return [
        str(args.binary), "--model", model, "--config", str(config),
        "--mode", mode, "--context", str(context),
        "--steps", str(args.steps), "--quiescent-handoff", "true",
    ]


def execute(args: argparse.Namespace, model: str, context: int,
            mode: str) -> dict:
    completed = subprocess.run(command(args, model, context, mode),
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        row = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("optimizer model Graph benchmark emitted invalid JSON") from error
    parameters, tensors = EXPECTED[model]
    if mode == "preflight":
        if row.get("gradient_snapshot_matches") is not False or \
           row.get("graph_launched") is not False or \
           row.get("captured_nodes") != 0 or \
           row.get("quiescent_handoff_count") != 3:
            raise RuntimeError("DeepSeek T512 rejection changed")
        return row
    if row.get("schema_version") != 1 or row.get("status") != "pass" or \
       row.get("record_type") != "optimizer_graph_model_measurement" or \
       row.get("model") != model or row.get("mode") != mode or \
       row.get("context") != context or row.get("steps") != args.steps or \
       row.get("parameter_count") != parameters or \
       row.get("parameter_tensors") != tensors or \
       row.get("gradient_snapshot_matches") is not True or \
       row.get("quiescent_handoff") is not True or \
       row.get("quiescent_handoff_count") != args.steps + 2 or \
       row.get("optimizer_step") != args.steps or \
       row.get("captured_nodes") != (2 if mode == "graph" else 0) or \
       row.get("graph_launched") is not (mode == "graph") or \
       row.get("optimizer_host_to_device_calls") != (
           0 if mode == "graph" else args.steps) or \
       row.get("optimizer_device_to_host_calls") != 0 or \
       row.get("optimizer_device_to_device_calls") != 0 or \
       len(row.get("losses", [])) != args.steps or \
       any(not math.isfinite(float(value)) for value in row.get("losses", [])):
        raise RuntimeError(f"invalid model Graph row: {model}/T{context}/{mode}")
    return row


def summarize(records: list[dict], runs: int, steps: int) -> dict:
    comparisons = []
    for model, context in CASES:
        selected = [row for row in records
                    if row["model"] == model and row["context"] == context and
                    row["mode"] in MODES]
        policies = {}
        for mode in MODES:
            rows = [row for row in selected if row["mode"] == mode]
            if len(rows) != runs:
                raise RuntimeError(f"incomplete model Graph case: {model}/T{context}/{mode}")
            policies[mode] = {
                "mean_optimizer_ms": statistics.median(
                    float(row["mean_optimizer_ms"]) for row in rows),
                "mean_step_ms": statistics.median(
                    float(row["mean_step_ms"]) for row in rows),
                "preparation_ms": statistics.median(
                    float(row["preparation_ms"]) for row in rows),
                "optimizer_h2d_calls": statistics.median(
                    int(row["optimizer_host_to_device_calls"]) for row in rows),
            }
        maximum_loss_error = 0.0
        maximum_parameter_error = 0.0
        for process_run in range(1, runs + 1):
            eager = next(row for row in selected
                         if row["mode"] == "eager" and
                         row["process_run"] == process_run)
            graph = next(row for row in selected
                         if row["mode"] == "graph" and
                         row["process_run"] == process_run)
            maximum_loss_error = max(
                maximum_loss_error,
                max(abs(float(a) - float(b))
                    for a, b in zip(eager["losses"], graph["losses"])))
            maximum_parameter_error = max(
                maximum_parameter_error,
                abs(float(eager["observed_parameter"]) -
                    float(graph["observed_parameter"])))
        comparisons.append({
            "model": model, "context": context, "runs": runs,
            "steps": steps, "policies": policies,
            "maximum_loss_error": maximum_loss_error,
            "maximum_parameter_error": maximum_parameter_error,
            "optimizer_speedup": (
                policies["eager"]["mean_optimizer_ms"] /
                policies["graph"]["mean_optimizer_ms"]),
            "step_speedup": (policies["eager"]["mean_step_ms"] /
                             policies["graph"]["mean_step_ms"]),
        })
    rejected = [row for row in records if row["mode"] == "preflight"]
    gates = {
        "loss_and_parameter_exact": all(
            row["maximum_loss_error"] == 0.0 and
            row["maximum_parameter_error"] == 0.0 for row in comparisons),
        "graph_has_two_nodes_and_no_optimizer_metadata_copy": all(
            row["captured_nodes"] == 2 and
            row["optimizer_host_to_device_calls"] == 0
            for row in records if row["mode"] == "graph"),
        "all_graph_safe_snapshots_match": all(
            row["gradient_snapshot_matches"] for row in records
            if row["mode"] in MODES),
        "optimizer_speedup_at_least_1_01": all(
            row["optimizer_speedup"] >= 1.01 for row in comparisons),
        "end_to_end_step_speedup_at_least_1_01": all(
            row["step_speedup"] >= 1.01 for row in comparisons),
        "deepseek_t512_remains_zero_launch": len(rejected) == runs and all(
            not row["gradient_snapshot_matches"] and
            not row["graph_launched"] for row in rejected),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "experiment": "optimizer_graph_model_gate",
        "processes": len(records), "runs_per_policy_case": runs,
        "comparisons": comparisons, "gates": gates,
        "decision": (
            "keep model optimizer Graph route"
            if all(gates[name] for name in (
                "optimizer_speedup_at_least_1_01",
                "end_to_end_step_speedup_at_least_1_01")) else
            "reject model optimizer Graph route; close optimizer-only Graph track"),
    }


def main() -> int:
    args = options()
    records = []
    base = tuple((model, context, mode)
                 for model, context in CASES for mode in MODES)
    for process_run in range(1, args.runs + 1):
        ordered = base if process_run % 2 else tuple(reversed(base))
        for model, context, mode in ordered:
            row = execute(args, model, context, mode)
            row["process_run"] = process_run
            records.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        rejected = execute(args, "deepseek", 512, "preflight")
        rejected["process_run"] = process_run
        records.append(rejected)
        print(json.dumps(rejected, sort_keys=True), flush=True)
    summary = summarize(records, args.runs, args.steps)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"optimizer_graph_model_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
