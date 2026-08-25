#!/usr/bin/env python3
"""Measure explicit quiescent allocator handoff in model Graph preflight."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys


CASES = (("qwen", 8), ("qwen", 512),
         ("deepseek", 8), ("deepseek", 512))
POLICIES = (False, True)
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
    result = parser.parse_args()
    for path in (result.binary, result.qwen_config, result.deepseek_config):
        if not path.is_file(): parser.error(f"input does not exist: {path}")
    if result.runs < 3: parser.error("handoff matrix requires at least three runs")
    return result


def execute(args: argparse.Namespace, model: str, context: int,
            handoff: bool) -> dict:
    config = args.qwen_config if model == "qwen" else args.deepseek_config
    completed = subprocess.run([
        str(args.binary), "--model", model, "--config", str(config),
        "--mode", "preflight", "--context", str(context), "--steps", "2",
        "--quiescent-handoff", "true" if handoff else "false",
    ], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        row = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("quiescent handoff preflight emitted invalid JSON") from error
    parameters, tensors = EXPECTED[model]
    expected_match = handoff and not (model == "deepseek" and context == 512)
    expected_count = 3 if handoff else 0
    expected_cache = handoff
    expected_fields = {
        "schema_version": 1, "status": "pass",
        "record_type": "optimizer_graph_model_measurement",
        "model": model, "mode": "preflight", "context": context,
        "parameter_count": parameters, "parameter_tensors": tensors,
        "quiescent_handoff": handoff,
        "quiescent_handoff_count": expected_count,
        "gradient_snapshot_matches": expected_match,
        "caching_allocator_enabled": expected_cache,
        "graph_launched": False, "captured_nodes": 0,
    }
    if any(row.get(key) != value for key, value in expected_fields.items()) or \
       float(row.get("preparation_ms", 0.0)) <= 0.0:
        raise RuntimeError(
            f"invalid quiescent handoff row: {model}/T{context}/{handoff}")
    return row


def summarize(records: list[dict], runs: int) -> dict:
    comparisons = []
    for model, context in CASES:
        policies = {}
        for handoff in POLICIES:
            rows = [row for row in records
                    if row["model"] == model and row["context"] == context and
                    row["quiescent_handoff"] == handoff]
            if len(rows) != runs:
                raise RuntimeError(
                    f"incomplete handoff case: {model}/T{context}/{handoff}")
            policies["handoff" if handoff else "disabled"] = {
                "snapshot_matches": all(
                    row["gradient_snapshot_matches"] for row in rows),
                "pool_enabled": all(
                    row["caching_allocator_enabled"] for row in rows),
                "handoff_count": min(
                    int(row["quiescent_handoff_count"]) for row in rows),
                "preparation_ms_median": statistics.median(
                    float(row["preparation_ms"]) for row in rows),
            }
        comparisons.append({
            "model": model, "context": context, "runs": runs,
            "policies": policies,
            "rescued": (policies["handoff"]["snapshot_matches"] and
                        not policies["disabled"]["snapshot_matches"]),
        })
    by_key = {(row["model"], row["context"]): row for row in comparisons}
    gates = {
        "disabled_policy_rejects_all_cases": all(
            not row["policies"]["disabled"]["snapshot_matches"]
            for row in comparisons),
        "handoff_reenables_pool_three_times": all(
            row["policies"]["handoff"]["pool_enabled"] and
            row["policies"]["handoff"]["handoff_count"] == 3
            for row in comparisons),
        "qwen_t8_t512_rescued": all(
            by_key[("qwen", context)]["rescued"] for context in (8, 512)),
        "deepseek_t8_rescued": by_key[("deepseek", 8)]["rescued"],
        "deepseek_t512_still_rejected":
            not by_key[("deepseek", 512)]["policies"]["handoff"][
                "snapshot_matches"],
        "no_graph_launched_during_preflight": all(
            not row["graph_launched"] for row in records),
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "experiment": "quiescent_allocator_handoff",
        "processes": len(records), "runs_per_policy_case": runs,
        "comparisons": comparisons, "gates": gates,
        "decision": (
            "keep explicit quiescent handoff primitive; continue Qwen and "
            "DeepSeek T8 model Graph gate; retain DeepSeek T512 rejection"),
    }


def main() -> int:
    args = options()
    records = []
    base = tuple((model, context, handoff)
                 for model, context in CASES for handoff in POLICIES)
    for process_run in range(1, args.runs + 1):
        ordered = base if process_run % 2 else tuple(reversed(base))
        for model, context, handoff in ordered:
            row = execute(args, model, context, handoff)
            row["process_run"] = process_run
            records.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = summarize(records, args.runs)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"quiescent_allocator_handoff_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
