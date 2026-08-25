#!/usr/bin/env python3
"""Prove whether model optimizer Graph snapshots survive Graph Stream setup."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys


CASES = (("qwen", 8), ("qwen", 512),
         ("deepseek", 8), ("deepseek", 512))
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
    if result.runs < 3: parser.error("preflight requires at least three runs")
    return result


def execute(args: argparse.Namespace, model: str, context: int) -> dict:
    config = args.qwen_config if model == "qwen" else args.deepseek_config
    completed = subprocess.run([
        str(args.binary), "--model", model, "--config", str(config),
        "--mode", "preflight", "--context", str(context), "--steps", "2",
    ], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        row = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("optimizer Graph preflight emitted invalid JSON") from error
    parameters, tensors = EXPECTED[model]
    if row != {
        **row,
        "schema_version": 1,
        "status": "pass",
        "record_type": "optimizer_graph_model_measurement",
        "model": model,
        "mode": "preflight",
        "context": context,
        "parameter_count": parameters,
        "parameter_tensors": tensors,
        "gradient_snapshot_matches": False,
        "caching_allocator_enabled": False,
        "graph_launched": False,
        "captured_nodes": 0,
    } or float(row.get("preparation_ms", 0.0)) <= 0.0:
        raise RuntimeError(f"invalid optimizer Graph preflight: {model}/T{context}")
    return row


def summarize(records: list[dict], runs: int) -> dict:
    comparisons = []
    for model, context in CASES:
        rows = [row for row in records
                if row["model"] == model and row["context"] == context]
        if len(rows) != runs:
            raise RuntimeError(f"incomplete preflight: {model}/T{context}")
        comparisons.append({
            "model": model,
            "context": context,
            "runs": runs,
            "gradient_snapshot_matches": all(
                row["gradient_snapshot_matches"] for row in rows),
            "caching_allocator_enabled": any(
                row["caching_allocator_enabled"] for row in rows),
            "graph_launches": sum(row["graph_launched"] for row in rows),
            "preparation_ms_median": statistics.median(
                float(row["preparation_ms"]) for row in rows),
        })
    gates = {
        "non_default_stream_disables_exact_size_pool": all(
            not row["caching_allocator_enabled"] for row in records),
        "all_model_context_snapshots_rejected": all(
            not row["gradient_snapshot_matches"] for row in records),
        "no_graph_launched_after_failed_snapshot": all(
            not row["graph_launched"] and row["captured_nodes"] == 0
            for row in records),
        "four_cases_repeat_three_times": len(records) == 12,
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "experiment": "optimizer_graph_model_preflight",
        "processes": len(records),
        "runs_per_case": runs,
        "comparisons": comparisons,
        "gates": gates,
        "decision": (
            "reject optimizer-only model Graph under permanent non-default "
            "Stream allocator disable; require Stream-aware retirement"),
    }


def main() -> int:
    args = options()
    records = []
    for process_run in range(1, args.runs + 1):
        ordered = CASES if process_run % 2 else tuple(reversed(CASES))
        for model, context in ordered:
            row = execute(args, model, context)
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
        print(f"optimizer_graph_model_preflight: {error}", file=sys.stderr)
        raise SystemExit(2)
