#!/usr/bin/env python3
"""Screen repeatable FP32 gate/up weight-gradient solution indices."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


MODELS = ("qwen", "deepseek")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--maximum-algorithms", type=int, default=64)
    parser.add_argument("--workspace-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    result = parser.parse_args()
    if not result.binary.is_file():
        parser.error(f"binary does not exist: {result.binary}")
    if result.runs < 3 or result.rows <= 0 or \
       result.maximum_algorithms <= 0 or result.workspace_bytes < 0 or \
       result.warmup < 0 or result.repetitions <= 0:
        parser.error("solution matrix options are invalid")
    return result


def run(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(completed.stdout)


def main() -> int:
    args = options()
    records = []
    for process_run in range(1, args.runs + 1):
        models = MODELS if process_run % 2 else tuple(reversed(MODELS))
        for model in models:
            record = run([
                str(args.binary), "--model", model,
                "--operation", "gate-up", "--rows", str(args.rows),
                "--maximum-algorithms", str(args.maximum_algorithms),
                "--workspace-bytes", str(args.workspace_bytes),
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ])
            candidates = record.get("candidates", [])
            if record.get("schema_version") != 1 or \
               record.get("status") != "pass" or \
               record.get("record_type") != "fp32_weight_gradient_algorithm_tune" or \
               record.get("model") != model or \
               record.get("operation") != "gate-up" or \
               record.get("rows") != args.rows or \
               len(candidates) != args.maximum_algorithms or \
               any(candidate.get("correctness_passed") is not True or
                   candidate.get("finite") is not True
                   for candidate in candidates):
                raise RuntimeError(f"invalid FP32 weight-gradient tuner row: {model}")
            record["process_run"] = process_run
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    summaries = []
    for model in MODELS:
        model_rows = [row for row in records if row["model"] == model]
        by_run = [{candidate["index"]: candidate for candidate in row["candidates"]}
                  for row in model_rows]
        common = set(by_run[0])
        for candidates in by_run[1:]:
            common &= set(candidates)
        ranked = []
        for index in common:
            speedups = [float(candidates[index]["event_speedup_vs_default"])
                        for candidates in by_run]
            ranked.append({
                "index": index,
                "median_speedup": statistics.median(speedups),
                "minimum_speedup": min(speedups),
            })
        ranked.sort(key=lambda row: (-row["median_speedup"], row["index"]))
        winner = ranked[0] if ranked else {
            "index": -1, "median_speedup": 0.0, "minimum_speedup": 0.0}
        summaries.append({
            "model": model,
            "runs": len(model_rows),
            "common_candidate_count": len(common),
            "selected_index": winner["index"],
            "selected_median_speedup": winner["median_speedup"],
            "selected_minimum_speedup": winner["minimum_speedup"],
            "performance_gate": winner["median_speedup"] >= 1.05,
            "top_common_candidates": ranked[:8],
        })
    passed = all(row["performance_gate"] for row in summaries)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "fp32_weight_gradient_solution_matrix",
        "processes": len(records),
        "candidate_evaluations": sum(len(row["candidates"]) for row in records),
        "model_gate_ready": passed,
        "summaries": summaries,
        "decision": ("continue exact-solution model gate" if passed
                     else "discard unstable FP32 weight-gradient solutions"),
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
