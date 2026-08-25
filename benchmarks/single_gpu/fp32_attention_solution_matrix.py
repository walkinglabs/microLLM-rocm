#!/usr/bin/env python3
"""Repeat FP32 Attention QK/PV solution screening in fresh processes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


CASES = (("qwen", "qk"), ("qwen", "pv"),
         ("deepseek", "qk"), ("deepseek", "pv"))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--maximum-algorithms", type=int, default=64)
    parser.add_argument("--workspace-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    result = parser.parse_args()
    if (result.runs <= 0 or result.sequence <= 0 or result.sequence > 4096 or
            result.maximum_algorithms <= 0 or
            result.workspace_bytes < 0 or result.warmup < 0 or
            result.repetitions <= 0 or not result.binary.is_file()):
        parser.error("runner arguments are outside the safe contract")
    return result


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("tuner emitted no JSON object")


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    raw: list[dict] = []
    for model, operation in CASES:
        for process_run in range(1, args.runs + 1):
            command = [
                str(args.binary), "--model", model,
                "--operation", operation, "--sequence", str(args.sequence),
                "--maximum-algorithms", str(args.maximum_algorithms),
                "--workspace-bytes", str(args.workspace_bytes),
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ]
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False)
            stem = f"{model}-{operation}-p{process_run}"
            (logs / f"{stem}.stdout.txt").write_text(
                completed.stdout, encoding="utf-8")
            (logs / f"{stem}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"tuner failed for {stem}: {completed.stderr}")
            record = last_json(completed.stdout)
            if (record.get("status") != "pass" or
                    record.get("record_type") !=
                        "fp32_attention_algorithm_tune" or
                    not record.get("candidates")):
                raise RuntimeError(f"invalid tuner record: {stem}")
            record["process_run"] = process_run
            raw.append(record)

    comparisons = []
    for model, operation in CASES:
        selected = [row for row in raw
                    if row["model"] == model and row["operation"] == operation]
        by_run = []
        for record in selected:
            by_run.append({int(row["index"]): row for row in record["candidates"]
                           if row["correctness_passed"]})
        common = set(by_run[0])
        for values in by_run[1:]:
            common.intersection_update(values)
        candidates = []
        for index in sorted(common):
            events = [float(values[index]["event_ms_p50"]) for values in by_run]
            walls = [float(values[index]["wall_ms_p50"]) for values in by_run]
            maximum = max(float(values[index]["maximum_absolute_error"])
                          for values in by_run)
            rms = max(float(values[index]["rms_error"]) for values in by_run)
            candidates.append({
                "index": index,
                "event_ms_p50_median": statistics.median(events),
                "wall_ms_p50_median": statistics.median(walls),
                "maximum_absolute_error": maximum,
                "maximum_rms_error": rms,
                "workspace_bytes": max(
                    int(values[index]["workspace_bytes"]) for values in by_run),
            })
        default_event = statistics.median(
            float(row["default_event_ms_p50"]) for row in selected)
        default_wall = statistics.median(
            float(row["default_wall_ms_p50"]) for row in selected)
        candidates.sort(key=lambda row: (row["event_ms_p50_median"], row["index"]))
        recommended = candidates[0] if candidates else None
        comparisons.append({
            "model": model, "operation": operation,
            "sequence": args.sequence,
            "raw_candidate_count_min": min(int(row["candidate_count"])
                                           for row in selected),
            "common_passing_candidates": len(candidates),
            "default_event_ms_p50_median": default_event,
            "default_wall_ms_p50_median": default_wall,
            "recommended_index": recommended["index"] if recommended else -1,
            "recommended_event_ms_p50_median":
                recommended["event_ms_p50_median"] if recommended else 0.0,
            "recommended_wall_ms_p50_median":
                recommended["wall_ms_p50_median"] if recommended else 0.0,
            "recommended_event_speedup":
                default_event / recommended["event_ms_p50_median"]
                if recommended else 0.0,
            "recommended_maximum_absolute_error":
                recommended["maximum_absolute_error"] if recommended else 0.0,
            "recommended_maximum_rms_error":
                recommended["maximum_rms_error"] if recommended else 0.0,
            "recommended_workspace_bytes":
                recommended["workspace_bytes"] if recommended else 0,
            "candidates": candidates,
        })
    keep = sum(row["recommended_event_speedup"] >= 1.05
               for row in comparisons)
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "fp32_attention_solution_matrix_summary",
        "raw_processes": len(raw), "comparisons": comparisons,
        "keep_rows": keep,
        "decision": ("register exact FP32 Attention candidates"
                     if keep == len(comparisons) else
                     "reject incomplete FP32 Attention solution policy"),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in raw:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
