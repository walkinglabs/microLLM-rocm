#!/usr/bin/env python3
"""Compare 256/128-thread causal-softmax kernels across official row shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


CASES = (
    ("qwen", 14, 256), ("qwen", 14, 512), ("qwen", 14, 1024),
    ("deepseek", 12, 256), ("deepseek", 12, 512),
    ("deepseek", 12, 1024),
)
POLICIES = (("threads256", "false"), ("threads128", "true"))


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("causal-softmax benchmark emitted no JSON")


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0 or args.warmup < 0 or \
            args.repetitions <= 0 or args.minimum_speedup <= 1:
        parser.error("causal-softmax comparison options are invalid")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        cases = list(CASES)
        policies = list(POLICIES)
        if process_run % 2 == 0:
            cases.reverse()
            policies.reverse()
        for family, heads, sequence in cases:
            for policy, enabled in policies:
                completed = subprocess.run([
                    str(args.binary), "--heads", str(heads),
                    "--sequence", str(sequence), "--threads-128", enabled,
                    "--warmup", str(args.warmup),
                    "--repetitions", str(args.repetitions),
                ], text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stdout + completed.stderr)
                record = last_json(completed.stdout)
                if record.get("status") != "pass" or \
                        int(record["threads"]) != (128 if enabled == "true" else 256):
                    raise RuntimeError("invalid causal-softmax operator record")
                record.update({
                    "record_type": "causal_softmax_thread_measurement",
                    "family": family, "policy": policy,
                    "process_run": process_run,
                    "case_order": [f"{item[0]}-t{item[2]}" for item in cases],
                    "policy_order": [item[0] for item in policies],
                })
                records.append(record)
    comparisons = []
    for family, heads, sequence in CASES:
        selected = [row for row in records if row["family"] == family and
                    int(row["sequence"]) == sequence]
        grouped = {policy: [row for row in selected if row["policy"] == policy]
                   for policy, _ in POLICIES}
        control = median(grouped["threads256"], "event_ms_p50")
        candidate = median(grouped["threads128"], "event_ms_p50")
        comparisons.append({
            "family": family, "heads": heads, "sequence": sequence,
            "threads256_event_ms": control,
            "threads128_event_ms": candidate,
            "event_speedup": control / candidate,
            "maximum_absolute_error": max(
                float(row["maximum_absolute_error"]) for row in selected),
            "maximum_rms_error": max(float(row["rms_error"])
                                     for row in selected),
        })
    correctness = all(row["maximum_absolute_error"] <= 2.0e-6 and
                      row["maximum_rms_error"] <= 1.0e-7
                      for row in comparisons)
    performance = all(row["event_speedup"] >= args.minimum_speedup
                      for row in comparisons)
    t512 = all(row["event_speedup"] >= args.minimum_speedup
               for row in comparisons if row["sequence"] == 512)
    summary = {
        "schema_version": 1, "status": "pass" if correctness else "fail",
        "record_type": "causal_softmax_thread_summary",
        "processes": len(records), "correctness_gate": correctness,
        "universal_performance_gate": performance,
        "t512_performance_gate": t512, "comparisons": comparisons,
        "decision": ("test T512-only model policy" if correctness and t512
                     else "reject 128-thread causal softmax"),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
