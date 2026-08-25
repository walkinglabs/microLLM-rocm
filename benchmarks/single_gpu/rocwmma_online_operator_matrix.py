#!/usr/bin/env python3
"""Validate the public online-Attention operator, native route and fallbacks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


CASES = (
    ("qwen-b1-t32", "qwen", 1, 14, 2, 32, 64, True),
    ("qwen-b2-t32", "qwen", 2, 14, 2, 32, 64, True),
    ("qwen-b1-t512", "qwen", 1, 14, 2, 512, 64, True),
    ("qwen-b2-t512", "qwen", 2, 14, 2, 512, 64, True),
    ("qwen-b1-t1024", "qwen", 1, 14, 2, 1024, 64, True),
    ("deep-b1-t32", "deepseek", 1, 12, 2, 32, 128, True),
    ("deep-b2-t32", "deepseek", 2, 12, 2, 32, 128, True),
    ("deep-b1-t512", "deepseek", 1, 12, 2, 512, 128, True),
    ("deep-b2-t512", "deepseek", 2, 12, 2, 512, 128, True),
    ("deep-b1-t1024", "deepseek", 1, 12, 2, 1024, 128, True),
    ("qwen-tail31", "qwen", 1, 14, 2, 31, 64, False),
    ("qwen-tail33", "qwen", 1, 14, 2, 33, 64, False),
    ("deep-b2-tail33", "deepseek", 2, 12, 2, 33, 128, False),
    ("width32-fallback", "synthetic", 1, 4, 2, 32, 32, False),
)


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("online operator benchmark emitted no JSON")


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--minimum-native-speedup", type=float, default=1.05)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0 or args.warmup < 0 or \
            args.repetitions <= 0 or args.minimum_native_speedup <= 1.0:
        parser.error("online operator matrix options are invalid")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for process_run in range(1, args.runs + 1):
        cases = list(CASES)
        if process_run % 2 == 0:
            cases.reverse()
        for case_index, case in enumerate(cases):
            name, family, batch, heads, kv_heads, sequence, width, native = case
            command = [
                str(args.binary), "--batch", str(batch), "--heads", str(heads),
                "--kv-heads", str(kv_heads), "--sequence", str(sequence),
                "--width", str(width), "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            record = last_json(completed.stdout)
            expected_calls = args.warmup + args.repetitions
            if record.get("status") != "pass" or not record.get("accuracy_passed") or \
                    bool(record.get("native_expected")) is not native or \
                    int(record.get("native_calls", -1)) != (expected_calls if native else 0) or \
                    int(record.get("fallback_calls", -1)) != (0 if native else expected_calls) or \
                    int(record.get("complete_output_elements", -1)) != \
                        batch * heads * sequence * width or \
                    int(record.get("candidate_global_score_bytes", -1)) != 0:
                raise RuntimeError("online operator routing record changed")
            numeric = (
                "candidate_max_error", "candidate_rms_error",
                "current_max_error", "current_rms_error",
                "candidate_event_ms_p50", "current_event_ms_p50",
                "candidate_over_current",
            )
            if any(not math.isfinite(float(record[field])) for field in numeric):
                raise RuntimeError("online operator emitted non-finite evidence")
            record.update({
                "record_type": "rocwmma_online_operator_measurement",
                "case": name,
                "family": family,
                "process_run": process_run,
                "case_order_index": case_index,
                "case_order": [item[0] for item in cases],
            })
            records.append(record)

    comparisons = []
    for name, family, batch, heads, kv_heads, sequence, width, native in CASES:
        selected = [row for row in records if row["case"] == name]
        candidate_ms = median(selected, "candidate_event_ms_p50")
        current_ms = median(selected, "current_event_ms_p50")
        comparisons.append({
            "case": name,
            "family": family,
            "batch": batch,
            "heads": heads,
            "kv_heads": kv_heads,
            "sequence": sequence,
            "width": width,
            "native": native,
            "runs": len(selected),
            "candidate_event_ms_p50": candidate_ms,
            "current_event_ms_p50": current_ms,
            "candidate_over_current": current_ms / candidate_ms,
            "maximum_candidate_error": max(float(row["candidate_max_error"])
                                             for row in selected),
            "maximum_candidate_rms_error": max(float(row["candidate_rms_error"])
                                                 for row in selected),
            "current_score_bytes": batch * heads * sequence * sequence * 4,
            "candidate_global_score_bytes": 0,
        })
    correctness_gate = all(
        row["maximum_candidate_error"] <= 2.0e-3 and
        row["maximum_candidate_rms_error"] <= 2.0e-4
        for row in comparisons)
    native_rows = [row for row in comparisons if row["native"]]
    fallback_rows = [row for row in comparisons if not row["native"]]
    native_performance_gate = all(
        row["candidate_over_current"] >= args.minimum_native_speedup
        for row in native_rows)
    routing_gate = all(
        (int(row["native_calls"]) == args.warmup + args.repetitions and
         int(row["fallback_calls"]) == 0) if row["native_expected"] else
        (int(row["native_calls"]) == 0 and
         int(row["fallback_calls"]) == args.warmup + args.repetitions)
        for row in records)
    memory_gate = all(row["candidate_global_score_bytes"] == 0
                      for row in native_rows)
    fallback_counterexample = any(
        row["candidate_over_current"] < 1.0 for row in fallback_rows)
    model_gate_admitted = (
        correctness_gate and native_performance_gate and routing_gate and memory_gate)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness_gate else "fail",
        "record_type": "rocwmma_online_operator_summary",
        "architecture": records[0]["architecture"],
        "processes": len(records),
        "runs_per_case": args.runs,
        "native_cases": len(native_rows),
        "fallback_cases": len(fallback_rows),
        "correctness_gate": correctness_gate,
        "native_performance_gate": native_performance_gate,
        "routing_gate": routing_gate,
        "memory_gate": memory_gate,
        "fallback_counterexample": fallback_counterexample,
        "model_gate_admitted": model_gate_admitted,
        "model_route_accepted": False,
        "comparisons": comparisons,
        "decision": (
            "admit explicit model-level gate; keep default model route disabled"
            if model_gate_admitted else "reject public online Attention operator"),
    }
    verification = {
        "schema_version": 1,
        "status": summary["status"],
        "raw_records": len(records),
        "expected_raw_records": len(CASES) * args.runs,
        "matrix_cases": len(comparisons),
        "native_cases": len(native_rows),
        "fallback_cases": len(fallback_rows),
        "all_complete_outputs": all(bool(row["accuracy_passed"]) for row in records),
        "routing_gate": routing_gate,
        "native_performance_gate": native_performance_gate,
        "fallback_counterexample": fallback_counterexample,
        "model_gate_admitted": model_gate_admitted,
        "model_route_accepted": False,
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
