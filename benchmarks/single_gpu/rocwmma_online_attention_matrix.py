#!/usr/bin/env python3
"""Run the bounded rocWMMA online-Attention prototype matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


SEQUENCES = (32, 64, 128, 256, 512, 1024, 2048)
CONFIGS = (
    ("qwen", 14, 2, 64),
    ("deepseek", 12, 2, 128),
)


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("online Attention benchmark emitted no JSON")


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--minimum-current-speedup", type=float, default=1.05)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0 or args.warmup < 0 or \
            args.repetitions <= 0 or args.minimum_current_speedup <= 1.0:
        parser.error("online Attention matrix options are invalid")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    base_cases = [(family, heads, kv_heads, width, sequence)
                  for family, heads, kv_heads, width in CONFIGS
                  for sequence in SEQUENCES]
    records: list[dict] = []
    for process_run in range(1, args.runs + 1):
        cases = list(base_cases)
        if process_run % 2 == 0:
            cases.reverse()
        for case_index, (family, heads, kv_heads, width, sequence) in enumerate(cases):
            command = [
                str(args.binary), "--sequence", str(sequence),
                "--heads", str(heads), "--kv-heads", str(kv_heads),
                "--width", str(width), "--worker-threads", "512",
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"online Attention process failed: {' '.join(command)}\n"
                    f"{completed.stdout}\n{completed.stderr}")
            record = last_json(completed.stdout)
            required = (
                "online_event_ms_p50", "scalar_event_ms_p50",
                "current_event_ms_p50", "online_max_error", "online_rms_error",
                "current_max_error", "current_rms_error", "current_score_bytes",
                "complete_output_elements",
            )
            if record.get("status") != "pass" or not record.get("accuracy_passed") or \
                    any(field not in record for field in required) or \
                    int(record.get("sequence", -1)) != sequence or \
                    int(record.get("heads", -1)) != heads or \
                    int(record.get("kv_heads", -1)) != kv_heads or \
                    int(record.get("width", -1)) != width or \
                    int(record.get("worker_threads", -1)) != 512 or \
                    record.get("pv_path") != "rocwmma_bf16" or \
                    int(record.get("global_score_bytes", -1)) != 0:
                raise RuntimeError("online Attention benchmark record contract changed")
            if any(not math.isfinite(float(record[field])) for field in required):
                raise RuntimeError("online Attention benchmark emitted non-finite evidence")
            record.update({
                "record_type": "rocwmma_online_attention_measurement",
                "family": family,
                "process_run": process_run,
                "case_order_index": case_index,
                "case_order": [f"{case[0]}-t{case[4]}" for case in cases],
            })
            records.append(record)

    comparisons = []
    for family, heads, kv_heads, width, sequence in base_cases:
        selected = [row for row in records if row["family"] == family and
                    int(row["sequence"]) == sequence]
        online_ms = median(selected, "online_event_ms_p50")
        scalar_ms = median(selected, "scalar_event_ms_p50")
        current_ms = median(selected, "current_event_ms_p50")
        comparisons.append({
            "family": family,
            "heads": heads,
            "kv_heads": kv_heads,
            "width": width,
            "sequence": sequence,
            "runs": len(selected),
            "complete_output_elements": heads * sequence * width,
            "online_event_ms_p50": online_ms,
            "scalar_event_ms_p50": scalar_ms,
            "current_event_ms_p50": current_ms,
            "online_over_scalar": scalar_ms / online_ms,
            "online_over_current": current_ms / online_ms,
            "maximum_online_error": max(float(row["online_max_error"])
                                         for row in selected),
            "maximum_online_rms_error": max(float(row["online_rms_error"])
                                             for row in selected),
            "maximum_current_error": max(float(row["current_max_error"])
                                          for row in selected),
            "current_score_bytes": heads * sequence * sequence * 4,
            "online_global_score_bytes": 0,
        })

    correctness_gate = all(
        row["maximum_online_error"] <= 2.0e-3 and
        row["maximum_online_rms_error"] <= 2.0e-4 and
        row["maximum_current_error"] <= 3.0e-4
        for row in comparisons)
    current_performance_gate = all(
        row["online_over_current"] >= args.minimum_current_speedup
        for row in comparisons)
    long_scalar_gate = all(
        row["online_over_scalar"] >= 1.0 for row in comparisons
        if row["sequence"] >= 1024)
    short_scalar_counterexample = any(
        row["online_over_scalar"] < 1.0 for row in comparisons
        if row["sequence"] <= 512)
    memory_gate = all(
        row["online_global_score_bytes"] == 0 and row["current_score_bytes"] > 0
        for row in comparisons)
    operator_integration_admitted = (
        correctness_gate and current_performance_gate and memory_gate)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness_gate else "fail",
        "record_type": "rocwmma_online_attention_summary",
        "architecture": records[0]["architecture"],
        "rocwmma_version": records[0]["rocwmma_version"],
        "processes": len(records),
        "runs_per_case": args.runs,
        "sequences": list(SEQUENCES),
        "families": [config[0] for config in CONFIGS],
        "correctness_gate": correctness_gate,
        "current_performance_gate": current_performance_gate,
        "long_scalar_gate": long_scalar_gate,
        "short_scalar_counterexample": short_scalar_counterexample,
        "memory_gate": memory_gate,
        "batch_support": False,
        "tail_support": False,
        "operator_integration_admitted": operator_integration_admitted,
        "model_route_accepted": False,
        "comparisons": comparisons,
        "decision": (
            "admit public operator integration with explicit fallback; keep model route disabled"
            if operator_integration_admitted else
            "reject rocWMMA online Attention prototype"),
    }
    verification = {
        "schema_version": 1,
        "status": summary["status"],
        "raw_records": len(records),
        "expected_raw_records": len(base_cases) * args.runs,
        "matrix_cases": len(comparisons),
        "expected_matrix_cases": len(base_cases),
        "all_complete_outputs": all(
            int(row["complete_output_elements"]) ==
            int(row["heads"]) * int(row["sequence"]) * int(row["width"])
            for row in records),
        "all_accuracy_passed": all(bool(row["accuracy_passed"]) for row in records),
        "current_performance_gate": current_performance_gate,
        "memory_gate": memory_gate,
        "operator_integration_admitted": operator_integration_admitted,
        "model_route_accepted": False,
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
