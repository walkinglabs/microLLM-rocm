#!/usr/bin/env python3
"""Run the baseline/external Autograd leaf-pool gate in rotated fresh processes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def run_case(binary: Path, model: str, context: int, first: str,
             warmup: int, repetitions: int) -> dict:
    command = [
        str(binary), "--model", model, "--context", str(context),
        "--first", first, "--warmup", str(warmup),
        "--repetitions", str(repetitions),
    ]
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"case failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}")
    record = json.loads(completed.stdout.strip())
    if record.get("status") != "pass":
        raise RuntimeError(f"case did not pass: {record}")
    return record


def summarize(records: list[dict]) -> dict:
    groups: list[dict] = []
    for model, context in (("tiny", 8), ("model-s", 8), ("model-s", 32)):
        selected = [row for row in records
                    if row["model"] == model and row["context"] == context]
        event_speedups = [row["baseline"]["event_median_ms"] /
                          row["external"]["event_median_ms"]
                          for row in selected]
        wall_speedups = [row["baseline"]["wall_median_ms"] /
                         row["external"]["wall_median_ms"]
                         for row in selected]
        baseline_peaks = [row["baseline"]["peak_extra_bytes"] for row in selected]
        external_peaks = [row["external"]["peak_extra_bytes"] for row in selected]
        groups.append({
            "model": model,
            "context": context,
            "processes": len(selected),
            "maximum_gradient_error": max(
                row["maximum_gradient_error"] for row in selected),
            "maximum_rms_gradient_error": max(
                row["rms_gradient_error"] for row in selected),
            "all_addresses_stable": all(
                row["all_external_addresses_stable"] for row in selected),
            "compared_gradient_elements": selected[0]["compared_gradient_elements"],
            "parameter_tensors": selected[0]["external"]["parameter_tensors"],
            "pool_bytes": selected[0]["external"]["pool_bytes"],
            "event_speedup_median": statistics.median(event_speedups),
            "event_speedup_minimum": min(event_speedups),
            "event_speedup_maximum": max(event_speedups),
            "wall_speedup_median": statistics.median(wall_speedups),
            "wall_speedup_minimum": min(wall_speedups),
            "wall_speedup_maximum": max(wall_speedups),
            "baseline_peak_extra_bytes_median": statistics.median(baseline_peaks),
            "external_peak_extra_bytes_median": statistics.median(external_peaks),
            "peak_extra_bytes_delta_median": (
                statistics.median(external_peaks) -
                statistics.median(baseline_peaks)),
            "baseline_allocation_calls_median": statistics.median(
                row["baseline"]["allocation_calls"] for row in selected),
            "external_allocation_calls_median": statistics.median(
                row["external"]["allocation_calls"] for row in selected),
        })
    correctness = all(
        group["maximum_gradient_error"] == 0.0 and
        group["maximum_rms_gradient_error"] == 0.0 and
        group["all_addresses_stable"] for group in groups)
    performance_pass = all(
        group["event_speedup_median"] >= 1.0 and
        group["wall_speedup_median"] >= 1.0 and
        group["peak_extra_bytes_delta_median"] <= 0 for group in groups)
    return {
        "schema_version": 1,
        "status": "pass" if correctness else "fail",
        "record_type": "external_gradient_pool_matrix_summary",
        "correctness_pass": correctness,
        "performance_pass": performance_pass,
        "decision": ("admit_model_policy" if performance_pass
                     else "keep_explicit_interop_only"),
        "groups": groups,
    }


def main() -> int:
    args = arguments()
    if args.runs < 1 or args.repetitions < 1 or args.warmup < 1:
        raise ValueError("runs, repetitions and warmup must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for run in range(args.runs):
        for model, context in (("tiny", 8), ("model-s", 8), ("model-s", 32)):
            for first in ("baseline", "external"):
                record = run_case(args.binary, model, context, first,
                                  args.warmup, args.repetitions)
                record["matrix_run"] = run + 1
                records.append(record)
    raw_path = args.output / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    summary = summarize(records)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification = {
        "schema_version": 1,
        "status": "pass" if summary["correctness_pass"] else "fail",
        "processes": len(records),
        "rotated_orders": sorted({row["first"] for row in records}),
        "models": sorted({row["model"] for row in records}),
        "contexts": sorted({row["context"] for row in records}),
        "all_raw_records_pass": all(row["status"] == "pass" for row in records),
        "all_gradients_exact": all(
            row["maximum_gradient_error"] == 0.0 and
            row["rms_gradient_error"] == 0.0 for row in records),
        "all_addresses_stable": all(
            row["all_external_addresses_stable"] for row in records),
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["correctness_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

