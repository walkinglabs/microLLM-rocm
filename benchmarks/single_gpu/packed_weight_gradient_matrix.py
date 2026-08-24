#!/usr/bin/env python3
"""Run repeated official packed weight-gradient probes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


CASES = tuple((model, projection) for model in ("qwen", "deepseek")
              for projection in ("qkv", "gate-up"))


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    result = parser.parse_args()
    if not result.binary.is_file():
        parser.error(f"binary does not exist: {result.binary}")
    if result.runs < 3 or result.rows <= 0 or result.warmup < 0 or \
       result.repetitions <= 0:
        parser.error("matrix options are invalid")
    return result


def run(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("packed probe did not emit one JSON object") from error


def main() -> int:
    args = options()
    records = []
    for process_run in range(1, args.runs + 1):
        cases = CASES if process_run % 2 else tuple(reversed(CASES))
        for model, projection in cases:
            record = run([
                str(args.binary), "--model", model, "--projection", projection,
                "--rows", str(args.rows), "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ])
            if record.get("schema_version") != 1 or \
               record.get("status") != "pass" or \
               record.get("record_type") != "packed_weight_gradient_probe" or \
               record.get("model") != model or \
               record.get("projection") != projection or \
               record.get("rows") != args.rows or \
               record.get("pack_copies_per_step") != record.get("groups") or \
               int(record.get("packed_gradient_bytes", 0)) <= 0 or \
               int(record.get("packed_output_bytes", 0)) <= 0 or \
               not math.isfinite(float(record.get("event_speedup", 0))) or \
               float(record.get("maximum_absolute_error", math.inf)) > 2.0e-3:
                raise RuntimeError(f"invalid packed row: {model}/{projection}")
            record["process_run"] = process_run
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    comparisons = []
    for model, projection in CASES:
        rows = [row for row in records
                if row["model"] == model and row["projection"] == projection]
        speedup = statistics.median(float(row["event_speedup"]) for row in rows)
        comparisons.append({
            "model": model,
            "projection": projection,
            "runs": len(rows),
            "event_speedup_median": speedup,
            "performance_gate": speedup >= 1.05,
            "maximum_absolute_error": max(
                float(row["maximum_absolute_error"]) for row in rows),
        })
    passed = sum(row["performance_gate"] for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "packed_weight_gradient",
        "processes": len(records),
        "performance_cases_passed": passed,
        "performance_cases_total": len(comparisons),
        "comparisons": comparisons,
        "decision": ("continue packed weight-gradient model gate"
                     if passed == len(comparisons)
                     else "discard packed weight-gradient route"),
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
