#!/usr/bin/env python3
"""Repeated exact-shape gate for direct BF16 RMSNorm output."""

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
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    result = parser.parse_args()
    if (not result.binary.is_file() or result.runs <= 0 or
            result.rows <= 0 or result.rows > 4096 or result.warmup < 0 or
            result.repetitions <= 0):
        parser.error("BF16 RMSNorm matrix options are invalid")
    return result


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("BF16 RMSNorm probe emitted no JSON")


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        order = list(MODELS)
        if process_run % 2 == 0:
            order.reverse()
        for model in order:
            completed = subprocess.run([
                str(args.binary), "--model", model, "--rows", str(args.rows),
                "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions),
            ], text=True, capture_output=True, check=False)
            stem = f"{model}-p{process_run}"
            (logs / f"{stem}.stdout.txt").write_text(
                completed.stdout, encoding="utf-8")
            (logs / f"{stem}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"{stem} failed: {completed.stderr}")
            row = last_json(completed.stdout)
            if (row.get("status") != "pass" or
                    row.get("record_type") != "bf16_rms_norm_output_probe" or
                    row.get("model") != model or row.get("rows") != args.rows or
                    row.get("complete_output_equal") is not True):
                raise RuntimeError(f"{stem} violated the probe contract")
            row.update({"process_run": process_run, "process_order": order})
            records.append(row)

    comparisons = []
    for model in MODELS:
        rows = [row for row in records if row["model"] == model]
        comparisons.append({
            "model": model,
            "rows": args.rows,
            "width": int(rows[0]["width"]),
            "runs": args.runs,
            "complete_output_equal": all(
                row["complete_output_equal"] for row in rows),
            "event_speedup_median": statistics.median(
                float(row["event_speedup"]) for row in rows),
            "wall_speedup_median": statistics.median(
                float(row["wall_speedup"]) for row in rows),
            "host_to_device_calls": max(
                int(row["host_to_device_calls"]) for row in rows),
            "device_to_host_calls": max(
                int(row["device_to_host_calls"]) for row in rows),
        })
    keep = all(
        row["complete_output_equal"] and row["event_speedup_median"] >= 1.2 and
        row["wall_speedup_median"] >= 1.1 and
        row["host_to_device_calls"] == 0 and row["device_to_host_calls"] == 0
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "bf16_rms_norm_output_matrix_summary",
        "raw_processes": len(records),
        "comparisons": comparisons,
        "operator_gate_passed": keep,
        "decision": "admit FFN model route" if keep else "reject fused output",
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if keep else 2


if __name__ == "__main__":
    raise SystemExit(main())
