#!/usr/bin/env python3
"""Run the official FP32 grouped weight-gradient capability matrix."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


MODELS = ("qwen", "deepseek")
PROJECTIONS = ("qkv", "gate-up")
LAYOUTS = ("direct", "materialized")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--maximum-algorithms", type=int, default=16)
    parser.add_argument("--workspace-bytes", type=int, default=64 * 1024 * 1024)
    result = parser.parse_args()
    if not result.binary.is_file():
        parser.error(f"binary does not exist: {result.binary}")
    if result.rows <= 0 or result.warmup < 0 or result.repetitions <= 0 or \
       result.maximum_algorithms <= 0 or result.workspace_bytes < 0:
        parser.error("matrix numeric options are invalid")
    return result


def run(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("probe did not emit one JSON object") from error


def main() -> int:
    args = options()
    records = []
    for layout in LAYOUTS:
        for model in MODELS:
            for projection in PROJECTIONS:
                record = run([
                    str(args.binary), "--model", model,
                    "--projection", projection, "--input-layout", layout,
                    "--rows", str(args.rows), "--warmup", str(args.warmup),
                    "--repetitions", str(args.repetitions),
                    "--maximum-algorithms", str(args.maximum_algorithms),
                    "--workspace-bytes", str(args.workspace_bytes),
                ])
                if record.get("schema_version") != 1 or \
                   record.get("status") != "pass" or \
                   record.get("record_type") != "grouped_weight_gradient_probe" or \
                   record.get("model") != model or \
                   record.get("projection") != projection or \
                   record.get("input_layout") != layout or \
                   record.get("rows") != args.rows or \
                   int(record.get("algorithm_count", 0)) <= 0 or \
                   not math.isfinite(float(record.get("baseline_event_ms_p50", 0))) or \
                   float(record.get("baseline_event_ms_p50", 0)) <= 0:
                    raise RuntimeError(
                        f"invalid grouped weight-gradient row: {layout}/{model}/{projection}")
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    supported = sum(record.get("grouped_supported") is True for record in records)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "grouped_weight_gradient_capability",
        "cases": len(records),
        "supported_cases": supported,
        "unsupported_cases": len(records) - supported,
        "decision": ("continue grouped weight-gradient performance gate"
                     if supported else "discard grouped weight-gradient route"),
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
