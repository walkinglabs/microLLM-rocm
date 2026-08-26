#!/usr/bin/env python3
"""Gate native 128-lane materialized-score Attention finalization."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


SEQUENCES = (512, 2048)
BATCHES = (1, 2)
DTYPES = ("fp32", "bf16")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if (not args.benchmark.is_file() or args.runs != 2 or
            args.warmup < 3 or args.repetitions <= 0 or
            args.timeout_seconds <= 0):
        parser.error("native128 finalize inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("native128 benchmark emitted no JSON")


def run_one(args: argparse.Namespace, sequence: int, batch: int,
            dtype: str, process_run: int, ordinal: int) -> dict:
    order = "forward" if (ordinal + process_run) % 2 else "reverse"
    completed = subprocess.run([
        str(args.benchmark), "--batch", str(batch), "--heads", "12",
        "--kv-heads", "2", "--sequence", str(sequence),
        "--width", "128", "--cache-dtype", dtype,
        "--materialized", "true", "--native128", "true",
        "--finalize-threads", "256", "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions), "--order", order,
    ], text=True, capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    row = last_json(completed.stdout)
    expected = {
        "status": "pass", "record_type": "cached_attention_stage_probe",
        "batch": batch, "heads": 12, "kv_heads": 2,
        "sequence": sequence, "width": 128, "cache_dtype": dtype,
        "materialized_finalize_threads": 256,
        "complete_output_accuracy_passed": True,
        "host_to_device_calls": 0, "device_to_host_calls": 0,
    }
    for name, wanted in expected.items():
        if row.get(name) != wanted:
            raise ValueError(
                f"native128 {name} expected {wanted!r}, got {row.get(name)!r}")
    for name in ("materialized_event_ms_p50", "materialized_wall_ms_p50",
                 "native128_event_ms_p50", "native128_wall_ms_p50"):
        value = float(row.get(name, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"native128 {name} is invalid")
    if (not math.isfinite(float(row.get("native128_max_error", math.nan))) or
            not math.isfinite(float(row.get("native128_rms_error", math.nan))) or
            float(row["native128_max_error"]) > 8.0e-4 or
            float(row["native128_rms_error"]) > 8.0e-5 or
            row.get("native128_backend_allocation_calls_per_invocation") != 0 or
            row.get("native128_allocation_calls_per_invocation") != 2):
        raise ValueError("native128 accuracy or resource gate changed")
    row.update({
        "process_run": process_run,
        "native128_event_speedup":
            row["materialized_event_ms_p50"] / row["native128_event_ms_p50"],
        "native128_wall_speedup":
            row["materialized_wall_ms_p50"] / row["native128_wall_ms_p50"],
    })
    return row


def summarize(rows: list[dict]) -> dict:
    cases = []
    for sequence in SEQUENCES:
        for batch in BATCHES:
            for dtype in DTYPES:
                selected = [row for row in rows
                            if row["sequence"] == sequence and
                            row["batch"] == batch and
                            row["cache_dtype"] == dtype]
                if len(selected) != 2:
                    raise ValueError("native128 process matrix is incomplete")
                cases.append({
                    "sequence": sequence, "batch": batch,
                    "cache_dtype": dtype, "runs": 2,
                    "maximum_error": max(float(row["native128_max_error"])
                                         for row in selected),
                    "maximum_rms_error": max(float(row["native128_rms_error"])
                                             for row in selected),
                    "all_finite": True,
                    "all_bitwise_equal_materialized": all(
                        row["native128_bitwise_equal_materialized"]
                        for row in selected),
                    "event_speedup": statistics.median(
                        row["native128_event_speedup"] for row in selected),
                    "wall_speedup": statistics.median(
                        row["native128_wall_speedup"] for row in selected),
                    "zero_backend_allocations": all(
                        row["native128_backend_allocation_calls_per_invocation"] == 0
                        for row in selected),
                })
    t2048 = [case for case in cases if case["sequence"] == 2048]
    return {
        "schema_version": 1,
        "record_type": "native128_finalize_matrix",
        "status": "pass", "process_rows": len(rows),
        "case_count": len(cases), "runs_per_case": 2,
        "sequences": list(SEQUENCES), "batches": list(BATCHES),
        "cache_dtypes": list(DTYPES),
        "all_accuracy_gates_passed": True,
        "minimum_t2048_event_speedup": min(
            case["event_speedup"] for case in t2048),
        "minimum_t2048_wall_speedup": min(
            case["wall_speedup"] for case in t2048),
        "t2048_performance_pass_count": sum(
            case["event_speedup"] >= 1.05 and case["wall_speedup"] >= 1.02
            for case in t2048),
        "candidate_admitted": all(
            case["event_speedup"] >= 1.05 and case["wall_speedup"] >= 1.02
            for case in t2048),
        "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1250, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,monospace;fill:#e5e7eb}.title{font-size:23px;font-weight:700}.label{font-size:13px}.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="35" y="42" class="title">Native 128-lane finalize gate</text>',
        '<text x="35" y="67" class="muted">candidate/current · green requires T2048 Event 1.05x and wall 1.02x</text>',
    ]
    for index, case in enumerate(summary["cases"]):
        y = 105 + index * 47
        color = ("#22c55e" if case["sequence"] == 2048 and
                 case["event_speedup"] >= 1.05 and
                 case["wall_speedup"] >= 1.02 else "#ef4444")
        parts.extend((
            f'<text x="40" y="{y + 18}" class="label">T{case["sequence"]} B{case["batch"]} {case["cache_dtype"].upper()}</text>',
            f'<rect x="350" y="{y}" width="{max(2.0, case["event_speedup"] * 300):.2f}" height="24" rx="4" fill="{color}"/>',
            f'<text x="690" y="{y + 18}" class="label">Event {case["event_speedup"]:.3f}x · wall {case["wall_speedup"]:.3f}x · Max {case["maximum_error"]:.2e}</text>',
        ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    ordinal = 0
    for sequence in SEQUENCES:
        for batch in BATCHES:
            for dtype in DTYPES:
                for run in (1, 2):
                    row = run_one(args, sequence, batch, dtype, run, ordinal)
                    rows.append(row)
                    print(json.dumps({
                        "sequence": sequence, "batch": batch,
                        "cache_dtype": dtype, "process_run": run,
                        "event_speedup": row["native128_event_speedup"],
                    }, sort_keys=True), flush=True)
                ordinal += 1
    summary = summarize(rows)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "native128.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "cases"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"native128_finalize_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
