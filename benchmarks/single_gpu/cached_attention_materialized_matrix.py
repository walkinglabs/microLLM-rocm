#!/usr/bin/env python3
"""Measure exact-order materialized-score cached Attention against current fused."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def csv_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def csv_strings(value: str) -> list[str]:
    result = [item for item in value.split(",") if item]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty list")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sequences", type=csv_ints, default=[512, 2048])
    parser.add_argument("--batches", type=csv_ints, default=[1, 2])
    parser.add_argument("--cache-dtypes", type=csv_strings, default=["fp32", "bf16"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 3 or result.repetitions <= 0 or
            result.heads <= 0 or result.kv_heads <= 0 or
            result.heads % result.kv_heads or result.width <= 0):
        parser.error("runs/repetitions/dimensions must be positive and warmup >= 3")
    if any(dtype not in {"fp32", "bf16"} for dtype in result.cache_dtypes):
        parser.error("cache dtypes must be fp32 or bf16")
    if max(result.sequences) > 4096:
        parser.error("materialized-score fused finalizer supports T <= 4096")
    if not result.benchmark.is_file():
        parser.error("benchmark executable does not exist")
    return result


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("benchmark emitted no JSON object")


def number(record: dict, field: str, positive: bool = False) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is missing or non-numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{field} is outside its numeric contract")
    return result


def validate(record: dict, args: argparse.Namespace, *, batch: int,
             sequence: int, dtype: str, order: str) -> None:
    expected = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "cached_attention_stage_probe",
        "batch": batch,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "sequence": sequence,
        "width": args.width,
        "cache_dtype": dtype,
        "order": order,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "materialized_score_bytes": batch * args.heads * sequence * 4,
        "materialized_bitwise_equal_current": True,
        "complete_output_accuracy_passed": True,
        "host_to_device_calls": 0,
        "device_to_host_calls": 0,
    }
    for field, wanted in expected.items():
        if record.get(field) != wanted:
            raise ValueError(f"{field} expected {wanted!r}, got {record.get(field)!r}")
    if not record.get("device_name") or not record.get("architecture"):
        raise ValueError("device identity is missing")
    for prefix in ("fused", "materialized"):
        for suffix in ("event_ms_p50", "event_ms_p95", "wall_ms_p50", "wall_ms_p95"):
            number(record, f"{prefix}_{suffix}", positive=True)
        number(record, f"{prefix}_allocation_calls_per_invocation")
        if number(record, f"{prefix}_backend_allocation_calls_per_invocation") != 0:
            raise ValueError(f"{prefix} reached backend allocation after warm-up")
        number(record, f"{prefix}_cache_reuse_calls_per_invocation")
    if record.get("materialized_allocation_calls_per_invocation") != 2:
        raise ValueError("materialized route no longer makes two logical outputs")
    if (number(record, "materialized_max_error") !=
            number(record, "fused_max_error") or
            number(record, "materialized_rms_error") !=
            number(record, "fused_rms_error")):
        raise ValueError("materialized and current error evidence diverged")
    number(record, "materialized_speedup_over_fused", positive=True)


def run_one(args: argparse.Namespace, batch: int, sequence: int,
            dtype: str, order: str) -> dict:
    completed = subprocess.run([
        str(args.benchmark), "--batch", str(batch),
        "--heads", str(args.heads), "--kv-heads", str(args.kv_heads),
        "--sequence", str(sequence), "--width", str(args.width),
        "--cache-dtype", dtype, "--materialized", "true",
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
        "--order", order,
    ], text=True, capture_output=True, timeout=300, check=True)
    record = last_json(completed.stdout)
    validate(record, args, batch=batch, sequence=sequence, dtype=dtype, order=order)
    return record


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["batch"], row["sequence"], row["cache_dtype"])].append(row)
    cases = []
    for key in sorted(groups, key=lambda item: (item[1], item[0], item[2])):
        samples = groups[key]
        fused_event = statistics.median(
            float(row["fused_event_ms_p50"]) for row in samples)
        candidate_event = statistics.median(
            float(row["materialized_event_ms_p50"]) for row in samples)
        fused_wall = statistics.median(
            float(row["fused_wall_ms_p50"]) for row in samples)
        candidate_wall = statistics.median(
            float(row["materialized_wall_ms_p50"]) for row in samples)
        case = {
            "batch": key[0], "sequence": key[1], "cache_dtype": key[2],
            "runs": len(samples),
            "orders": sorted({row["order"] for row in samples}),
            "fused_event_ms_p50": fused_event,
            "materialized_event_ms_p50": candidate_event,
            "event_speedup": fused_event / candidate_event,
            "fused_wall_ms_p50": fused_wall,
            "materialized_wall_ms_p50": candidate_wall,
            "wall_speedup": fused_wall / candidate_wall,
            "score_bytes": samples[0]["materialized_score_bytes"],
            "materialized_allocation_calls_per_invocation": 2,
            "materialized_backend_allocation_calls_per_invocation": 0,
            "bitwise_equal_current": all(
                row["materialized_bitwise_equal_current"] for row in samples),
            "materialized_event_range_ms": [
                min(float(row["materialized_event_ms_p50"]) for row in samples),
                max(float(row["materialized_event_ms_p50"]) for row in samples),
            ],
        }
        case["operator_gate_passed"] = (
            case["bitwise_equal_current"] and case["event_speedup"] >= 1.05)
        cases.append(case)
    return cases


def chart(cases: list[dict]) -> str:
    width, height = 1220, 120 + len(cases) * 68
    maximum = max(max(case["fused_event_ms_p50"],
                      case["materialized_event_ms_p50"]) for case in cases)
    scale = 760 / maximum
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:700}'
        '.muted{fill:#94a3b8}</style>',
        '<text x="28" y="34" class="title">Exact-order materialized-score cached Attention</text>',
        '<text x="28" y="58" class="small muted">MI300X Event median · every context is bitwise equal to current</text>',
    ]
    for index, case in enumerate(cases):
        y = 88 + index * 68
        label = f'B{case["batch"]} T{case["sequence"]} {case["cache_dtype"].upper()}'
        current = case["fused_event_ms_p50"] * scale
        candidate = case["materialized_event_ms_p50"] * scale
        color = "#22c55e" if case["operator_gate_passed"] else "#ef4444"
        parts.extend([
            f'<text x="28" y="{y + 21}" class="label">{label}</text>',
            f'<rect x="215" y="{y}" width="{current:.2f}" height="20" rx="3" fill="#64748b"/>',
            f'<rect x="215" y="{y + 27}" width="{candidate:.2f}" height="20" rx="3" fill="{color}"/>',
            f'<text x="1010" y="{y + 31}" class="label">{case["event_speedup"]:.3f}x</text>',
            f'<text x="1090" y="{y + 31}" class="small">{case["score_bytes"] / 1024:.1f} KiB</text>',
        ])
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    case_index = 0
    for sequence in args.sequences:
        for batch in args.batches:
            for dtype in args.cache_dtypes:
                for run in range(1, args.runs + 1):
                    order = "forward" if (case_index + run) % 2 else "reverse"
                    record = run_one(args, batch, sequence, dtype, order)
                    record["process_run"] = run
                    rows.append(record)
                    print(json.dumps({
                        "sequence": sequence, "batch": batch,
                        "cache_dtype": dtype, "process_run": run,
                        "order": order,
                        "event_speedup": record["materialized_speedup_over_fused"],
                    }, sort_keys=True), flush=True)
                case_index += 1
    cases = aggregate(rows)
    expected = len(args.sequences) * len(args.batches) * len(args.cache_dtypes)
    if len(cases) != expected or len(rows) != expected * args.runs:
        raise RuntimeError("materialized-score matrix is incomplete")
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "cached_attention_materialized_matrix",
        "matrix_complete": True,
        "process_rows": len(rows), "case_count": len(cases),
        "runs_per_case": args.runs, "warmup": args.warmup,
        "repetitions": args.repetitions,
        "heads": args.heads, "kv_heads": args.kv_heads, "width": args.width,
        "device_name": rows[0]["device_name"],
        "architecture": rows[0]["architecture"],
        "all_bitwise_equal_current": all(
            case["bitwise_equal_current"] for case in cases),
        "zero_payload_transfers": True,
        "zero_warm_backend_allocations": True,
        "all_cases_pass_operator_gate": all(
            case["operator_gate_passed"] for case in cases),
        "minimum_event_speedup": min(case["event_speedup"] for case in cases),
        "maximum_event_speedup": max(case["event_speedup"] for case in cases),
        "minimum_wall_speedup": min(case["wall_speedup"] for case in cases),
        "maximum_wall_speedup": max(case["wall_speedup"] for case in cases),
        "cases": cases,
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "comparison.svg").write_text(
        chart(cases), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"cached_attention_materialized_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
