#!/usr/bin/env python3
"""Search split counts for the two-stage cached-decode Attention candidate."""

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
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def csv_strings(value: str) -> list[str]:
    values = [item for item in value.split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty list")
    return values


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sequences", type=csv_ints, default=[512, 2048])
    parser.add_argument("--batches", type=csv_ints, default=[1, 2])
    parser.add_argument("--cache-dtypes", type=csv_strings, default=["fp32", "bf16"])
    parser.add_argument("--splits", type=csv_ints, default=[1, 2, 4, 8, 16, 32])
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
    if max(result.splits) > min(result.sequences) or max(result.splits) > 32:
        parser.error("every split count must fit every sequence and be <= 32")
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


def number(record: dict, field: str, *, positive: bool = False) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is missing or non-numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{field} is outside its numeric contract")
    return result


def validate(record: dict, args: argparse.Namespace, *, batch: int,
             sequence: int, dtype: str, splits: int, order: str) -> None:
    partial_bytes = batch * args.heads * splits * (args.width + 2) * 4
    expected = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "cached_attention_stage_probe",
        "batch": batch,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "sequence": sequence,
        "width": args.width,
        "repeats": args.heads // args.kv_heads,
        "cache_dtype": dtype,
        "splits": splits,
        "order": order,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "split_partial_blocks": batch * args.heads * splits,
        "split_combine_blocks": batch * args.heads,
        "split_partial_bytes": partial_bytes,
        "complete_output_accuracy_passed": True,
        "host_to_device_calls": 0,
        "device_to_host_calls": 0,
    }
    for field, wanted in expected.items():
        if record.get(field) != wanted:
            raise ValueError(f"{field} expected {wanted!r}, got {record.get(field)!r}")
    if not record.get("device_name") or not record.get("architecture"):
        raise ValueError("device identity is missing")
    for prefix in ("fused", "split"):
        for suffix in ("event_ms_p50", "event_ms_p95", "wall_ms_p50", "wall_ms_p95"):
            number(record, f"{prefix}_{suffix}", positive=True)
        number(record, f"{prefix}_allocation_calls_per_invocation")
        if number(record, f"{prefix}_backend_allocation_calls_per_invocation") != 0:
            raise ValueError(f"{prefix} reached the backend allocator after warm-up")
        number(record, f"{prefix}_cache_reuse_calls_per_invocation")
    if number(record, "split_max_error") > 8.0e-4 or \
            number(record, "split_rms_error") > 8.0e-5:
        raise ValueError("split complete-output error exceeds the gate")
    number(record, "fused_max_error")
    number(record, "fused_rms_error")
    number(record, "split_speedup_over_fused", positive=True)


def run_one(args: argparse.Namespace, batch: int, sequence: int, dtype: str,
            splits: int, order: str) -> dict:
    completed = subprocess.run([
        str(args.benchmark), "--batch", str(batch),
        "--heads", str(args.heads), "--kv-heads", str(args.kv_heads),
        "--sequence", str(sequence), "--width", str(args.width),
        "--cache-dtype", dtype, "--splits", str(splits),
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
        "--order", order,
    ], check=True, text=True, capture_output=True, timeout=300)
    record = last_json(completed.stdout)
    validate(record, args, batch=batch, sequence=sequence, dtype=dtype,
             splits=splits, order=order)
    return record


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[int, int, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["batch"], row["sequence"], row["cache_dtype"],
                row["splits"])].append(row)
    candidates = []
    for key in sorted(groups, key=lambda item: (item[1], item[0], item[2], item[3])):
        samples = groups[key]
        fused = statistics.median(float(row["fused_event_ms_p50"]) for row in samples)
        split = statistics.median(float(row["split_event_ms_p50"]) for row in samples)
        candidate = {
            "batch": key[0], "sequence": key[1], "cache_dtype": key[2],
            "splits": key[3], "runs": len(samples),
            "orders": sorted({row["order"] for row in samples}),
            "fused_event_ms_p50": fused,
            "split_event_ms_p50": split,
            "fused_wall_ms_p50": statistics.median(
                float(row["fused_wall_ms_p50"]) for row in samples),
            "split_wall_ms_p50": statistics.median(
                float(row["split_wall_ms_p50"]) for row in samples),
            "event_speedup": fused / split,
            "wall_speedup": statistics.median(
                float(row["fused_wall_ms_p50"]) for row in samples) /
                statistics.median(float(row["split_wall_ms_p50"]) for row in samples),
            "maximum_split_error": max(float(row["split_max_error"]) for row in samples),
            "maximum_split_rms_error": max(
                float(row["split_rms_error"]) for row in samples),
            "partial_blocks": samples[0]["split_partial_blocks"],
            "combine_blocks": samples[0]["split_combine_blocks"],
            "partial_bytes": samples[0]["split_partial_bytes"],
            "split_allocation_calls_per_invocation": statistics.median(
                float(row["split_allocation_calls_per_invocation"])
                for row in samples),
            "split_backend_allocation_calls_per_invocation": max(
                float(row["split_backend_allocation_calls_per_invocation"])
                for row in samples),
            "split_event_range_ms": [
                min(float(row["split_event_ms_p50"]) for row in samples),
                max(float(row["split_event_ms_p50"]) for row in samples),
            ],
        }
        candidate["operator_gate_passed"] = (
            candidate["event_speedup"] >= 1.05 and
            candidate["maximum_split_error"] <= 8.0e-4 and
            candidate["maximum_split_rms_error"] <= 8.0e-5)
        candidates.append(candidate)
    case_groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for candidate in candidates:
        case_groups[(candidate["batch"], candidate["sequence"],
                     candidate["cache_dtype"])].append(candidate)
    winners = []
    for key in sorted(case_groups, key=lambda item: (item[1], item[0], item[2])):
        winner = max(case_groups[key], key=lambda row: row["event_speedup"])
        winners.append({
            "batch": key[0], "sequence": key[1], "cache_dtype": key[2],
            "best_splits": winner["splits"],
            "best_event_speedup": winner["event_speedup"],
            "best_wall_speedup": winner["wall_speedup"],
            "best_split_event_ms_p50": winner["split_event_ms_p50"],
            "current_fused_event_ms_p50": winner["fused_event_ms_p50"],
            "partial_bytes": winner["partial_bytes"],
            "operator_gate_passed": winner["operator_gate_passed"],
        })
    return candidates, winners


def chart(candidates: list[dict], winners: list[dict]) -> str:
    width = 1280
    left = 235
    plot_width = 900
    row_height = 64
    height = 130 + row_height * len(winners)
    split_values = sorted({row["splits"] for row in candidates})
    maximum = max(1.1, max(row["event_speedup"] for row in candidates) * 1.08)
    minimum = min(0.8, min(row["event_speedup"] for row in candidates) * 0.95)
    x_step = plot_width / max(1, len(split_values) - 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:700}'
        '.muted{fill:#94a3b8}</style>',
        '<text x="28" y="34" class="title">Split-sequence cached Attention search</text>',
        '<text x="28" y="58" class="small muted">MI300X Event median · above 1.05x passes the operator gate</text>',
    ]
    for index, winner in enumerate(winners):
        y_top = 91 + index * row_height
        label = f'B{winner["batch"]} T{winner["sequence"]} {winner["cache_dtype"].upper()}'
        rows = [row for row in candidates
                if (row["batch"], row["sequence"], row["cache_dtype"]) ==
                (winner["batch"], winner["sequence"], winner["cache_dtype"])]
        rows.sort(key=lambda row: row["splits"])
        def y(value: float) -> float:
            return y_top + 42 - (value - minimum) / (maximum - minimum) * 38
        gate_y = y(1.05)
        parts.append(f'<text x="28" y="{y_top + 24}" class="label">{label}</text>')
        parts.append(f'<line x1="{left}" y1="{gate_y:.2f}" x2="{left + plot_width}" '
                     f'y2="{gate_y:.2f}" stroke="#22c55e" stroke-dasharray="5 5"/>')
        points = []
        for split_index, row in enumerate(rows):
            x = left + split_index * x_step
            py = y(row["event_speedup"])
            points.append(f'{x:.2f},{py:.2f}')
            color = "#22c55e" if row["operator_gate_passed"] else "#ef4444"
            parts.append(f'<circle cx="{x:.2f}" cy="{py:.2f}" r="5" fill="{color}"/>')
            parts.append(f'<text x="{x:.2f}" y="{y_top + 57}" class="small" text-anchor="middle">S{row["splits"]}</text>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#60a5fa" stroke-width="2"/>')
        parts.append(f'<text x="{left + plot_width + 18}" y="{y_top + 23}" class="small">'
                     f'best S{winner["best_splits"]} · {winner["best_event_speedup"]:.2f}x</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    cell = 0
    for sequence in args.sequences:
        for batch in args.batches:
            for dtype in args.cache_dtypes:
                for splits in args.splits:
                    for run in range(1, args.runs + 1):
                        order = "forward" if (cell + run) % 2 else "reverse"
                        record = run_one(
                            args, batch, sequence, dtype, splits, order)
                        record["process_run"] = run
                        rows.append(record)
                        print(json.dumps({
                            "batch": batch, "sequence": sequence,
                            "cache_dtype": dtype, "splits": splits,
                            "process_run": run, "order": order,
                            "event_speedup": record["split_speedup_over_fused"],
                        }, sort_keys=True), flush=True)
                    cell += 1
    expected_rows = (len(args.sequences) * len(args.batches) *
                     len(args.cache_dtypes) * len(args.splits) * args.runs)
    if len(rows) != expected_rows:
        raise RuntimeError("split matrix did not produce every process row")
    candidates, winners = aggregate(rows)
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "cached_attention_split_matrix",
        "matrix_complete": True,
        "process_rows": len(rows), "candidate_rows": len(candidates),
        "case_count": len(winners), "runs_per_candidate": args.runs,
        "warmup": args.warmup, "repetitions": args.repetitions,
        "heads": args.heads, "kv_heads": args.kv_heads, "width": args.width,
        "device_name": rows[0]["device_name"],
        "architecture": rows[0]["architecture"],
        "complete_output_accuracy_passed": True,
        "zero_payload_transfers": True,
        "zero_warm_backend_allocations": True,
        "all_case_winners_pass_operator_gate": all(
            winner["operator_gate_passed"] for winner in winners),
        "candidates": candidates,
        "winners": winners,
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "split-search.svg").write_text(
        chart(candidates, winners), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"cached_attention_split_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
