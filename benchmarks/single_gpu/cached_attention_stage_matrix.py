#!/usr/bin/env python3
"""Run and summarize the complete cached-decode Attention stage matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


STAGES = ("score", "softmax", "context", "pipeline", "fused")
TIMING_SUFFIXES = (
    "event_ms_p50", "event_ms_p95", "wall_ms_p50", "wall_ms_p95",
    "allocation_calls_per_invocation",
    "backend_allocation_calls_per_invocation",
    "cache_reuse_calls_per_invocation",
)
ERROR_FIELDS = (
    "score_max_error", "score_rms_error",
    "probability_max_error", "probability_rms_error",
    "context_max_error", "context_rms_error",
    "pipeline_max_error", "pipeline_rms_error",
    "fused_max_error", "fused_rms_error",
)


def csv_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("matrix dimensions must be positive")
    return values


def csv_strings(value: str) -> list[str]:
    values = [item for item in value.split(",") if item]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sequences", type=csv_ints, default=[512, 2048])
    parser.add_argument("--batches", type=csv_ints, default=[1, 2])
    parser.add_argument(
        "--cache-dtypes", type=csv_strings, default=["fp32", "bf16"])
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


def finite_number(record: dict, field: str, *, positive: bool = False) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is missing or non-numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or (positive and numeric <= 0):
        raise ValueError(f"{field} is outside its numeric contract")
    return numeric


def validate(record: dict, *, batch: int, sequence: int, dtype: str,
             order: str, warmup: int, repetitions: int,
             heads: int, kv_heads: int, width: int) -> None:
    expected = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "cached_attention_stage_probe",
        "batch": batch,
        "heads": heads,
        "kv_heads": kv_heads,
        "sequence": sequence,
        "width": width,
        "repeats": heads // kv_heads,
        "cache_dtype": dtype,
        "order": order,
        "warmup": warmup,
        "repetitions": repetitions,
        "score_elements": batch * heads * sequence,
        "context_elements": batch * heads * width,
        "global_score_bytes": batch * heads * sequence * 4,
        "complete_output_accuracy_passed": True,
        "host_to_device_calls": 0,
        "device_to_host_calls": 0,
    }
    for field, wanted in expected.items():
        if record.get(field) != wanted:
            raise ValueError(f"{field} expected {wanted!r}, got {record.get(field)!r}")
    if not record.get("device_name") or not record.get("architecture"):
        raise ValueError("device identity is missing")
    for field in ERROR_FIELDS:
        finite_number(record, field)
    for stage in STAGES:
        for suffix in TIMING_SUFFIXES:
            finite_number(
                record, f"{stage}_{suffix}",
                positive=suffix.startswith(("event_", "wall_")))
        if finite_number(
                record,
                f"{stage}_backend_allocation_calls_per_invocation") != 0:
            raise ValueError(f"{stage} reached the backend allocator after warm-up")
    if finite_number(record, "score_max_error") > 2.0e-4:
        raise ValueError("score maximum error exceeds the gate")
    if finite_number(record, "probability_max_error") > 3.0e-4:
        raise ValueError("probability maximum error exceeds the gate")
    for field in ("context_max_error", "pipeline_max_error", "fused_max_error"):
        if finite_number(record, field) > 8.0e-4:
            raise ValueError(f"{field} exceeds the gate")
    for field in (
            "stage_sum_event_ms_p50", "stage_sum_over_pipeline",
            "fused_speedup_over_pipeline"):
        finite_number(record, field, positive=True)


def run_one(args: argparse.Namespace, batch: int, sequence: int,
            dtype: str, order: str) -> dict:
    command = [
        str(args.benchmark),
        "--batch", str(batch), "--heads", str(args.heads),
        "--kv-heads", str(args.kv_heads), "--sequence", str(sequence),
        "--width", str(args.width), "--cache-dtype", dtype,
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions), "--order", order,
    ]
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, timeout=300)
    record = last_json(completed.stdout)
    validate(
        record, batch=batch, sequence=sequence, dtype=dtype, order=order,
        warmup=args.warmup, repetitions=args.repetitions,
        heads=args.heads, kv_heads=args.kv_heads, width=args.width)
    return record


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["batch"], row["sequence"], row["cache_dtype"])].append(row)
    cases = []
    for key in sorted(groups, key=lambda item: (item[1], item[0], item[2])):
        samples = groups[key]
        case = {
            "batch": key[0], "sequence": key[1], "cache_dtype": key[2],
            "runs": len(samples), "orders": sorted({row["order"] for row in samples}),
            "score_elements": samples[0]["score_elements"],
            "context_elements": samples[0]["context_elements"],
            "global_score_bytes": samples[0]["global_score_bytes"],
        }
        for stage in STAGES:
            for suffix in ("event_ms_p50", "event_ms_p95", "wall_ms_p50"):
                field = f"{stage}_{suffix}"
                case[field] = statistics.median(float(row[field]) for row in samples)
        for field in ERROR_FIELDS:
            case[field] = max(float(row[field]) for row in samples)
        stage_sum = sum(case[f"{stage}_event_ms_p50"]
                        for stage in ("score", "softmax", "context"))
        case["stage_sum_event_ms_p50"] = stage_sum
        case["score_share"] = case["score_event_ms_p50"] / stage_sum
        case["softmax_share"] = case["softmax_event_ms_p50"] / stage_sum
        case["context_share"] = case["context_event_ms_p50"] / stage_sum
        case["fused_speedup_over_pipeline"] = (
            case["pipeline_event_ms_p50"] / case["fused_event_ms_p50"])
        case["pipeline_event_range_ms"] = [
            min(float(row["pipeline_event_ms_p50"]) for row in samples),
            max(float(row["pipeline_event_ms_p50"]) for row in samples),
        ]
        case["fused_event_range_ms"] = [
            min(float(row["fused_event_ms_p50"]) for row in samples),
            max(float(row["fused_event_ms_p50"]) for row in samples),
        ]
        cases.append(case)
    return cases


def svg_chart(cases: list[dict]) -> str:
    width = 1240
    left = 220
    plot_width = 860
    row_height = 68
    height = 125 + row_height * len(cases)
    maximum = max(max(case["stage_sum_event_ms_p50"],
                      case["pipeline_event_ms_p50"],
                      case["fused_event_ms_p50"]) for case in cases)
    scale = plot_width / maximum
    colors = {"score": "#ef4444", "softmax": "#f59e0b", "context": "#3b82f6"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:700}'
        '.muted{fill:#94a3b8}</style>',
        '<text x="28" y="34" class="title">Cached Attention: transparent stages vs fused</text>',
        '<text x="28" y="58" class="small muted">MI300X HIP Event median; longer bars are slower</text>',
    ]
    legend_x = 590
    for index, (label, color) in enumerate(
            (("QK score", colors["score"]), ("softmax", colors["softmax"]),
             ("PV context", colors["context"]), ("fused", "#22c55e"))):
        x = legend_x + index * 145
        parts.append(f'<rect x="{x}" y="25" width="14" height="14" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 21}" y="37" class="small">{label}</text>')
    for index, case in enumerate(cases):
        y = 94 + index * row_height
        label = f'B{case["batch"]} T{case["sequence"]} {case["cache_dtype"].upper()}'
        parts.append(f'<text x="28" y="{y + 17}" class="label">{label}</text>')
        x = left
        for stage in ("score", "softmax", "context"):
            value = case[f"{stage}_event_ms_p50"]
            bar = max(1.0, value * scale)
            parts.append(f'<rect x="{x:.2f}" y="{y}" width="{bar:.2f}" height="18" '
                         f'rx="2" fill="{colors[stage]}"/>')
            x += bar
        pipeline = case["pipeline_event_ms_p50"]
        fused = case["fused_event_ms_p50"]
        parts.append(f'<rect x="{left}" y="{y + 25}" width="{max(1.0, fused * scale):.2f}" '
                     'height="15" rx="2" fill="#22c55e"/>')
        parts.append(f'<text x="{left + pipeline * scale + 8:.2f}" y="{y + 15}" class="small">'
                     f'pipeline {pipeline:.4f} ms</text>')
        parts.append(f'<text x="{left + fused * scale + 8:.2f}" y="{y + 38}" class="small">'
                     f'fused {fused:.4f} ms · {case["fused_speedup_over_pipeline"]:.2f}x</text>')
        parts.append(f'<line x1="{left}" y1="{y + 50}" x2="{left + plot_width}" y2="{y + 50}" '
                     'stroke="#1e293b"/>')
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
                        "batch": batch, "sequence": sequence,
                        "cache_dtype": dtype, "process_run": run,
                        "order": order,
                        "pipeline_event_ms_p50": record["pipeline_event_ms_p50"],
                        "fused_event_ms_p50": record["fused_event_ms_p50"],
                    }, sort_keys=True), flush=True)
                case_index += 1
    expected_rows = (len(args.sequences) * len(args.batches) *
                     len(args.cache_dtypes) * args.runs)
    if len(rows) != expected_rows:
        raise RuntimeError("matrix did not produce every requested process row")
    cases = aggregate(rows)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "cached_attention_stage_matrix",
        "matrix_complete": True,
        "process_rows": len(rows),
        "case_count": len(cases),
        "runs_per_case": args.runs,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "width": args.width,
        "device_name": rows[0]["device_name"],
        "architecture": rows[0]["architecture"],
        "complete_output_accuracy_passed": True,
        "zero_payload_transfers": True,
        "zero_warm_backend_allocations": True,
        "cases": cases,
    }
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "stage-timing.svg").write_text(
        svg_chart(cases), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"cached_attention_stage_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
