#!/usr/bin/env python3
"""Measure exact-order GQA value-load reuse against materialized current."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


MODELS = {
    "qwen2.5-0.5b": (14, 2, 64),
    "deepseek-r1-distill-qwen-1.5b": (12, 2, 128),
}


def csv_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("integer list must be positive")
    return result


def csv_strings(value: str) -> list[str]:
    result = [item for item in value.split(",") if item]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty list")
    return result


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--models", type=csv_strings, default=list(MODELS))
    parser.add_argument("--sequences", type=csv_ints, default=[512, 2048])
    parser.add_argument("--batches", type=csv_ints, default=[1, 2])
    parser.add_argument("--cache-dtypes", type=csv_strings,
                        default=["fp32", "bf16"])
    parser.add_argument("--tile-columns", type=csv_ints,
                        default=[8, 16, 32, 64])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if not args.benchmark.is_file() or args.runs <= 0 or args.warmup < 3 or \
            args.repetitions <= 0 or max(args.sequences) > 4096:
        parser.error("benchmark, run count, warmup, or sequence is invalid")
    if any(model not in MODELS for model in args.models):
        parser.error("models must use fixed Qwen/DeepSeek GQA signatures")
    if set(args.cache_dtypes) - {"fp32", "bf16"}:
        parser.error("cache dtypes must be fp32 or bf16")
    if set(args.tile_columns) != {8, 16, 32, 64}:
        parser.error("tile search must contain exactly 8,16,32,64")
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
    raise ValueError("benchmark emitted no JSON")


def number(row: dict, name: str, *, positive: bool = False) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is missing or non-numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} is outside its numeric contract")
    return result


def run_one(args: argparse.Namespace, model: str, sequence: int, batch: int,
            dtype: str, tile: int, order: str) -> dict:
    heads, kv_heads, width = MODELS[model]
    completed = subprocess.run([
        str(args.benchmark), "--batch", str(batch), "--heads", str(heads),
        "--kv-heads", str(kv_heads), "--sequence", str(sequence),
        "--width", str(width), "--cache-dtype", dtype,
        "--materialized", "true", "--finalize-threads", "256",
        "--gqa-value-reuse", "true", "--gqa-tile-columns", str(tile),
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
        "--order", order,
    ], text=True, capture_output=True, timeout=300, check=True)
    row = last_json(completed.stdout)
    expected = {
        "schema_version": 1, "status": "pass",
        "record_type": "cached_attention_stage_probe", "batch": batch,
        "heads": heads, "kv_heads": kv_heads, "sequence": sequence,
        "width": width, "cache_dtype": dtype, "order": order,
        "warmup": args.warmup, "repetitions": args.repetitions,
        "materialized_finalize_threads": 256,
        "gqa_value_reuse_tile_columns": tile,
        "gqa_value_reuse_bitwise_equal_materialized": True,
        "complete_output_accuracy_passed": True,
        "host_to_device_calls": 0, "device_to_host_calls": 0,
    }
    for name, wanted in expected.items():
        if row.get(name) != wanted:
            raise ValueError(f"{name} expected {wanted!r}, got {row.get(name)!r}")
    for prefix in ("materialized", "gqa_value_reuse"):
        for suffix in ("event_ms_p50", "event_ms_p95",
                       "wall_ms_p50", "wall_ms_p95"):
            number(row, f"{prefix}_{suffix}", positive=True)
        if number(row, f"{prefix}_backend_allocation_calls_per_invocation") != 0:
            raise ValueError(f"{prefix} reached backend allocator after warm-up")
    if row.get("materialized_allocation_calls_per_invocation") != 2 or \
            row.get("gqa_value_reuse_allocation_calls_per_invocation") != 3:
        raise ValueError("GQA value-reuse logical allocation identity changed")
    if (number(row, "gqa_value_reuse_max_error") !=
            number(row, "materialized_max_error") or
            number(row, "gqa_value_reuse_rms_error") !=
            number(row, "materialized_rms_error")):
        raise ValueError("GQA value-reuse exact output evidence changed")
    row["model"] = model
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["sequence"], row["batch"],
                row["cache_dtype"], row["gqa_value_reuse_tile_columns"])].append(row)
    roots = sorted({key[:4] for key in groups},
                   key=lambda key: (key[1], key[0], key[2], key[3]))
    cases = []
    for root in roots:
        candidates = []
        for tile in (8, 16, 32, 64):
            samples = groups[root + (tile,)]
            current_event = statistics.median(
                float(row["materialized_event_ms_p50"]) for row in samples)
            candidate_event = statistics.median(
                float(row["gqa_value_reuse_event_ms_p50"]) for row in samples)
            current_wall = statistics.median(
                float(row["materialized_wall_ms_p50"]) for row in samples)
            candidate_wall = statistics.median(
                float(row["gqa_value_reuse_wall_ms_p50"]) for row in samples)
            candidates.append({
                "tile_columns": tile, "runs": len(samples),
                "current_event_ms_p50": current_event,
                "candidate_event_ms_p50": candidate_event,
                "event_speedup": current_event / candidate_event,
                "current_wall_ms_p50": current_wall,
                "candidate_wall_ms_p50": candidate_wall,
                "wall_speedup": current_wall / candidate_wall,
                "probability_bytes": samples[0]["gqa_value_reuse_probability_bytes"],
                "all_bitwise_materialized": all(
                    row["gqa_value_reuse_bitwise_equal_materialized"]
                    for row in samples),
                "zero_backend_allocations": all(
                    row["gqa_value_reuse_backend_allocation_calls_per_invocation"] == 0
                    for row in samples),
            })
        winner = min(candidates, key=lambda row: row["candidate_event_ms_p50"])
        cases.append({
            "model": root[0], "sequence": root[1], "batch": root[2],
            "cache_dtype": root[3], "candidates": candidates,
            "winner_tile_columns": winner["tile_columns"],
            "winner_event_speedup": winner["event_speedup"],
            "winner_wall_speedup": winner["wall_speedup"],
            "accuracy_gate_passed": all(
                candidate["all_bitwise_materialized"] for candidate in candidates),
            "performance_gate_passed": (
                winner["event_speedup"] >= 1.05 and
                winner["wall_speedup"] >= 1.02),
        })
    return cases


def render(cases: list[dict]) -> str:
    width, height = 1260, 110 + len(cases) * 47
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:21px;font-weight:700}.label{font-size:13px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="35" class="title">Exact-order GQA value-load reuse</text>',
        '<text x="30" y="59" class="muted">best tile versus materialized current · '
        'all contexts must be bitwise equal</text>',
    ]
    for index, case in enumerate(cases):
        y = 82 + index * 47
        short = "Qwen" if case["model"].startswith("qwen") else "DeepSeek"
        label = (f'{short} T{case["sequence"]} B{case["batch"]} '
                 f'{case["cache_dtype"].upper()}')
        speedup = case["winner_event_speedup"]
        bar = max(2.0, min(600.0, speedup * 420.0))
        color = "#22c55e" if case["performance_gate_passed"] else "#ef4444"
        parts.extend((
            f'<text x="30" y="{y + 19}" class="label">{label}</text>',
            f'<rect x="360" y="{y}" width="{bar:.2f}" height="25" rx="4" '
            f'fill="{color}"/>',
            f'<text x="990" y="{y + 19}" class="label">'
            f'{speedup:.3f}x · tile {case["winner_tile_columns"]}</text>',
        ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    ordinal = 0
    for sequence in args.sequences:
        for model in args.models:
            for batch in args.batches:
                for dtype in args.cache_dtypes:
                    for tile in args.tile_columns:
                        for process_run in range(1, args.runs + 1):
                            order = "forward" if (ordinal + process_run) % 2 else "reverse"
                            row = run_one(args, model, sequence, batch, dtype,
                                          tile, order)
                            row["process_run"] = process_run
                            rows.append(row)
                            print(json.dumps({
                                "model": model, "sequence": sequence,
                                "batch": batch, "cache_dtype": dtype,
                                "tile_columns": tile, "process_run": process_run,
                                "speedup": row["gqa_value_reuse_speedup_over_materialized"],
                            }, sort_keys=True), flush=True)
                        ordinal += 1
    cases = aggregate(rows)
    expected_cases = (len(args.models) * len(args.sequences) *
                      len(args.batches) * len(args.cache_dtypes))
    expected_rows = expected_cases * 4 * args.runs
    if len(cases) != expected_cases or len(rows) != expected_rows:
        raise RuntimeError("GQA value-reuse matrix is incomplete")
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "cached_attention_gqa_value_reuse_matrix",
        "matrix_complete": True, "process_rows": len(rows),
        "candidate_rows": expected_cases * 4, "case_count": len(cases),
        "runs_per_candidate": args.runs, "warmup": args.warmup,
        "repetitions": args.repetitions, "models": args.models,
        "sequences": args.sequences, "batches": args.batches,
        "cache_dtypes": args.cache_dtypes, "tile_columns": [8, 16, 32, 64],
        "device_name": rows[0]["device_name"],
        "architecture": rows[0]["architecture"],
        "all_accuracy_gates_passed": all(
            case["accuracy_gate_passed"] for case in cases),
        "zero_payload_transfers": True,
        "zero_warm_backend_allocations": True,
        "performance_pass_count": sum(
            case["performance_gate_passed"] for case in cases),
        "minimum_winner_event_speedup": min(
            case["winner_event_speedup"] for case in cases),
        "maximum_winner_event_speedup": max(
            case["winner_event_speedup"] for case in cases),
        "minimum_winner_wall_speedup": min(
            case["winner_wall_speedup"] for case in cases),
        "maximum_winner_wall_speedup": max(
            case["winner_wall_speedup"] for case in cases),
        "cases": cases,
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "value-reuse.svg").write_text(
        render(cases), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"cached_attention_gqa_value_reuse_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
