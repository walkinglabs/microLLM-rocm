#!/usr/bin/env python3
"""Search exact-order cached-Attention finalize physical-thread mappings."""

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
    parser.add_argument("--finalize-threads", type=csv_ints,
                        default=[64, 128, 256])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if not args.benchmark.is_file() or args.runs <= 0 or args.warmup < 3 or \
            args.repetitions <= 0 or max(args.sequences) > 4096:
        parser.error("benchmark, run count, warmup, or sequence is invalid")
    if any(model not in MODELS for model in args.models):
        parser.error("models must use the fixed Qwen/DeepSeek head signatures")
    if set(args.cache_dtypes) - {"fp32", "bf16"}:
        parser.error("cache dtypes must be fp32 or bf16")
    if set(args.finalize_threads) != {64, 128, 256}:
        parser.error("finalize thread search must contain exactly 64,128,256")
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
    raise ValueError("benchmark emitted no JSON object")


def number(row: dict, name: str, *, positive: bool = False) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is missing or not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} is outside its numeric contract")
    return result


def run_one(args: argparse.Namespace, model: str, sequence: int, batch: int,
            dtype: str, threads: int, order: str) -> dict:
    heads, kv_heads, width = MODELS[model]
    completed = subprocess.run([
        str(args.benchmark), "--batch", str(batch), "--heads", str(heads),
        "--kv-heads", str(kv_heads), "--sequence", str(sequence),
        "--width", str(width), "--cache-dtype", dtype,
        "--materialized", "true", "--finalize-threads", str(threads),
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
        "materialized_finalize_threads": threads,
        "materialized_bitwise_equal_current": True,
        "complete_output_accuracy_passed": True,
        "host_to_device_calls": 0, "device_to_host_calls": 0,
    }
    for name, wanted in expected.items():
        if row.get(name) != wanted:
            raise ValueError(f"{name} expected {wanted!r}, got {row.get(name)!r}")
    for name in ("materialized_event_ms_p50", "materialized_event_ms_p95",
                 "materialized_wall_ms_p50", "materialized_wall_ms_p95"):
        number(row, name, positive=True)
    if number(row, "materialized_backend_allocation_calls_per_invocation") != 0:
        raise ValueError("warm mapped finalize reached the backend allocator")
    if row.get("materialized_allocation_calls_per_invocation") != 2:
        raise ValueError("mapped route no longer makes two logical allocations")
    row["model"] = model
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["sequence"], row["batch"],
                row["cache_dtype"], row["materialized_finalize_threads"])].append(row)
    cases = []
    roots = sorted({key[:4] for key in groups},
                   key=lambda key: (key[1], key[0], key[2], key[3]))
    for root in roots:
        policies = {}
        for threads in (64, 128, 256):
            samples = groups[root + (threads,)]
            policies[threads] = {
                "threads": threads,
                "runs": len(samples),
                "event_ms_p50": statistics.median(
                    float(row["materialized_event_ms_p50"]) for row in samples),
                "wall_ms_p50": statistics.median(
                    float(row["materialized_wall_ms_p50"]) for row in samples),
                "bitwise_equal_current": all(
                    row["materialized_bitwise_equal_current"] for row in samples),
                "zero_backend_allocations": all(
                    row["materialized_backend_allocation_calls_per_invocation"] == 0
                    for row in samples),
            }
        baseline = policies[256]
        candidates = [policies[64], policies[128]]
        winner = min(candidates, key=lambda row: row["event_ms_p50"])
        event_speedup = baseline["event_ms_p50"] / winner["event_ms_p50"]
        wall_speedup = baseline["wall_ms_p50"] / winner["wall_ms_p50"]
        cases.append({
            "model": root[0], "sequence": root[1], "batch": root[2],
            "cache_dtype": root[3], "policies": [policies[key]
                                                    for key in (64, 128, 256)],
            "winner_threads": winner["threads"],
            "winner_event_speedup_vs_256": event_speedup,
            "winner_wall_speedup_vs_256": wall_speedup,
            "accuracy_gate_passed": all(
                policy["bitwise_equal_current"] for policy in policies.values()),
            "performance_gate_passed": event_speedup >= 1.05 and wall_speedup >= 1.02,
        })
    return cases


def render(cases: list[dict]) -> str:
    width, height = 1280, 112 + len(cases) * 46
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:21px;font-weight:700}.label{font-size:13px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="35" class="title">Exact-order finalize mapping search</text>',
        '<text x="30" y="59" class="muted">winner versus 256 physical threads · '
        'green requires Event >=1.05x and wall >=1.02x</text>',
        '<line x1="900" y1="78" x2="900" y2="100%" stroke="#64748b"/>',
    ]
    for index, case in enumerate(cases):
        y = 84 + index * 46
        short = "Qwen" if case["model"].startswith("qwen") else "DeepSeek"
        label = (f'{short} T{case["sequence"]} B{case["batch"]} '
                 f'{case["cache_dtype"].upper()}')
        speedup = case["winner_event_speedup_vs_256"]
        bar = max(2.0, min(300.0, (speedup - 0.8) * 1000.0))
        color = "#22c55e" if case["performance_gate_passed"] else "#ef4444"
        parts.extend((
            f'<text x="30" y="{y + 18}" class="label">{label}</text>',
            f'<rect x="700" y="{y}" width="{bar:.2f}" height="24" rx="4" '
            f'fill="{color}"/>',
            f'<text x="1020" y="{y + 18}" class="label">'
            f'{speedup:.3f}x · {case["winner_threads"]} threads</text>',
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
                    for threads in args.finalize_threads:
                        for process_run in range(1, args.runs + 1):
                            order = "forward" if (ordinal + process_run) % 2 else "reverse"
                            row = run_one(args, model, sequence, batch, dtype,
                                          threads, order)
                            row["process_run"] = process_run
                            rows.append(row)
                            print(json.dumps({
                                "model": model, "sequence": sequence,
                                "batch": batch, "cache_dtype": dtype,
                                "threads": threads, "process_run": process_run,
                                "event_ms": row["materialized_event_ms_p50"],
                            }, sort_keys=True), flush=True)
                        ordinal += 1
    cases = aggregate(rows)
    expected_cases = (len(args.models) * len(args.sequences) *
                      len(args.batches) * len(args.cache_dtypes))
    expected_rows = expected_cases * 3 * args.runs
    if len(cases) != expected_cases or len(rows) != expected_rows:
        raise RuntimeError("finalize mapping matrix is incomplete")
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "cached_attention_finalize_mapping_matrix",
        "matrix_complete": True, "process_rows": len(rows),
        "case_count": len(cases), "runs_per_policy": args.runs,
        "warmup": args.warmup, "repetitions": args.repetitions,
        "models": args.models, "sequences": args.sequences,
        "batches": args.batches, "cache_dtypes": args.cache_dtypes,
        "finalize_threads": [64, 128, 256],
        "device_name": rows[0]["device_name"],
        "architecture": rows[0]["architecture"],
        "all_accuracy_gates_passed": all(
            case["accuracy_gate_passed"] for case in cases),
        "performance_pass_count": sum(
            case["performance_gate_passed"] for case in cases),
        "minimum_winner_event_speedup": min(
            case["winner_event_speedup_vs_256"] for case in cases),
        "maximum_winner_event_speedup": max(
            case["winner_event_speedup_vs_256"] for case in cases),
        "minimum_winner_wall_speedup": min(
            case["winner_wall_speedup_vs_256"] for case in cases),
        "maximum_winner_wall_speedup": max(
            case["winner_wall_speedup_vs_256"] for case in cases),
        "cases": cases,
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "mapping.svg").write_text(
        render(cases), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"cached_attention_finalize_mapping_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
