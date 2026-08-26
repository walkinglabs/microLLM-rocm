#!/usr/bin/env python3
"""Screen FP32 Q and K/V forward solutions for repeated-block invariance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "fp32_qkv_row_invariance_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

OPERATIONS = {"q": 1536, "kv": 256}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.binary.is_file() or args.warmup < 0 or
            args.repetitions <= 0 or args.timeout_seconds <= 0):
        parser.error("FP32 QKV row-invariance inputs are invalid")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def run(args: argparse.Namespace, operation: str, columns: int) -> dict:
    completed = subprocess.run([
        str(args.binary), "--block-rows", "2048",
        "--multipliers", "1,2,4,8", "--inner", "1536",
        "--columns", str(columns), "--maximum-algorithms", "64",
        "--workspace-bytes", "33554432", "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ], text=True, capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    result = COMMON.last_json(completed.stdout)
    if (result.get("status") != "pass" or
            result.get("record_type") != "fp32_forward_row_invariance" or
            result.get("block_rows") != 2048 or
            result.get("multipliers") != [1, 2, 4, 8] or
            result.get("inner") != 1536 or result.get("columns") != columns or
            result.get("requested_algorithms") != 64 or
            result.get("shape_candidate_counts") != [64, 64, 64, 64] or
            result.get("common_candidate_count") != len(result.get("candidates", []))):
        raise ValueError(f"FP32 {operation} row-invariance output changed")
    return result


def summarize(results: dict[str, dict]) -> dict:
    operations = []
    candidates = []
    for operation, result in results.items():
        rows = []
        for candidate in result["candidates"]:
            row = dict(candidate)
            row["operation"] = operation
            row["event_ms_sum"] = sum(float(value)
                                      for value in row["event_ms_p50"])
            rows.append(row)
            candidates.append(row)
        invariant = [row for row in rows if row["block_invariant"]]
        if not invariant:
            fastest = None
        else:
            fastest = min(invariant, key=lambda row: (
                row["event_ms_sum"], row["maximum_workspace_bytes"],
                row["index"]))
        operations.append({
            "operation": operation,
            "columns": OPERATIONS[operation],
            "common_candidate_count": result["common_candidate_count"],
            "supported_count": result["supported_count"],
            "sentinel_pass_count": result["sentinel_pass_count"],
            "block_invariant_count": result["block_invariant_count"],
            "block_invariant_indices": [row["index"] for row in invariant],
            "fastest_invariant_index": fastest["index"] if fastest else -1,
            "fastest_invariant_event_ms_sum":
                fastest["event_ms_sum"] if fastest else 0.0,
            "fastest_invariant_workspace_bytes":
                fastest["maximum_workspace_bytes"] if fastest else -1,
            "maximum_block_error": max(
                float(row["block_maximum_error"]) for row in rows),
            "maximum_sentinel_error": max(
                float(row["sentinel_maximum_error"]) for row in rows),
        })
    return {
        "schema_version": 1,
        "record_type": "fp32_qkv_row_invariance_matrix",
        "status": "pass", "block_rows": 2048,
        "multipliers": [1, 2, 4, 8], "inner": 1536,
        "workspace_limit_bytes": 33554432,
        "operations": operations, "candidates": candidates,
    }


def render(summary: dict) -> str:
    width, height = 1500, 610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:12px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">FP32 Q/KV repeated-block invariance</text>',
        '<text x="30" y="62" class="muted">M2048/4096/8192/16384 · K1536 · '
        'green means CPU sentinel + bitwise 2048-row block</text>',
    ]
    for panel, operation in enumerate(("q", "kv")):
        rows = [row for row in summary["candidates"]
                if row["operation"] == operation]
        y0 = 100 + panel * 245
        parts.append(f'<text x="30" y="{y0}" class="label">'
                     f'{operation.upper()} · N{OPERATIONS[operation]}</text>')
        for index, row in enumerate(rows):
            column = index % 16
            line = index // 16
            x = 160 + column * 80
            y = y0 - 24 + line * 98
            color = "#166534" if row["block_invariant"] else (
                "#92400e" if row["sentinel_passed"] else "#7f1d1d")
            parts.extend((
                f'<rect x="{x}" y="{y}" width="66" height="72" rx="6" '
                f'fill="{color}"/>',
                f'<text x="{x + 33}" y="{y + 25}" class="label" '
                f'text-anchor="middle">{row["index"]}</text>',
                f'<text x="{x + 33}" y="{y + 48}" class="label" '
                f'text-anchor="middle">{"exact" if row["block_invariant"] else "drift"}</text>',
            ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    results = {operation: run(args, operation, columns)
               for operation, columns in OPERATIONS.items()}
    summary = summarize(results)
    for operation, result in results.items():
        (args.output_directory / f"{operation}-inventory.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row in summary["candidates"]), encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "qkv-row-invariance.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "candidates"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_qkv_row_invariance_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
