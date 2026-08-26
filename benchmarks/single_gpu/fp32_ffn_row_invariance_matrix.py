#!/usr/bin/env python3
"""Screen real DeepSeek FP32 FFN gate/up solutions across prefill batch M."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "fp32_ffn_row_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

ROWS = [2048, 4096, 8192, 16384]
INNER = 1536
COLUMNS = 8960


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if (not args.binary.is_file() or args.warmup < 0 or
            args.repetitions <= 0 or args.timeout_seconds <= 0):
        parser.error("FP32 FFN row-invariance inputs are invalid")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def run(args: argparse.Namespace) -> dict:
    completed = subprocess.run([
        str(args.binary), "--block-rows", "2048",
        "--multipliers", "1,2,4,8", "--inner", str(INNER),
        "--columns", str(COLUMNS), "--maximum-algorithms", "64",
        "--workspace-bytes", "33554432", "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ], text=True, capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    result = COMMON.last_json(completed.stdout)
    candidates = result.get("candidates", [])
    if (result.get("status") != "pass" or
            result.get("record_type") != "fp32_forward_row_invariance" or
            result.get("block_rows") != 2048 or
            result.get("multipliers") != [1, 2, 4, 8] or
            result.get("inner") != INNER or result.get("columns") != COLUMNS or
            result.get("requested_algorithms") != 64 or
            result.get("shape_candidate_counts") != [64, 64, 64, 64] or
            result.get("common_candidate_count") != len(candidates) or
            len(result.get("default_event_ms_p50", [])) != 4 or
            any(len(row.get("event_ms_p50", [])) != 4 or
                len(row.get("speedup_vs_default", [])) != 4
                for row in candidates)):
        raise ValueError("FP32 FFN row-invariance output changed")
    return result


def summarize(result: dict) -> dict:
    candidates = []
    for source in result["candidates"]:
        row = dict(source)
        row["event_ms_sum"] = sum(float(value)
                                  for value in row["event_ms_p50"])
        row["minimum_speedup"] = min(float(value)
                                     for value in row["speedup_vs_default"])
        row["geometric_mean_speedup"] = math.prod(
            float(value) for value in row["speedup_vs_default"]) ** 0.25
        candidates.append(row)
    invariant = [row for row in candidates if row["block_invariant"]]
    admitted = [row for row in invariant if row["minimum_speedup"] >= 0.95]
    fastest = min(admitted, key=lambda row: (
        row["event_ms_sum"], row["maximum_workspace_bytes"], row["index"])) \
        if admitted else None
    return {
        "schema_version": 1,
        "record_type": "fp32_ffn_row_invariance_matrix",
        "status": "pass", "rows": ROWS, "inner": INNER,
        "columns": COLUMNS, "workspace_limit_bytes": 33554432,
        "default_event_ms_p50": result["default_event_ms_p50"],
        "common_candidate_count": result["common_candidate_count"],
        "supported_count": result["supported_count"],
        "sentinel_pass_count": result["sentinel_pass_count"],
        "block_invariant_count": len(invariant),
        "block_invariant_indices": [row["index"] for row in invariant],
        "performance_admitted_count": len(admitted),
        "performance_admitted_indices": [row["index"] for row in admitted],
        "recommended_index": fastest["index"] if fastest else -1,
        "recommended_minimum_speedup": (
            fastest["minimum_speedup"] if fastest else 0.0),
        "recommended_geometric_mean_speedup": (
            fastest["geometric_mean_speedup"] if fastest else 0.0),
        "candidates": candidates,
    }


def render(summary: dict) -> str:
    width, height = 1510, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:11px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">FP32 FFN gate/up row invariance</text>',
        '<text x="30" y="62" class="muted">M2048/4096/8192/16384 · K1536 · N8960 · exact before timing</text>',
    ]
    for position, row in enumerate(summary["candidates"]):
        column, line = position % 16, position // 16
        x, y = 95 + column * 87, 105 + line * 98
        color = ("#166534" if row["block_invariant"] and
                 row["minimum_speedup"] >= 0.95 else
                 "#92400e" if row["block_invariant"] else "#7f1d1d")
        parts.extend((
            f'<rect x="{x}" y="{y}" width="72" height="78" rx="6" fill="{color}"/>',
            f'<text x="{x + 36}" y="{y + 24}" class="label" text-anchor="middle">{row["index"]}</text>',
            f'<text x="{x + 36}" y="{y + 45}" class="label" text-anchor="middle">'
            f'{"exact" if row["block_invariant"] else "drift"}</text>',
            f'<text x="{x + 36}" y="{y + 65}" class="label" text-anchor="middle">'
            f'{row["minimum_speedup"]:.3f}x</text>',
        ))
    parts.extend((
        f'<text x="40" y="520" class="muted">exact {summary["block_invariant_count"]} · '
        f'performance-admitted {summary["performance_admitted_count"]} · '
        f'recommended {summary["recommended_index"]}</text>',
        '</svg>',
    ))
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    inventory = run(args)
    summary = summarize(inventory)
    (args.output_directory / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row in summary["candidates"]), encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "ffn-row-invariance.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "candidates"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_ffn_row_invariance_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
