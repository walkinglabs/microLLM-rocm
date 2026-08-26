#!/usr/bin/env python3
"""Run correctness-first QK and P*V solution matrices across request batch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "fp32_attention_batch_invariance_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

OPERATIONS = {
    "qk": {"m": 2048, "k": 128, "n": 2048, "transpose_right": True},
    "pv": {"m": 2048, "k": 2048, "n": 128, "transpose_right": False},
}
BATCHES = [1, 2, 4, 8]


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
        parser.error("FP32 Attention batch matrix inputs are invalid")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, operation: str,
            inventory_only: bool) -> list[str]:
    return [
        str(args.binary), "--operation", operation,
        "--sequence", "2048", "--heads", "12", "--kv-heads", "2",
        "--width", "128", "--request-batches", "1,2,4,8",
        "--maximum-algorithms", "64", "--workspace-bytes", "33554432",
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
        "--inventory-only", "true" if inventory_only else "false",
    ]


def run(args: argparse.Namespace, operation: str,
        inventory_only: bool) -> dict:
    completed = subprocess.run(
        command(args, operation, inventory_only), text=True,
        capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    result = COMMON.last_json(completed.stdout)
    expected = OPERATIONS[operation]
    record_type = ("fp32_attention_batch_inventory" if inventory_only
                   else "fp32_attention_batch_invariance")
    if (result.get("status") != "pass" or
            result.get("record_type") != record_type or
            result.get("operation") != operation or
            result.get("sequence") != 2048 or result.get("heads") != 12 or
            result.get("kv_heads") != 2 or result.get("width") != 128 or
            result.get("request_batches") != BATCHES or
            result.get("shape_candidate_counts") != [64, 64, 64, 64] or
            any(result.get(key) != value for key, value in expected.items())):
        raise ValueError(f"FP32 Attention {operation} output contract changed")
    if inventory_only:
        indices = result.get("common_candidate_indices", [])
        if not indices or indices != sorted(set(indices)):
            raise ValueError(f"FP32 Attention {operation} inventory changed")
    else:
        candidates = result.get("candidates", [])
        if (result.get("common_candidate_count") != len(candidates) or
                [row.get("index") for row in candidates] !=
                sorted(row.get("index") for row in candidates) or
                len(result.get("default_event_ms_p50", [])) != 4 or
                len(result.get("default_wall_ms_p50", [])) != 4):
            raise ValueError(f"FP32 Attention {operation} matrix changed")
    return result


def summarize(results: dict[str, dict], inventories: dict[str, dict]) -> dict:
    operations = []
    candidates = []
    for operation in ("qk", "pv"):
        result = results[operation]
        inventory = inventories[operation]
        inventory_indices = inventory["common_candidate_indices"]
        if [row["index"] for row in result["candidates"]] != inventory_indices:
            raise ValueError(f"{operation} inventory and execution diverged")
        rows = []
        for candidate in result["candidates"]:
            row = dict(candidate)
            row["operation"] = operation
            speedups = [float(value) for value in row["event_speedup_vs_default"]]
            row["minimum_event_speedup"] = min(speedups) if speedups else 0.0
            row["geometric_mean_event_speedup"] = (
                math.prod(speedups) ** (1.0 / len(speedups)) if speedups else 0.0)
            row["non_regressing"] = (
                row["block_invariant"] and row["minimum_event_speedup"] >= 0.95)
            rows.append(row)
            candidates.append(row)
        invariant = [row for row in rows if row["block_invariant"]]
        non_regressing = [row for row in rows if row["non_regressing"]]
        best_exact = max(invariant, key=lambda row: (
            row["geometric_mean_event_speedup"],
            row["minimum_event_speedup"],
            -row["maximum_workspace_bytes"], -row["index"])) if invariant else None
        admitted = max(non_regressing, key=lambda row: (
            row["geometric_mean_event_speedup"],
            row["minimum_event_speedup"],
            -row["maximum_workspace_bytes"], -row["index"])) \
            if non_regressing else None
        operations.append({
            "operation": operation,
            **OPERATIONS[operation],
            "shape_candidate_counts": result["shape_candidate_counts"],
            "common_candidate_count": len(rows),
            "correctness_passed_count": result["correctness_passed_count"],
            "block_invariant_count": result["block_invariant_count"],
            "block_invariant_indices": [row["index"] for row in invariant],
            "non_regressing_invariant_count": len(non_regressing),
            "non_regressing_invariant_indices": [
                row["index"] for row in non_regressing],
            "default_block_invariant": result["default_block_invariant"],
            "default_block_maximum_error": result["default_block_maximum_error"],
            "default_block_rms_error": result["default_block_rms_error"],
            "best_exact_index": best_exact["index"] if best_exact else -1,
            "best_exact_minimum_event_speedup": (
                best_exact["minimum_event_speedup"] if best_exact else 0.0),
            "best_exact_geometric_mean_event_speedup": (
                best_exact["geometric_mean_event_speedup"]
                if best_exact else 0.0),
            "best_exact_workspace_bytes": (
                best_exact["maximum_workspace_bytes"] if best_exact else -1),
            "admitted_index": admitted["index"] if admitted else -1,
        })
    return {
        "schema_version": 1,
        "record_type": "fp32_attention_batch_invariance_matrix",
        "status": "pass",
        "sequence": 2048,
        "heads": 12,
        "kv_heads": 2,
        "width": 128,
        "request_batches": BATCHES,
        "backend_batch_counts": [12, 24, 48, 96],
        "workspace_limit_bytes": 33554432,
        "operations": operations,
        "candidates": candidates,
    }


def render(summary: dict) -> str:
    width, height = 1640, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:24px;font-weight:700}.sub{font-size:12px;fill:#94a3b8}'
        '.label{font-size:11px}</style>',
        '<text x="34" y="42" class="title">FP32 Attention solution invariance by request batch</text>',
        '<text x="34" y="66" class="sub">T2048 · backend batch 12/24/48/96 · '
        'green=exact and min speedup ≥0.95 · orange=exact with regression · red=drift</text>',
    ]
    for panel, operation in enumerate(("qk", "pv")):
        rows = [row for row in summary["candidates"]
                if row["operation"] == operation]
        y0 = 110 + panel * 285
        parts.append(f'<text x="34" y="{y0}" class="label">{operation.upper()}</text>')
        for index, row in enumerate(rows):
            column = index % 17
            line = index // 17
            x = 125 + column * 87
            y = y0 - 24 + line * 105
            color = ("#166534" if row["non_regressing"] else
                     "#92400e" if row["block_invariant"] else "#7f1d1d")
            parts.extend((
                f'<rect x="{x}" y="{y}" width="74" height="80" rx="7" '
                f'fill="{color}"/>',
                f'<text x="{x + 37}" y="{y + 24}" class="label" '
                f'text-anchor="middle">{row["index"]}</text>',
                f'<text x="{x + 37}" y="{y + 46}" class="label" '
                f'text-anchor="middle">{"exact" if row["block_invariant"] else "drift"}</text>',
                f'<text x="{x + 37}" y="{y + 66}" class="label" '
                f'text-anchor="middle">min {row["minimum_event_speedup"]:.3f}x</text>',
            ))
        operation_summary = next(
            row for row in summary["operations"] if row["operation"] == operation)
        parts.append(
            f'<text x="125" y="{y0 + 220}" class="sub">common '
            f'{operation_summary["common_candidate_count"]} · exact '
            f'{operation_summary["block_invariant_count"]} · non-regressing '
            f'{operation_summary["non_regressing_invariant_count"]} · selected '
            f'{operation_summary["admitted_index"]}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    inventories = {operation: run(args, operation, True)
                   for operation in ("qk", "pv")}
    results = {operation: run(args, operation, False)
               for operation in ("qk", "pv")}
    summary = summarize(results, inventories)
    for operation in ("qk", "pv"):
        (args.output_directory / f"{operation}-inventory.json").write_text(
            json.dumps(inventories[operation], indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (args.output_directory / f"{operation}-matrix.json").write_text(
            json.dumps(results[operation], indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row in summary["candidates"]), encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "attention-solutions.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "candidates"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_attention_batch_invariance_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
