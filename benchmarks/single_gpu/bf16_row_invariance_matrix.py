#!/usr/bin/env python3
"""Screen common BF16 solutions for identical-row invariance across M1/2/4/8."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "bf16_row_invariance_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-binary", type=Path, required=True)
    parser.add_argument("--row-invariance-binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.inventory_binary.is_file() or
            not args.row_invariance_binary.is_file() or args.warmup < 0 or
            args.repetitions <= 0 or args.timeout_seconds <= 0):
        parser.error("BF16 row-invariance matrix inputs are invalid")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def run_json(command: list[str], timeout: int) -> dict:
    completed = subprocess.run(
        command, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return COMMON.last_json(completed.stdout)


def collect(args: argparse.Namespace) -> tuple[dict, dict]:
    inventory = run_json([
        str(args.inventory_binary), "--rows", "1,2,4,8",
        "--inner", "1536", "--columns", "8960",
        "--max-algorithms", "64", "--workspace-bytes", "33554432",
    ], args.timeout_seconds)
    candidates = [int(value) for value in inventory.get("common_indices", [])]
    if (inventory.get("status") != "pass" or len(candidates) != 64 or
            any(row.get("candidate_count") != 64
                for row in inventory.get("shapes", []))):
        raise ValueError("BF16 inventory did not expose 64 common candidates")
    matrix = run_json([
        str(args.row_invariance_binary), "--rows", "1,2,4,8",
        "--inner", "1536", "--columns", "8960",
        "--candidates", ",".join(str(value) for value in candidates),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ], args.timeout_seconds)
    if (matrix.get("status") != "pass" or
            matrix.get("candidate_count") != 64 or
            [row.get("index") for row in matrix.get("candidates", [])] !=
                candidates):
        raise ValueError("BF16 row-invariance result changed candidate identity")
    return inventory, matrix


def summarize(inventory: dict, matrix: dict) -> dict:
    workspace = {}
    for shape in inventory["shapes"]:
        for candidate in shape["candidates"]:
            index = int(candidate["index"])
            workspace[index] = max(
                workspace.get(index, 0), int(candidate["workspace_bytes"]))
    rows = []
    for candidate in matrix["candidates"]:
        row = dict(candidate)
        row["maximum_workspace_bytes"] = workspace[int(row["index"])]
        rows.append(row)
    invariant = [row for row in rows if row["row_invariant"]]
    return {
        "schema_version": 1,
        "record_type": "bf16_operator_row_invariance_matrix",
        "status": "pass",
        "rows": [1, 2, 4, 8],
        "inner": 1536,
        "columns": 8960,
        "output_dtype": "bfloat16",
        "candidate_count": len(rows),
        "supported_count": sum(row["supported"] for row in rows),
        "reference_pass_count": sum(row["reference_passed"] for row in rows),
        "row_invariant_count": len(invariant),
        "row_invariant_indices": [row["index"] for row in invariant],
        "minimum_invariant_workspace_bytes": min(
            (row["maximum_workspace_bytes"] for row in invariant), default=-1),
        "maximum_row_error": max(row["row_maximum_error"] for row in rows),
        "maximum_reference_error": max(
            row["reference_maximum_error"] for row in rows),
        "candidates": rows,
    }


def render(summary: dict) -> str:
    width, height = 1500, 680
    columns = 16
    cell_width, cell_height = 86, 110
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:12px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">BF16 gate/up row invariance · M1/2/4/8</text>',
        '<text x="30" y="62" class="muted">K1536 · N8960 · complete BF16 row · '
        'green means CPU gate + bitwise row invariance</text>',
    ]
    for index, row in enumerate(summary["candidates"]):
        column = index % columns
        line = index // columns
        x = 30 + column * cell_width
        y = 95 + line * cell_height
        color = "#166534" if row["row_invariant"] else (
            "#92400e" if row["reference_passed"] else "#7f1d1d")
        parts.extend((
            f'<rect x="{x}" y="{y}" width="72" height="82" rx="7" fill="{color}"/>',
            f'<text x="{x + 36}" y="{y + 26}" class="label" '
            f'text-anchor="middle">{row["index"]}</text>',
            f'<text x="{x + 36}" y="{y + 48}" class="label" '
            f'text-anchor="middle">{"exact" if row["row_invariant"] else "drift"}</text>',
            f'<text x="{x + 36}" y="{y + 68}" class="muted" '
            f'text-anchor="middle">{row["maximum_workspace_bytes"] // 1024} KiB</text>',
        ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    inventory, matrix = collect(args)
    summary = summarize(inventory, matrix)
    (args.output_directory / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row in summary["candidates"]), encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "row-invariance.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "candidates"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"bf16_row_invariance_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
