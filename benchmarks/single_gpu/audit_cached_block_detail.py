#!/usr/bin/env python3
"""Locate the first internal operation that amplifies cached block-0 batch drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


BLOCK_SPEC = importlib.util.spec_from_file_location(
    "audit_cached_block_detail_block",
    Path(__file__).with_name("audit_cached_block_drift.py"))
BLOCK = importlib.util.module_from_spec(BLOCK_SPEC)
assert BLOCK_SPEC.loader is not None
BLOCK_SPEC.loader.exec_module(BLOCK)

POLICIES = BLOCK.POLICIES
PREFIX = "inference.cached.blocks.0"
MATERIAL_THRESHOLD = 1.0e-3


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--trace-max-elements", type=int, default=400000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context <= 0 or args.runs != 2 or
            args.trace_max_elements < 151936 or args.timeout_seconds <= 0):
        parser.error("cached block-detail inputs are outside the fixed contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, policy: str, batch: int,
            trace: Path, logits: Path) -> list[str]:
    result = BLOCK.command(args, model, policy, batch, trace, logits)
    result.extend([
        "--trace-all-layer-details", "true",
        "--trace-value-filter", PREFIX,
    ])
    return result


def selected(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line]
    rows = [row for row in records
            if row.get("kind") == "layer" and
            (row.get("name") == PREFIX or
             str(row.get("name", "")).startswith(PREFIX + "."))]
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("cached block-detail names are missing or repeated")
    if any(row.get("values_truncated") or not row.get("values") for row in rows):
        raise ValueError("cached block-detail values are missing or truncated")
    return rows


def difference(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("block-detail comparison needs equal non-empty vectors")
    square = 0.0
    reference_square = 0.0
    maximum = 0.0
    maximum_index = 0
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        delta = abs(float(left_value) - float(right_value))
        square += delta * delta
        reference_square += float(left_value) * float(left_value)
        if delta > maximum:
            maximum = delta
            maximum_index = index
    return {
        "elements": len(left),
        "maximum": maximum,
        "maximum_index": maximum_index,
        "rms": math.sqrt(square / len(left)),
        "relative_l2": math.sqrt(square / reference_square)
        if reference_square > 0.0 else 0.0,
        "bitwise_equal": left == right,
    }


def compare(b1: list[dict], b2: list[dict]) -> list[dict]:
    if [row["name"] for row in b1] != [row["name"] for row in b2]:
        raise ValueError("cached B1/B2 block-detail stage order changed")
    result = []
    for left, right in zip(b1, b2):
        left_shape = [int(value) for value in left["shape"]]
        right_shape = [int(value) for value in right["shape"]]
        if (left.get("dtype") != right.get("dtype") or not left_shape or
                right_shape[0] != 2 * left_shape[0] or
                right_shape[1:] != left_shape[1:]):
            raise ValueError(f"cached block-detail shape changed: {left['name']}")
        elements = math.prod(left_shape)
        left_values = [float(value) for value in left["values"]]
        right_values = [float(value) for value in right["values"]]
        if len(left_values) != elements or len(right_values) != 2 * elements:
            raise ValueError(f"cached block-detail value count changed: {left['name']}")
        result.append({
            "name": left["name"], "dtype": left["dtype"],
            "shape_b1": left_shape, "shape_b2": right_shape,
            "b1_vs_b2_row0": difference(left_values, right_values[:elements]),
            "b2_row0_vs_row1": difference(
                right_values[:elements], right_values[elements:]),
        })
    return result


def summarize(processes: list[dict]) -> dict:
    policies = {}
    for policy in POLICIES:
        rows = [row for row in processes if row["precision_island"] == policy]
        if len(rows) != 2 or rows[0]["stages"] != rows[1]["stages"]:
            raise ValueError(f"{policy} block-detail metrics are not deterministic")
        stages = rows[0]["stages"]
        first_nonzero = next((row["name"] for row in stages
                              if not row["b1_vs_b2_row0"]["bitwise_equal"]), None)
        first_material = next((row["name"] for row in stages
                               if row["b1_vs_b2_row0"]["maximum"] >=
                               MATERIAL_THRESHOLD), None)
        policies[policy] = {
            "precision_island": policy,
            "stage_count": len(stages),
            "first_nonzero_stage": first_nonzero,
            "first_stage_at_or_above_maximum_1e_3": first_material,
            "maximum_error": max(
                row["b1_vs_b2_row0"]["maximum"] for row in stages),
            "maximum_relative_l2": max(
                row["b1_vs_b2_row0"]["relative_l2"] for row in stages),
            "all_b2_rows_bitwise_equal": all(
                row["b2_row0_vs_row1"]["bitwise_equal"] for row in stages),
            "stages": stages,
        }
    fp32 = {row["name"]: row for row in policies["fp32-linear"]["stages"]}
    common = [row for row in policies["bf16-ffn"]["stages"]
              if row["name"] in fp32]
    first_hundredfold = next((row["name"] for row in common
                              if row["b1_vs_b2_row0"]["maximum"] >=
                              MATERIAL_THRESHOLD and
                              row["b1_vs_b2_row0"]["relative_l2"] >=
                              100.0 * max(
                                  fp32[row["name"]]["b1_vs_b2_row0"]
                                  ["relative_l2"], 1.0e-12)), None)
    return {
        "schema_version": 1,
        "record_type": "cached_block_detail_audit",
        "status": "pass",
        "process_rows": len(processes),
        "policies": list(POLICIES),
        "runs_per_policy": 2,
        "block": 0,
        "material_maximum_threshold": MATERIAL_THRESHOLD,
        "first_hundredfold_bf16_ffn_stage": first_hundredfold,
        "policy_summaries": list(policies.values()),
    }


def short_name(name: str) -> str:
    return name.removeprefix(PREFIX + ".") if name != PREFIX else "output"


def render(summary: dict) -> str:
    policies = {row["precision_island"]: row
                for row in summary["policy_summaries"]}
    stages = policies["bf16-ffn"]["stages"]
    width, height = 1600, 760
    plot_left, plot_right = 110, 1560
    plot_top, plot_bottom = 105, 590
    minimum_log, maximum_log = -8.0, 0.0

    def y(value: float) -> float:
        level = max(minimum_log, min(maximum_log,
                                     math.log10(max(value, 1.0e-8))))
        return plot_bottom - (level - minimum_log) / (
            maximum_log - minimum_log) * (plot_bottom - plot_top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:23px;font-weight:700}.label{font-size:12px}'
        '.muted{fill:#94a3b8;font-size:12px}.axis{stroke:#334155;stroke-width:1}'
        '</style>',
        '<text x="30" y="38" class="title">Cached block-0 internal batch drift</text>',
        '<text x="30" y="63" class="muted">B1 vs B2 row0 · relative L2 · '
        'log10 scale · DeepSeek T2048 step0</text>',
    ]
    for exponent in range(-8, 1):
        line_y = y(10.0 ** exponent)
        parts.append(f'<line x1="{plot_left}" y1="{line_y:.2f}" '
                     f'x2="{plot_right}" y2="{line_y:.2f}" class="axis"/>')
        parts.append(f'<text x="30" y="{line_y + 4:.2f}" class="label">1e{exponent}</text>')
    count = len(stages)
    step = (plot_right - plot_left) / max(count, 1)
    fp32 = {row["name"]: row for row in policies["fp32-linear"]["stages"]}
    for index, row in enumerate(stages):
        x = plot_left + (index + 0.5) * step
        value = row["b1_vs_b2_row0"]["relative_l2"]
        parts.append(f'<circle cx="{x:.2f}" cy="{y(value):.2f}" r="5" fill="#f97316"/>')
        if row["name"] in fp32:
            reference = fp32[row["name"]]["b1_vs_b2_row0"]["relative_l2"]
            parts.append(f'<circle cx="{x:.2f}" cy="{y(reference):.2f}" r="4" '
                         f'fill="#38bdf8"/>')
        label = short_name(row["name"])
        parts.append(f'<text x="{x:.2f}" y="{plot_bottom + 18}" class="label" '
                     f'text-anchor="start" transform="rotate(58 {x:.2f} '
                     f'{plot_bottom + 18})">{label}</text>')
    parts.extend([
        '<circle cx="1160" cy="42" r="5" fill="#38bdf8"/>',
        '<text x="1172" y="46" class="label">FP32 Linear</text>',
        '<circle cx="1320" cy="42" r="5" fill="#f97316"/>',
        '<text x="1332" y="46" class="label">BF16 FFN-only</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = BLOCK.COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-cached-block-detail-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            policy_order = list(POLICIES) if run % 2 else list(reversed(POLICIES))
            for policy in policy_order:
                traces = {}
                records = {}
                for batch in ((1, 2) if run % 2 else (2, 1)):
                    trace_path = temporary / f"{policy}-b{batch}-r{run}.jsonl"
                    logits_path = temporary / f"{policy}-b{batch}-r{run}.bin"
                    completed = subprocess.run(
                        command(args, model, policy, batch, trace_path, logits_path),
                        text=True, capture_output=True, timeout=args.timeout_seconds)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            completed.stderr.strip() or completed.stdout.strip())
                    records[batch] = BLOCK.COMMON.last_json(completed.stdout)
                    traces[batch] = selected(trace_path)
                stages = compare(traces[1], traces[2])
                processes.append({
                    "schema_version": 1,
                    "record_type": "cached_block_detail_process",
                    "status": "pass",
                    "model": args.model,
                    "revision": model["revision"],
                    "context": args.context,
                    "decode_step": 0,
                    "block": 0,
                    "precision_island": policy,
                    "process_run": run,
                    "trace_record_count_b1": records[1]["trace_record_count"],
                    "trace_record_count_b2": records[2]["trace_record_count"],
                    "stages": stages,
                })
                print(json.dumps({"precision_island": policy,
                                  "process_run": run, "status": "pass"},
                                 sort_keys=True), flush=True)
    summary = summarize(processes)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in processes),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "block-detail.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_cached_block_detail: {error}", file=sys.stderr)
        raise SystemExit(2) from error
