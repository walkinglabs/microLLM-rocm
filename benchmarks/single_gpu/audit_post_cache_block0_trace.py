#!/usr/bin/env python3
"""Locate the first drift after invariant block-0 prefill K/V cache."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_SPEC = importlib.util.spec_from_file_location(
    "audit_post_cache_block0_trace_base",
    Path(__file__).with_name("audit_prefill_block0_trace.py"))
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE)

COMMON = BASE.COMMON
BATCHES = BASE.BATCHES
PREFIX = BASE.PREFIX
Q_SOLUTION = 296100
KV_SOLUTION = 292135
STAGES = BASE.STAGES + (
    PREFIX + ".attention.context",
    PREFIX + ".attention.output",
    PREFIX + ".attention_output",
    PREFIX + ".attention_residual",
    PREFIX + ".ffn_norm",
    PREFIX + ".ffn_output",
    PREFIX + ".output",
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context <= 0 or args.runs != 2 or args.timeout_seconds <= 0):
        parser.error("post-cache block-0 trace inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, batch: int,
            trace: Path, cache: Path) -> list[str]:
    result = BASE.command(args, model, batch, trace, cache)
    filter_index = result.index("--trace-value-filter") + 1
    result[filter_index] = ",".join(STAGES)
    result.extend([
        "--fp32-prefill-q-solution-index", str(Q_SOLUTION),
        "--fp32-prefill-kv-solution-index", str(KV_SOLUTION),
    ])
    return result


def load_trace(path: Path, batch: int) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line]
    selected = {row["name"]: row for row in records if row.get("name") in STAGES}
    if set(selected) != set(STAGES):
        raise ValueError("post-cache trace stages changed")
    captured = min(batch, 2)
    for name in STAGES:
        row = selected[name]
        shape = [int(value) for value in row.get("shape", [])]
        if (not shape or shape[0] != captured or row.get("values_truncated") or
                len(row.get("values", [])) != math.prod(shape)):
            raise ValueError(f"post-cache trace values changed: {name}")
    return selected


def compare(reference: dict[str, dict], actual: dict[str, dict],
            batch: int) -> list[dict]:
    rows = []
    for name in STAGES:
        left = reference[name]
        right = actual[name]
        left_values = [float(value) for value in left["values"]]
        right_values = [float(value) for value in right["values"]]
        elements = len(left_values)
        if (left["dtype"] != right["dtype"] or
                len(right_values) != elements * min(batch, 2)):
            raise ValueError(f"post-cache trace row shape changed: {name}")
        within = BASE.difference(
            right_values[:elements], right_values[elements:]) \
            if batch > 1 else {
                "elements": elements, "maximum": 0.0, "rms": 0.0,
                "relative_l2": 0.0, "bitwise_equal": True,
            }
        rows.append({
            "name": name, "dtype": left["dtype"],
            "shape_b1": left["shape"], "shape_actual": right["shape"],
            "b1_vs_batch_row0": BASE.difference(
                left_values, right_values[:elements]),
            "batch_row0_vs_row1": within,
        })
    return rows


def summarize(processes: list[dict]) -> dict:
    cases = []
    for batch in BATCHES:
        rows = [row for row in processes if row["batch"] == batch]
        if len(rows) != 2 or rows[0]["stages"] != rows[1]["stages"]:
            raise ValueError(f"B{batch} post-cache metrics are not deterministic")
        stages = rows[0]["stages"]
        cases.append({
            "batch": batch, "runs": 2,
            "first_nonzero_stage": next((
                stage["name"] for stage in stages
                if not stage["b1_vs_batch_row0"]["bitwise_equal"]), None),
            "first_nonzero_after_cache": next((
                stage["name"] for stage in stages[len(BASE.STAGES):]
                if not stage["b1_vs_batch_row0"]["bitwise_equal"]), None),
            "maximum_error": max(
                stage["b1_vs_batch_row0"]["maximum"] for stage in stages),
            "all_within_batch_bitwise_equal": all(
                stage["batch_row0_vs_row1"]["bitwise_equal"]
                for stage in stages),
            "stages": stages,
        })
    first_after = next((case["first_nonzero_after_cache"] for case in cases
                        if case["first_nonzero_after_cache"] is not None), None)
    return {
        "schema_version": 1,
        "record_type": "post_cache_block0_trace_audit",
        "status": "pass", "process_rows": len(processes),
        "case_rows": len(cases), "batches": list(BATCHES),
        "runs_per_case": 2, "context": 2048,
        "q_solution_index": Q_SOLUTION, "kv_solution_index": KV_SOLUTION,
        "captured_batch_rows": 2, "stage_count": len(STAGES),
        "cache_stage_count": len(BASE.STAGES),
        "first_nonzero_after_cache": first_after,
        "all_repeat_metrics_equal": True,
        "all_cache_cross_batch_bitwise_equal": all(
            all(stage["b1_vs_batch_row0"]["bitwise_equal"]
                for stage in case["stages"]
                if stage["name"] in {
                    PREFIX + ".attention.cache_key",
                    PREFIX + ".attention.cache_value"})
            for case in cases),
        "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1640, 660
    maximum = max(stage["b1_vs_batch_row0"]["maximum"]
                  for case in summary["cases"] for stage in case["stages"])
    scale = 180.0 / maximum if maximum else 1.0
    colors = {2: "#38bdf8", 4: "#f97316", 8: "#22c55e"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:10px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">First drift after exact block-0 cache</text>',
        '<text x="30" y="62" class="muted">DeepSeek T2048 · Q296100 · KV292135 '
        '· complete first two batch rows</text>',
    ]
    for case_index, case in enumerate(
            row for row in summary["cases"] if row["batch"] > 1):
        y0 = 105 + case_index * 170
        parts.append(f'<text x="30" y="{y0 + 18}" class="label">B{case["batch"]}</text>')
        for index, stage in enumerate(case["stages"]):
            x = 100 + index * 88
            value = stage["b1_vs_batch_row0"]["maximum"]
            bar = max(2.0, value * scale)
            label = stage["name"].removeprefix(PREFIX + ".").removeprefix(
                "inference.cached_prefill.")
            parts.extend((
                f'<rect x="{x}" y="{y0 + 125 - bar:.2f}" width="28" '
                f'height="{bar:.2f}" fill="{colors[case["batch"]]}"/>',
                f'<text x="{x + 14}" y="{y0 + 145}" class="label" '
                f'text-anchor="middle" transform="rotate(58 {x + 14} {y0 + 145})">'
                f'{label}</text>',
            ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-post-cache-block0-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            ref_trace = temporary / f"b1-r{run}.jsonl"
            ref_cache = temporary / f"b1-r{run}.bin"
            completed = subprocess.run(
                command(args, model, 1, ref_trace, ref_cache), text=True,
                capture_output=True, timeout=args.timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            reference = load_trace(ref_trace, 1)
            for batch in BATCHES:
                if batch == 1:
                    actual = reference
                    record = COMMON.last_json(completed.stdout)
                else:
                    trace = temporary / f"b{batch}-r{run}.jsonl"
                    cache = temporary / f"b{batch}-r{run}.bin"
                    current = subprocess.run(
                        command(args, model, batch, trace, cache), text=True,
                        capture_output=True, timeout=args.timeout_seconds)
                    if current.returncode != 0:
                        raise RuntimeError(
                            current.stderr.strip() or current.stdout.strip())
                    actual = load_trace(trace, batch)
                    record = COMMON.last_json(current.stdout)
                if (record.get("status") != "pass" or
                        record.get("trace_record_count") != 50 or
                        record.get("fp32_prefill_q_solution_index") != Q_SOLUTION or
                        record.get("fp32_prefill_kv_solution_index") != KV_SOLUTION or
                        record.get("fp32_solution_registered_entries") != 2 or
                        record.get("fp32_solution_registry_hits") != 84):
                    raise ValueError(f"B{batch} post-cache route changed")
                processes.append({
                    "schema_version": 1,
                    "record_type": "post_cache_block0_trace_process",
                    "status": "pass", "model": args.model,
                    "revision": model["revision"], "context": args.context,
                    "batch": batch, "process_run": run,
                    "trace_record_count": record["trace_record_count"],
                    "stages": compare(reference, actual, batch),
                })
                print(json.dumps({"batch": batch, "process_run": run,
                                  "status": "pass"}, sort_keys=True), flush=True)
                if batch != 1:
                    del actual
            del reference
    summary = summarize(processes)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in processes),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "post-cache-trace.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_post_cache_block0_trace: {error}", file=sys.stderr)
        raise SystemExit(2) from error
