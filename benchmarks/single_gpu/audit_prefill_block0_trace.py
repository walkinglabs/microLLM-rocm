#!/usr/bin/env python3
"""Locate the first block-0 full-prefill boundary that changes across batch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "audit_prefill_block0_trace_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

BATCHES = (1, 2, 4, 8)
PREFIX = "inference.cached_prefill.blocks.0"
STAGES = (
    "inference.cached_prefill.embedding_rows",
    PREFIX + ".attention_norm",
    PREFIX + ".attention.q_projection",
    PREFIX + ".attention.k_projection",
    PREFIX + ".attention.v_projection",
    PREFIX + ".attention.q_rope",
    PREFIX + ".attention.k_rope",
    PREFIX + ".attention.value",
    PREFIX + ".attention.cache_key",
    PREFIX + ".attention.cache_value",
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
        parser.error("prefill block-0 trace inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, batch: int,
            trace: Path, cache: Path) -> list[str]:
    return [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", COMMON.expanded(model["inference"]["token_ids"], args.context),
        "--device", "hip", "--top-k", "1", "--batch", str(batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", str(args.context + 1), "--new-tokens", "1",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--bf16-ffn", "false", "--bf16-attention", "false",
        "--workload", "decode", "--prefill-cache-output", str(cache),
        "--prefill-cache-layer", "0", "--trace-output", str(trace),
        "--trace-max-elements", str(2 * args.context * 1536),
        "--trace-value-filter", ",".join(STAGES),
    ]


def load_trace(path: Path, batch: int, context: int) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line]
    selected = {row["name"]: row for row in records if row.get("name") in STAGES}
    if set(selected) != set(STAGES):
        raise ValueError("prefill block-0 trace stages changed")
    captured = min(batch, 2)
    for name in STAGES:
        row = selected[name]
        shape = [int(value) for value in row.get("shape", [])]
        if (not shape or shape[0] != captured or
                row.get("values_truncated") or
                len(row.get("values", [])) != math.prod(shape)):
            raise ValueError(f"prefill block-0 trace capture changed: {name}")
        if shape[1] != context and name not in {
                PREFIX + ".attention.q_rope",
                PREFIX + ".attention.k_rope",
                PREFIX + ".attention.value",
                PREFIX + ".attention.cache_key",
                PREFIX + ".attention.cache_value"}:
            raise ValueError(f"prefill trace sequence axis changed: {name}")
    return selected


def difference(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("prefill trace comparison needs equal rows")
    maximum = 0.0
    square = 0.0
    reference_square = 0.0
    for left_value, right_value in zip(left, right):
        delta = abs(float(left_value) - float(right_value))
        maximum = max(maximum, delta)
        square += delta * delta
        reference_square += float(left_value) * float(left_value)
    return {
        "elements": len(left), "maximum": maximum,
        "rms": math.sqrt(square / len(left)),
        "relative_l2": math.sqrt(square / reference_square)
        if reference_square > 0.0 else 0.0,
        "bitwise_equal": left == right,
    }


def compare(reference: dict[str, dict], actual: dict[str, dict],
            batch: int) -> list[dict]:
    rows = []
    for name in STAGES:
        left = reference[name]
        right = actual[name]
        left_values = [float(value) for value in left["values"]]
        right_values = [float(value) for value in right["values"]]
        row_elements = len(left_values)
        if (left["dtype"] != right["dtype"] or
                len(right_values) != row_elements * min(batch, 2)):
            raise ValueError(f"prefill block-0 row shape changed: {name}")
        within = difference(
            right_values[:row_elements], right_values[row_elements:]) \
            if batch > 1 else {
                "elements": row_elements, "maximum": 0.0, "rms": 0.0,
                "relative_l2": 0.0, "bitwise_equal": True,
            }
        rows.append({
            "name": name, "dtype": left["dtype"],
            "shape_b1": left["shape"], "shape_actual": right["shape"],
            "b1_vs_batch_row0": difference(
                left_values, right_values[:row_elements]),
            "batch_row0_vs_row1": within,
        })
    return rows


def summarize(processes: list[dict]) -> dict:
    cases = []
    for batch in BATCHES:
        rows = [row for row in processes if row["batch"] == batch]
        if len(rows) != 2 or rows[0]["stages"] != rows[1]["stages"]:
            raise ValueError(f"B{batch} prefill trace metrics are not deterministic")
        cases.append({
            "batch": batch, "runs": 2,
            "first_nonzero_stage": next((
                stage["name"] for stage in rows[0]["stages"]
                if not stage["b1_vs_batch_row0"]["bitwise_equal"]), None),
            "maximum_error": max(
                stage["b1_vs_batch_row0"]["maximum"]
                for stage in rows[0]["stages"]),
            "maximum_rms_error": max(
                stage["b1_vs_batch_row0"]["rms"]
                for stage in rows[0]["stages"]),
            "all_within_batch_bitwise_equal": all(
                stage["batch_row0_vs_row1"]["bitwise_equal"]
                for stage in rows[0]["stages"]),
            "stages": rows[0]["stages"],
        })
    first = next((case["first_nonzero_stage"] for case in cases
                  if case["first_nonzero_stage"] is not None), None)
    return {
        "schema_version": 1,
        "record_type": "prefill_block0_trace_audit",
        "status": "pass", "process_rows": len(processes),
        "case_rows": len(cases), "batches": list(BATCHES),
        "runs_per_case": 2, "context": 2048,
        "captured_batch_rows": 2, "stage_count": len(STAGES),
        "first_nonzero_stage": first,
        "all_repeat_metrics_equal": True,
        "all_within_batch_bitwise_equal": all(
            case["all_within_batch_bitwise_equal"] for case in cases),
        "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1580, 650
    maximum = max(stage["b1_vs_batch_row0"]["maximum"]
                  for case in summary["cases"] for stage in case["stages"])
    scale = 180.0 / maximum if maximum else 1.0
    colors = {2: "#38bdf8", 4: "#f97316", 8: "#22c55e"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:11px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">Block-0 full-prefill boundary drift</text>',
        '<text x="30" y="62" class="muted">DeepSeek T2048 · complete row0 values '
        '· linear Max scale</text>',
    ]
    selected_cases = [case for case in summary["cases"] if case["batch"] > 1]
    for case_index, case in enumerate(selected_cases):
        y0 = 105 + case_index * 165
        parts.append(f'<text x="30" y="{y0 + 18}" class="label">B{case["batch"]}</text>')
        for index, stage in enumerate(case["stages"]):
            x = 130 + index * 140
            value = stage["b1_vs_batch_row0"]["maximum"]
            height_value = max(2.0, value * scale)
            parts.extend((
                f'<rect x="{x}" y="{y0 + 125 - height_value:.2f}" width="34" '
                f'height="{height_value:.2f}" fill="{colors[case["batch"]]}"/>',
                f'<text x="{x + 17}" y="{y0 + 145}" class="label" '
                f'text-anchor="middle" transform="rotate(55 {x + 17} {y0 + 145})">'
                f'{stage["name"].removeprefix(PREFIX + ".").removeprefix("inference.cached_prefill.")}</text>',
            ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-prefill-block0-trace-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            reference_trace = temporary / f"b1-r{run}.jsonl"
            reference_cache = temporary / f"b1-r{run}.bin"
            completed = subprocess.run(
                command(args, model, 1, reference_trace, reference_cache),
                text=True, capture_output=True, timeout=args.timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            reference = load_trace(reference_trace, 1, args.context)
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
                    actual = load_trace(trace, batch, args.context)
                    record = COMMON.last_json(current.stdout)
                if (record.get("status") != "pass" or
                        record.get("prefill_cache_exported") is not True or
                        record.get("trace_record_count") != 50):
                    raise ValueError(f"B{batch} cached prefill trace route changed")
                processes.append({
                    "schema_version": 1,
                    "record_type": "prefill_block0_trace_process",
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
    (args.output_directory / "prefill-trace.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_prefill_block0_trace: {error}", file=sys.stderr)
        raise SystemExit(2) from error
