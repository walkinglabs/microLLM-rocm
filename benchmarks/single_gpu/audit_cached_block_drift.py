#!/usr/bin/env python3
"""Locate the first cached block that amplifies B1/B2 numerical drift."""

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
    "audit_cached_block_drift_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

POLICIES = {"fp32-linear": False, "bf16-ffn": True}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--trace-max-elements", type=int, default=200000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context <= 0 or args.runs < 2 or
            args.trace_max_elements < 151936 or args.timeout_seconds <= 0):
        parser.error("cached block drift inputs are outside the fixed contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, policy: str, batch: int,
            trace: Path, logits: Path) -> list[str]:
    return [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", COMMON.expanded(model["inference"]["token_ids"], args.context),
        "--device", "hip", "--top-k", "1", "--batch", str(batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", str(args.context + 1), "--new-tokens", "1",
        "--warmup", "0", "--steps", "1", "--prefill-warmup", "0",
        "--prefill-steps", "1", "--bf16-ffn",
        str(POLICIES[policy]).lower(), "--bf16-attention", "false",
        "--workload", "decode", "--cache-logits-output", str(logits),
        "--cache-logits-step", "0", "--trace-output", str(trace),
        "--trace-max-elements", str(args.trace_max_elements),
    ]


def selected_name(name: str) -> bool:
    if name in {"inference.cached.embedding", "inference.cached.final_norm",
                "inference.cached.logits"}:
        return True
    prefix = "inference.cached.blocks."
    suffix = name.removeprefix(prefix)
    return name.startswith(prefix) and suffix.isdigit()


def load_trace(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line]
    selected = {row["name"]: row for row in records
                if row.get("kind") in {"layer", "output"} and
                selected_name(str(row.get("name", "")))}
    if len(selected) != 31:
        raise ValueError(f"cached trace has {len(selected)} selected stages, expected 31")
    return selected


def ordered_names() -> list[str]:
    return (["inference.cached.embedding"] +
            [f"inference.cached.blocks.{index}" for index in range(28)] +
            ["inference.cached.final_norm", "inference.cached.logits"])


def compare_traces(b1: dict[str, dict], b2: dict[str, dict]) -> list[dict]:
    if list(b1) != list(b2) or set(b1) != set(ordered_names()):
        raise ValueError("cached B1/B2 trace names changed")
    rows = []
    for name in ordered_names():
        left = b1[name]
        right = b2[name]
        if left.get("values_truncated") or right.get("values_truncated"):
            raise ValueError(f"trace values truncated: {name}")
        left_shape = [int(value) for value in left["shape"]]
        right_shape = [int(value) for value in right["shape"]]
        if not left_shape or right_shape[0] != 2 * left_shape[0] or \
                right_shape[1:] != left_shape[1:]:
            raise ValueError(f"trace shape changed: {name}")
        elements = math.prod(left_shape)
        left_values = [float(value) for value in left["values"]]
        right_values = [float(value) for value in right["values"]]
        if len(left_values) != elements or len(right_values) != 2 * elements:
            raise ValueError(f"trace value count changed: {name}")
        maximum, rms, bitwise = COMMON.error(
            left_values, right_values[:elements])
        row_maximum, row_rms, row_bitwise = COMMON.error(
            right_values[:elements], right_values[elements:])
        rows.append({
            "name": name, "shape_b1": left_shape, "shape_b2": right_shape,
            "b1_vs_b2_row0": {"maximum": maximum, "rms": rms,
                               "bitwise_equal": bitwise},
            "b2_row0_vs_row1": {"maximum": row_maximum, "rms": row_rms,
                                 "bitwise_equal": row_bitwise},
        })
    return rows


def summarize(processes: list[dict]) -> dict:
    by_policy = {}
    for policy in POLICIES:
        rows = [row for row in processes if row["precision_island"] == policy]
        if len(rows) != 2 or rows[0]["stages"] != rows[1]["stages"]:
            raise ValueError(f"{policy} trace metrics are not deterministic")
        stages = rows[0]["stages"]
        by_policy[policy] = {
            "precision_island": policy,
            "first_nonzero_stage": next((
                row["name"] for row in stages
                if not row["b1_vs_b2_row0"]["bitwise_equal"]), None),
            "maximum_error": max(
                row["b1_vs_b2_row0"]["maximum"] for row in stages),
            "maximum_rms_error": max(
                row["b1_vs_b2_row0"]["rms"] for row in stages),
            "all_b2_rows_bitwise_equal": all(
                row["b2_row0_vs_row1"]["bitwise_equal"] for row in stages),
            "stages": stages,
        }
    fp32 = {row["name"]: row for row in by_policy["fp32-linear"]["stages"]}
    ffn = by_policy["bf16-ffn"]["stages"]
    first_amplified = next((row["name"] for row in ffn
                            if row["b1_vs_b2_row0"]["maximum"] > 1.0e-6 and
                            row["b1_vs_b2_row0"]["maximum"] >=
                            10.0 * fp32[row["name"]]["b1_vs_b2_row0"]["maximum"]),
                           None)
    return {
        "schema_version": 1, "record_type": "cached_block_drift_audit",
        "status": "pass", "process_rows": len(processes),
        "policies": list(POLICIES), "runs_per_policy": 2,
        "selected_stage_count": 31,
        "first_tenfold_bf16_ffn_stage": first_amplified,
        "policy_summaries": list(by_policy.values()),
    }


def render(summary: dict) -> str:
    width, height = 1420, 650
    colors = {"fp32-linear": "#38bdf8", "bf16-ffn": "#f97316"}
    all_values = [row["b1_vs_b2_row0"]["maximum"]
                  for policy in summary["policy_summaries"]
                  for row in policy["stages"]]
    maximum = max(all_values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:12px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">Cached step0 B1/B2 block drift</text>',
        '<text x="30" y="62" class="muted">complete block outputs · linear y scale · '
        'FP32 Linear vs BF16 FFN-only</text>',
    ]
    for policy_index, policy in enumerate(summary["policy_summaries"]):
        y0 = 100 + policy_index * 260
        parts.append(f'<text x="30" y="{y0}" class="label">'
                     f'{policy["precision_island"]}</text>')
        for index, row in enumerate(policy["stages"]):
            x = 170 + index * 38
            value = row["b1_vs_b2_row0"]["maximum"]
            bar = max(2.0, 190.0 * value / maximum) if maximum else 2.0
            parts.extend((
                f'<rect x="{x}" y="{y0 + 205 - bar:.2f}" width="24" '
                f'height="{bar:.2f}" fill="{colors[policy["precision_island"]]}"/>',
                f'<text x="{x + 12}" y="{y0 + 225}" class="label" '
                f'text-anchor="middle" transform="rotate(70 {x + 12} {y0 + 225})">'
                f'{index}</text>',
            ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with tempfile.TemporaryDirectory(prefix="microllm-cached-block-drift-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            policy_order = list(POLICIES) if run % 2 else list(reversed(POLICIES))
            for policy in policy_order:
                traces = {}
                app_records = {}
                for batch in ((1, 2) if run % 2 else (2, 1)):
                    trace_path = temporary / f"{policy}-b{batch}-r{run}.jsonl"
                    logits_path = temporary / f"{policy}-b{batch}-r{run}.bin"
                    completed = subprocess.run(
                        command(args, model, policy, batch, trace_path, logits_path),
                        text=True, capture_output=True, timeout=args.timeout_seconds)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            completed.stderr.strip() or completed.stdout.strip())
                    app_records[batch] = COMMON.last_json(completed.stdout)
                    traces[batch] = load_trace(trace_path)
                stages = compare_traces(traces[1], traces[2])
                row = {
                    "schema_version": 1, "status": "pass",
                    "record_type": "cached_block_drift_process",
                    "model": args.model, "revision": model["revision"],
                    "context": args.context, "decode_step": 0,
                    "precision_island": policy, "process_run": run,
                    "trace_record_count_b1": app_records[1]["trace_record_count"],
                    "trace_record_count_b2": app_records[2]["trace_record_count"],
                    "stages": stages,
                }
                processes.append(row)
                print(json.dumps({"precision_island": policy,
                                  "process_run": run, "status": "pass"},
                                 sort_keys=True), flush=True)
    summary = summarize(processes)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in processes),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "block-drift.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_cached_block_drift: {error}", file=sys.stderr)
        raise SystemExit(2) from error
