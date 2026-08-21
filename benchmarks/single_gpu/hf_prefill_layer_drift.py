#!/usr/bin/env python3
"""Compare complete B1/B2 prefill logits and every inference block output."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import hf_continuous_matrix as matrix


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--prompt-offset", type=int, default=5)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--trace-max-elements", type=int, default=700000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-traces", action="store_true")
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    if result.prompt_offset < 0 or result.prompt_length <= 0 or result.runs <= 0 or \
            result.trace_max_elements <= 0 or result.timeout_seconds <= 0:
        parser.error("offset must be nonnegative and other numeric inputs positive")
    return result


def command(binary: Path, model: dict, tokens: list[int], batch: int,
            trace: Path, trace_max_elements: int) -> list[str]:
    return [
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(map(str, tokens)), "--device", "hip",
        "--top-k", "1", "--new-tokens", "0", "--workload", "prefill",
        "--batch", str(batch), "--bf16-ffn", "true",
        "--bf16-attention", "true", "--prefill-logits", "last",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--trace-output", str(trace),
        "--trace-max-elements", str(trace_max_elements),
    ]


def load_trace(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines()]
    selected = {row["name"]: row for row in records
                if row["kind"] in ("layer", "output")}
    if len(selected) != len([row for row in records
                             if row["kind"] in ("layer", "output")]):
        raise RuntimeError("trace layer/output names must be unique")
    return selected


def difference(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        raise RuntimeError("numeric comparison needs equal non-empty vectors")
    absolute = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    maximum = max(absolute)
    maximum_index = absolute.index(maximum)
    square = sum(value * value for value in absolute)
    reference_square = sum(float(value) * float(value) for value in left)
    return {
        "elements": len(left),
        "max_abs": maximum,
        "max_abs_index": maximum_index,
        "mean_abs": sum(absolute) / len(absolute),
        "rms_abs": math.sqrt(square / len(absolute)),
        "relative_l2": math.sqrt(square / reference_square)
        if reference_square > 0 else 0.0,
        "exact": maximum == 0.0,
    }


def compare_traces(b1: dict[str, dict], b2: dict[str, dict]) -> list[dict]:
    if set(b1) != set(b2):
        raise RuntimeError("B1/B2 trace names differ")
    rows = []
    for name, left in b1.items():
        right = b2[name]
        if left.get("values_truncated") or right.get("values_truncated"):
            raise RuntimeError(f"trace values truncated: {name}")
        left_shape = [int(value) for value in left["shape"]]
        right_shape = [int(value) for value in right["shape"]]
        if not left_shape or right_shape[0] != 2 * left_shape[0] or \
                right_shape[1:] != left_shape[1:]:
            raise RuntimeError(f"B1/B2 trace shape changed: {name}")
        row_elements = math.prod(left_shape)
        left_values = [float(value) for value in left["values"]]
        right_values = [float(value) for value in right["values"]]
        if len(left_values) != row_elements or len(right_values) != 2 * row_elements:
            raise RuntimeError(f"trace value count changed: {name}")
        rows.append({
            "name": name,
            "shape_b1": left_shape,
            "shape_b2": right_shape,
            "b1_vs_b2_row0": difference(left_values, right_values[:row_elements]),
            "b2_row0_vs_row1": difference(
                right_values[:row_elements], right_values[row_elements:]),
        })
    return rows


def main() -> int:
    args = options()
    model = matrix.load_models(args.manifest, [args.model])[0]
    seed = [int(token) for token in model["inference"]["token_ids"]]
    tokens = [seed[(index + args.prompt_offset) % len(seed)]
              for index in range(args.prompt_length)]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for process_run in range(1, args.runs + 1):
        traces = {}
        app_records = {}
        for batch in (1, 2):
            trace_path = args.output_directory / f"run{process_run}-b{batch}.jsonl"
            completed = subprocess.run(
                command(args.binary, model, tokens, batch, trace_path,
                        args.trace_max_elements),
                capture_output=True, text=True, timeout=args.timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise RuntimeError("prefill trace worker must emit one JSON line")
            app_records[batch] = json.loads(lines[0])
            traces[batch] = load_trace(trace_path)
            if not args.keep_traces:
                trace_path.unlink()
        stages = compare_traces(traces[1], traces[2])
        row = {
            "schema_version": 1,
            "status": "pass",
            "record_type": "official_prefill_layer_drift",
            "model": model["name"],
            "revision": model["revision"],
            "process_run": process_run,
            "prompt_offset": args.prompt_offset,
            "prompt_length": args.prompt_length,
            "trace_record_count_b1": app_records[1]["trace_record_count"],
            "trace_record_count_b2": app_records[2]["trace_record_count"],
            "stages": stages,
        }
        rows.append(row)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps({"process_run": process_run, "status": "pass"},
                         sort_keys=True), flush=True)
    first = rows[0]["stages"]
    if any(row["stages"] != first for row in rows[1:]):
        raise RuntimeError("layer drift metrics changed across fresh processes")
    first_nonzero = next((row["name"] for row in first
                          if not row["b1_vs_b2_row0"]["exact"]), None)
    if first_nonzero is None:
        raise RuntimeError("B1/B2 layer audit expected a measured difference")
    if any(not row["b2_row0_vs_row1"]["exact"] for row in first):
        raise RuntimeError("duplicate B2 rows diverged inside the model")
    summary = {
        "schema_version": 1,
        "track": "official_prefill_layer_drift",
        "status": "pass",
        "model": model["name"],
        "revision": model["revision"],
        "runs": args.runs,
        "prompt_offset": args.prompt_offset,
        "prompt_length": args.prompt_length,
        "stage_count": len(first),
        "first_nonzero_stage": first_nonzero,
        "duplicate_b2_rows_exact_at_every_stage": True,
        "stages": first,
        "measurement_boundary": "full host snapshots; no performance claim",
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
