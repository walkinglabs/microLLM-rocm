#!/usr/bin/env python3
"""Measure token-row activation range skew inside official-model Linear inputs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fp8_matrix as matrix  # noqa: E402
from hf_activation_range import BOUNDARIES, LINEAR_INPUT  # noqa: E402


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file() or result.context <= 0:
        parser.error("manifest, binary or context is invalid")
    result.models = result.models.split(",") if result.models else None
    return result


def command(args: argparse.Namespace, model: dict, trace_path: Path) -> list[str]:
    tokens = model["inference"]["token_ids"]
    prompt = [tokens[index % len(tokens)] for index in range(args.context)]
    return [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(map(str, prompt)), "--device", "hip", "--top-k", "1",
        "--new-tokens", "0", "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-logits", "last", "--workload", "prefill",
        "--use-cache", "true", "--kv-cache-dtype", "fp32",
        "--trace-output", str(trace_path), "--trace-max-elements", "1000000",
        "--trace-all-layer-details", "true", "--trace-value-filter",
        ",".join(BOUNDARIES),
    ]


def row_range(record: dict, model: dict) -> dict | None:
    match = LINEAR_INPUT.match(record.get("name", ""))
    if match is None:
        return None
    shape = record["shape"]
    values = record["values"]
    width = shape[-1]
    if width <= 0 or not values or len(values) != math.prod(shape) or \
            record.get("values_truncated"):
        raise RuntimeError("selected activation trace must contain every value")
    row_count = len(values) // width
    row_amax = [max(abs(value) for value in values[row * width:(row + 1) * width])
                for row in range(row_count)]
    if any(not math.isfinite(value) or value <= 0 for value in row_amax):
        raise RuntimeError("activation row amax must be finite and positive")
    tensor_amax = max(row_amax)
    p50 = statistics.median(row_amax)
    return {
        "schema_version": 1, "status": "pass",
        "track": "official_fp8_activation_row_range",
        "model": model["name"], "revision": model["revision"],
        "layer": int(match.group(1)), "boundary": match.group(2),
        "shape": shape, "width": width, "rows": row_count,
        "tensor_amax": tensor_amax,
        "row_amax_min": min(row_amax), "row_amax_p50": p50,
        "row_amax_max": tensor_amax,
        "row_spread": tensor_amax / min(row_amax),
        "p50_to_tensor_amax": p50 / tensor_amax,
        "rows_at_or_below_quarter_tensor_amax": sum(
            value <= tensor_amax * 0.25 for value in row_amax),
        "row_amax": row_amax,
    }


def main() -> int:
    args = options()
    models = matrix.load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    worker_path = args.output_directory / "workers.jsonl"
    raw_path.write_text("", encoding="utf-8")
    worker_path.write_text("", encoding="utf-8")
    rows = []
    workers = []
    for model in models:
        config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
        layers = int(config["num_hidden_layers"])
        trace_path = args.output_directory / f"{model['name']}-trace.jsonl"
        pre = matrix.require_idle(
            args.physical_gpu_index, args.max_idle_vram_percent,
            args.max_idle_use_percent, f"{model['name']} row trace pre")
        completed = subprocess.run(
            command(args, model, trace_path), capture_output=True, text=True)
        post = matrix.require_idle(
            args.physical_gpu_index, args.max_idle_vram_percent,
            args.max_idle_use_percent, f"{model['name']} row trace post")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "activation row worker failed")
        output = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        if len(output) != 1 or output[0].get("status") != "pass":
            raise RuntimeError("activation row worker output contract failed")
        records = [json.loads(line) for line in trace_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        selected = [row for record in records
                    if (row := row_range(record, model)) is not None]
        expected = layers * len(BOUNDARIES)
        if len(selected) != expected:
            raise RuntimeError("activation row trace boundary count changed")
        rows.extend(selected)
        worker = {
            "schema_version": 1, "status": "pass",
            "track": "official_fp8_activation_row_range_worker",
            "model": model["name"], "context": args.context,
            "trace_records": len(records), "selected_boundaries": len(selected),
            "forward_ms": output[0]["forward_ms"],
            "pre_run_gpu_state": pre, "post_run_gpu_state": post,
        }
        workers.append(worker)
        with worker_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(worker, sort_keys=True) + "\n")
        with raw_path.open("a", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(worker, sort_keys=True), flush=True)
    aggregates = []
    for model in models:
        for boundary in BOUNDARIES:
            selected = [row for row in rows if row["model"] == model["name"] and
                        row["boundary"] == boundary]
            maximum = max(selected, key=lambda row: row["row_spread"])
            aggregates.append({
                "model": model["name"], "boundary": boundary,
                "layers": len(selected),
                "row_spread_p50": statistics.median(
                    row["row_spread"] for row in selected),
                "row_spread_max": maximum["row_spread"],
                "maximum_spread_layer": maximum["layer"],
                "p50_to_tensor_amax_p50": statistics.median(
                    row["p50_to_tensor_amax"] for row in selected),
                "quarter_range_rows": sum(
                    row["rows_at_or_below_quarter_tensor_amax"] for row in selected),
                "total_rows": sum(row["rows"] for row in selected),
            })
    summary = {
        "schema_version": 1, "status": "pass",
        "track": "official_fp8_activation_row_range", "context": args.context,
        "models": [model["name"] for model in models], "rows": len(rows),
        "workers": workers, "aggregates": aggregates,
        "boundary": (
            "FP32 T8 baseline; complete selected Tensor values; token-row amax "
            "diagnostic, not performance evidence"),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
