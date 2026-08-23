#!/usr/bin/env python3
"""Capture every official-model Linear input range before dynamic FP8 scaling."""

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


LINEAR_INPUT = re.compile(
    r"^inference\.blocks\.(\d+)\."
    r"(attention_norm|attention\.context|ffn_norm|ffn\.activated)$")
BOUNDARIES = (
    "attention_norm", "attention.context", "ffn_norm", "ffn.activated")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--activation-scale", type=float, default=0.2)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file() or \
            result.context <= 0 or not math.isfinite(result.activation_scale) or \
            result.activation_scale <= 0:
        parser.error("manifest, binary, context or activation scale is invalid")
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
        "--trace-output", str(trace_path), "--trace-max-elements", "1",
        "--trace-all-layer-details", "true",
    ]


def select_ranges(records: list[dict], model: dict, activation_scale: float) -> list[dict]:
    representable = activation_scale * 240.0
    selected = []
    for record in records:
        match = LINEAR_INPUT.match(record.get("name", ""))
        if match is None:
            continue
        stats = record["statistics"]
        if stats["numel"] <= 0 or stats["finite_count"] != stats["numel"]:
            raise RuntimeError("activation trace contains non-finite or empty values")
        absolute_maximum = max(abs(stats["minimum"]), abs(stats["maximum"]))
        selected.append({
            "schema_version": 1, "status": "pass",
            "track": "official_fp8_activation_range",
            "model": model["name"], "revision": model["revision"],
            "layer": int(match.group(1)), "boundary": match.group(2),
            "shape": record["shape"], "dtype": record["dtype"],
            "numel": stats["numel"], "minimum": stats["minimum"],
            "maximum": stats["maximum"], "absolute_maximum": absolute_maximum,
            "activation_scale": activation_scale,
            "representable_magnitude": representable,
            "range_ratio": absolute_maximum / representable,
            "potential_saturation": absolute_maximum > representable,
            "used_positive_levels": absolute_maximum / activation_scale,
        })
    expected = model["layers"] * len(BOUNDARIES)
    keys = {(row["layer"], row["boundary"]) for row in selected}
    if len(selected) != expected or len(keys) != expected or any(
            (layer, boundary) not in keys for layer in range(model["layers"])
            for boundary in BOUNDARIES):
        raise RuntimeError("all-layer trace is missing a Linear input boundary")
    return selected


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
        model = dict(model)
        config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
        model["layers"] = int(config["num_hidden_layers"])
        trace_path = args.output_directory / f"{model['name']}-trace.jsonl"
        pre = matrix.require_idle(
            args.physical_gpu_index, args.max_idle_vram_percent,
            args.max_idle_use_percent, f"{model['name']} activation trace pre")
        completed = subprocess.run(
            command(args, model, trace_path), capture_output=True, text=True)
        post = matrix.require_idle(
            args.physical_gpu_index, args.max_idle_vram_percent,
            args.max_idle_use_percent, f"{model['name']} activation trace post")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "activation trace worker failed")
        output = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        if len(output) != 1 or output[0].get("status") != "pass":
            raise RuntimeError("activation trace worker output contract failed")
        records = [json.loads(line) for line in trace_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        selected = select_ranges(records, model, args.activation_scale)
        rows.extend(selected)
        worker = {
            "schema_version": 1, "status": "pass",
            "track": "official_fp8_activation_range_worker",
            "model": model["name"], "revision": model["revision"],
            "context": args.context, "trace_records": len(records),
            "selected_boundaries": len(selected),
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
            maximum = max(selected, key=lambda row: row["absolute_maximum"])
            aggregates.append({
                "model": model["name"], "boundary": boundary,
                "layers": len(selected),
                "absolute_maximum_min": min(row["absolute_maximum"] for row in selected),
                "absolute_maximum_p50": statistics.median(
                    row["absolute_maximum"] for row in selected),
                "absolute_maximum_max": maximum["absolute_maximum"],
                "maximum_layer": maximum["layer"],
                "potential_saturation_layers": sum(
                    row["potential_saturation"] for row in selected),
                "range_ratio_max": maximum["range_ratio"],
            })
    summary = {
        "schema_version": 1, "status": "pass",
        "track": "official_fp8_activation_range", "context": args.context,
        "activation_scale": args.activation_scale,
        "representable_magnitude": args.activation_scale * 240.0,
        "models": [model["name"] for model in models],
        "boundaries": list(BOUNDARIES), "rows": len(rows),
        "workers": workers, "aggregates": aggregates,
        "potential_saturation_rows": sum(row["potential_saturation"] for row in rows),
        "boundary": (
            "FP32 baseline activations; complete-Tensor statistics with one sample value "
            "serialized; synchronous diagnostic trace, not performance evidence"),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
