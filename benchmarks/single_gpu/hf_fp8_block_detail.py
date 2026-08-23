#!/usr/bin/env python3
"""Compare FP32/FP8 internal stages for evidence-selected Transformer blocks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fp8_matrix as matrix  # noqa: E402
from hf_prefill_layer_drift import difference  # noqa: E402


DEFAULT_LAYERS = {
    "qwen2.5-0.5b": 21,
    "deepseek-r1-distill-qwen-1.5b": 27,
}


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
    result.models = result.models.split(",") if result.models else list(DEFAULT_LAYERS)
    if not set(result.models) <= set(DEFAULT_LAYERS):
        parser.error("every selected model needs an evidence-selected block")
    return result


def command(args: argparse.Namespace, model: dict, layer: int, policy: str,
            trace_path: Path) -> list[str]:
    seed = model["inference"]["token_ids"]
    tokens = [seed[index % len(seed)] for index in range(args.context)]
    prefix = f"inference.blocks.{layer}"
    result = [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(map(str, tokens)), "--device", "hip", "--top-k", "1",
        "--new-tokens", "0", "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-logits", "last", "--workload", "prefill",
        "--use-cache", "true", "--kv-cache-dtype", "fp32",
        "--trace-output", str(trace_path), "--trace-max-elements", "200000",
        "--trace-all-layer-details", "true", "--trace-value-filter", prefix,
    ]
    if policy == "fp8":
        result.extend([
            "--fp8-linear", "true", "--fp8-weight-scale-mode", "tensor-amax",
            "--fp8-weight-scale", "0.005", "--fp8-activation-scale-mode", "tensor-amax",
            "--fp8-activation-scale", "0.2", "--fp8-activation-minimum-scale", "0.0001",
        ])
    return result


def selected(path: Path, layer: int) -> list[dict]:
    prefix = f"inference.blocks.{layer}"
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in records if row["name"] == prefix or
            row["name"].startswith(prefix + ".")]
    if not rows or any(row.get("values_truncated") or not row.get("values")
                       for row in rows):
        raise RuntimeError("block detail trace is missing complete values")
    if len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError("block detail stage names must be unique")
    return rows


def compare(reference: list[dict], actual: list[dict]) -> list[dict]:
    if [row["name"] for row in reference] != [row["name"] for row in actual]:
        raise RuntimeError("FP32/FP8 block detail names changed")
    result = []
    for left, right in zip(reference, actual):
        if left["shape"] != right["shape"]:
            raise RuntimeError("FP32/FP8 block detail shapes changed")
        result.append({"name": left["name"], "shape": left["shape"],
                       **difference(left["values"], right["values"])})
    return result


def main() -> int:
    args = options()
    models = matrix.load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    summaries = []
    for model in models:
        layer = DEFAULT_LAYERS[model["name"]]
        traces = {}
        workers = []
        for policy in ("fp32", "fp8"):
            path = args.output_directory / f"{model['name']}-{policy}-block{layer}.jsonl"
            pre = matrix.require_idle(args.physical_gpu_index,
                args.max_idle_vram_percent, args.max_idle_use_percent,
                f"{model['name']} {policy} block detail pre")
            completed = subprocess.run(command(args, model, layer, policy, path),
                                       capture_output=True, text=True)
            post = matrix.require_idle(args.physical_gpu_index,
                args.max_idle_vram_percent, args.max_idle_use_percent,
                f"{model['name']} {policy} block detail post")
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "block detail worker failed")
            output = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            if len(output) != 1 or output[0].get("status") != "pass":
                raise RuntimeError("block detail worker output contract failed")
            traces[policy] = selected(path, layer)
            workers.append({"policy": policy, "pre_run_gpu_state": pre,
                            "post_run_gpu_state": post,
                            "trace_record_count": output[0]["trace_record_count"]})
        stages = compare(traces["fp32"], traces["fp8"])
        deltas = []
        previous = 0.0
        for row in stages:
            deltas.append({"name": row["name"],
                           "delta_relative_l2": row["relative_l2"] - previous})
            previous = row["relative_l2"]
        largest = max(deltas, key=lambda row: row["delta_relative_l2"])
        summary = {
            "schema_version": 1, "status": "pass",
            "track": "official_fp8_block_detail", "model": model["name"],
            "revision": model["revision"], "context": args.context, "layer": layer,
            "stage_count": len(stages), "largest_relative_l2_jump": largest,
            "workers": workers, "stages": stages,
        }
        summaries.append(summary)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps({"model": model["name"], "status": "pass"}), flush=True)
    document = {"schema_version": 1, "status": "pass",
                "track": "official_fp8_block_detail", "context": args.context,
                "summaries": summaries,
                "boundary": "complete selected block values; synchronous diagnostic, not performance evidence"}
    (args.output_directory / "summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
