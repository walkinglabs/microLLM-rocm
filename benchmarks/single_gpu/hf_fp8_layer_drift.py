#!/usr/bin/env python3
"""Compare complete FP32 and shared-dynamic-FP8 model stages."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fp8_matrix as matrix  # noqa: E402
from hf_prefill_layer_drift import difference  # noqa: E402


BLOCK = re.compile(r"^inference\.blocks\.(\d+)$")


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


def command(args: argparse.Namespace, model: dict, policy: str,
            trace_path: Path) -> list[str]:
    seed = model["inference"]["token_ids"]
    tokens = [seed[index % len(seed)] for index in range(args.context)]
    result = [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(map(str, tokens)), "--device", "hip", "--top-k", "1",
        "--new-tokens", "0", "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-logits", "last", "--workload", "prefill",
        "--use-cache", "true", "--kv-cache-dtype", "fp32",
        "--trace-output", str(trace_path), "--trace-max-elements", "200000",
    ]
    if policy == "fp8":
        result.extend([
            "--fp8-linear", "true", "--fp8-weight-scale-mode", "tensor-amax",
            "--fp8-weight-scale", "0.005", "--fp8-activation-scale-mode", "tensor-amax",
            "--fp8-activation-scale", "0.2", "--fp8-activation-minimum-scale", "0.0001",
        ])
    return result


def selected_trace(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in records if BLOCK.match(row["name"]) or
                row["name"] in ("inference.final_norm", "inference.logits")]
    selected.sort(key=lambda row: (
        int(BLOCK.match(row["name"]).group(1)) if BLOCK.match(row["name"])
        else 1_000_000 if row["name"] == "inference.final_norm" else 1_000_001))
    if not selected or any(row.get("values_truncated") or not row.get("values")
                           for row in selected):
        raise RuntimeError("FP8 layer drift requires complete selected values")
    return selected


def compare_stages(reference: list[dict], actual: list[dict]) -> list[dict]:
    if [row["name"] for row in reference] != [row["name"] for row in actual]:
        raise RuntimeError("FP32/FP8 stage names changed")
    rows = []
    for left, right in zip(reference, actual):
        if left["shape"] != right["shape"]:
            raise RuntimeError("FP32/FP8 stage shape changed")
        rows.append({"name": left["name"], "shape": left["shape"],
                     **difference(left["values"], right["values"])})
    return rows


def main() -> int:
    args = options()
    models = matrix.load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    summaries = []
    for model in models:
        traces = {}
        worker_rows = []
        for policy in ("fp32", "fp8"):
            trace_path = args.output_directory / f"{model['name']}-{policy}-trace.jsonl"
            pre = matrix.require_idle(args.physical_gpu_index,
                args.max_idle_vram_percent, args.max_idle_use_percent,
                f"{model['name']} {policy} layer trace pre")
            completed = subprocess.run(command(args, model, policy, trace_path),
                                       capture_output=True, text=True)
            post = matrix.require_idle(args.physical_gpu_index,
                args.max_idle_vram_percent, args.max_idle_use_percent,
                f"{model['name']} {policy} layer trace post")
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "layer trace worker failed")
            output = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
            if len(output) != 1 or output[0].get("status") != "pass":
                raise RuntimeError("layer trace worker output contract failed")
            traces[policy] = selected_trace(trace_path)
            worker_rows.append({"policy": policy, "pre_run_gpu_state": pre,
                                "post_run_gpu_state": post,
                                "trace_record_count": output[0]["trace_record_count"]})
        stages = compare_stages(traces["fp32"], traces["fp8"])
        maximum = max(stages, key=lambda row: row["relative_l2"])
        summary = {
            "schema_version": 1, "status": "pass",
            "track": "official_fp8_layer_drift", "model": model["name"],
            "revision": model["revision"], "context": args.context,
            "stage_count": len(stages), "maximum_relative_l2_stage": maximum["name"],
            "maximum_relative_l2": maximum["relative_l2"],
            "workers": worker_rows, "stages": stages,
        }
        summaries.append(summary)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps({"model": model["name"], "status": "pass"}), flush=True)
    document = {
        "schema_version": 1, "status": "pass", "track": "official_fp8_layer_drift",
        "models": [row["model"] for row in summaries], "context": args.context,
        "summaries": summaries,
        "boundary": "complete FP32/FP8 stage snapshots; synchronous diagnostic, not performance evidence",
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
