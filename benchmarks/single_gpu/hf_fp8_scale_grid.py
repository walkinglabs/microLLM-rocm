#!/usr/bin/env python3
"""Screen fixed FP8 scale pairs against complete official-model logits."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fp8_matrix as matrix  # noqa: E402


def parse_positive_grid(text: str, name: str) -> list[float]:
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {name}: {error}") from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive finite values")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--activation-scales", default="0.00625,0.0125,0.025,0.05")
    parser.add_argument("--weight-scales", default="0.00125,0.0025,0.005,0.01")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    try:
        result.activation_scales = parse_positive_grid(
            result.activation_scales, "activation scales")
        result.weight_scales = parse_positive_grid(
            result.weight_scales, "weight scales")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if not result.manifest.is_file() or not result.binary.is_file() or \
            result.context <= 0 or result.warmup < 0 or result.steps <= 0:
        parser.error("manifest, binary, context, warmup or steps are invalid")
    result.models = result.models.split(",") if result.models else None
    return result


def command(args: argparse.Namespace, model: dict, policy: str,
            logits_path: Path, activation_scale: float,
            weight_scale: float) -> list[str]:
    proxy = argparse.Namespace(
        binary=args.binary, warmup=args.warmup, steps=args.steps,
        fp8_activation_scale=activation_scale,
        fp8_weight_scale=weight_scale)
    return matrix.command(proxy, model, args.context, policy, logits_path)


def run_worker(args: argparse.Namespace, model: dict, policy: str,
               activation_scale: float, weight_scale: float,
               label: str) -> tuple[dict, list[float], dict | None, dict | None]:
    logits_path = args.output_directory / f".{model['name']}-{label}.bin"
    pre = matrix.require_idle(
        args.physical_gpu_index, args.max_idle_vram_percent,
        args.max_idle_use_percent, f"{model['name']} {label} pre")
    completed = subprocess.run(
        command(args, model, policy, logits_path, activation_scale, weight_scale),
        capture_output=True, text=True)
    post = matrix.require_idle(
        args.physical_gpu_index, args.max_idle_vram_percent,
        args.max_idle_use_percent, f"{model['name']} {label} post")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "FP8 scale worker failed")
    output = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(output) != 1 or output[0].get("status") != "pass":
        raise RuntimeError("FP8 scale worker output contract failed")
    values = matrix.read_float32(logits_path)
    logits_path.unlink()
    return output[0], values, pre, post


def make_row(model: dict, args: argparse.Namespace, output: dict,
             values: list[float], reference: list[float], policy: str,
             activation_scale: float, weight_scale: float,
             pre: dict | None, post: dict | None) -> dict:
    row = {
        "schema_version": 1, "status": "pass",
        "track": "official_fp8_scale_grid", "model": model["name"],
        "revision": model["revision"], "context": args.context,
        "policy": policy, "fp8_activation_scale": activation_scale,
        "fp8_weight_scale": weight_scale,
        "prefill_tokens_per_second": output["prefill_tokens_per_second"],
        "resident_weight_bytes": output["resident_weight_bytes"],
        "engine_peak_bytes": output["engine_peak_bytes"],
        "converted_tensors": output.get("fp8_converted_tensors", 0),
        "fp8_native_shapes": output.get("fp8_native_shapes", 0),
        "fp8_software_fallback_shapes": output.get(
            "fp8_software_fallback_shapes", 0),
        "fp8_software_fallback_calls": output.get(
            "fp8_software_fallback_calls", 0),
        "logit_count": len(values),
        **matrix.compare_logits(values, reference),
    }
    if pre is not None:
        row["pre_run_gpu_state"] = pre
        row["post_run_gpu_state"] = post
    return row


def select_best(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: (
        not row["precision_gate_passed"],
        not row["top_token_equal"],
        row["root_mean_square_error"],
        row["maximum_absolute_error"],
        -row["prefill_tokens_per_second"],
    ))


def main() -> int:
    args = options()
    models = matrix.load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for model in models:
        reference_output, reference, pre, post = run_worker(
            args, model, "fp32", 1.0, 1.0, "fp32")
        reference_row = make_row(
            model, args, reference_output, reference, reference, "fp32",
            1.0, 1.0, pre, post)
        rows.append(reference_row)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(reference_row, sort_keys=True) + "\n")
        print(json.dumps(reference_row, sort_keys=True), flush=True)
        for activation_scale in args.activation_scales:
            for weight_scale in args.weight_scales:
                label = f"a{activation_scale:g}-w{weight_scale:g}"
                output, values, pre, post = run_worker(
                    args, model, "fp8", activation_scale, weight_scale, label)
                row = make_row(
                    model, args, output, values, reference, "fp8",
                    activation_scale, weight_scale, pre, post)
                rows.append(row)
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
    candidates = [row for row in rows if row["policy"] == "fp8"]
    by_model = {}
    for model in models:
        selected = [row for row in candidates if row["model"] == model["name"]]
        best = select_best(selected)
        by_model[model["name"]] = {
            "candidate_count": len(selected),
            "precision_gate_pass_count": sum(
                row["precision_gate_passed"] for row in selected),
            "top_token_equal_count": sum(row["top_token_equal"] for row in selected),
            "best_candidate": best,
            "throughput_p50": statistics.median(
                row["prefill_tokens_per_second"] for row in selected),
        }
    gate_pass_count = sum(row["precision_gate_passed"] for row in candidates)
    summary = {
        "schema_version": 1, "track": "official_fp8_scale_grid",
        "status": "pass" if gate_pass_count else "complete_no_passing_scale",
        "execution_status": "pass", "models": [model["name"] for model in models],
        "context": args.context, "warmup": args.warmup, "steps": args.steps,
        "activation_scales": args.activation_scales,
        "weight_scales": args.weight_scales,
        "selection_rule": (
            "gate, top-token equality, RMS, maximum error, then throughput; "
            "the grid is fixed before execution"),
        "candidate_count": len(candidates),
        "precision_gate_pass_count": gate_pass_count,
        "by_model": by_model, "rows": rows,
        "boundary": (
            "one FP32 reference process and one fresh process per fixed global "
            "scale pair; complete last-token vocabulary logits; screening only"),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
