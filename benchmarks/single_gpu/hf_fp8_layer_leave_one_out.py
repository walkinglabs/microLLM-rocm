#!/usr/bin/env python3
"""Screen one-FP32-block counterfactuals against complete official-model logits."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_fp8_matrix as matrix  # noqa: E402


def model_layer_count(config_path: Path) -> int:
    document = json.loads(config_path.read_text(encoding="utf-8"))
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = document.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise RuntimeError(f"cannot determine positive layer count from {config_path}")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--fp8-activation-scale", type=float, default=0.2)
    parser.add_argument("--fp8-activation-minimum-scale", type=float,
                        default=0.0001)
    parser.add_argument("--fp8-weight-scale", type=float, default=0.005)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file() or \
            result.context <= 0 or result.warmup < 0 or result.steps <= 0 or \
            not math.isfinite(result.fp8_activation_scale) or \
            result.fp8_activation_scale <= 0 or \
            not math.isfinite(result.fp8_activation_minimum_scale) or \
            result.fp8_activation_minimum_scale <= 0 or \
            not math.isfinite(result.fp8_weight_scale) or \
            result.fp8_weight_scale <= 0:
        parser.error("manifest, binary, context, steps or scales are invalid")
    result.models = result.models.split(",") if result.models else None
    return result


def command(args: argparse.Namespace, model: dict, policy: str,
            logits_path: Path, layer: int | None) -> list[str]:
    proxy = argparse.Namespace(
        binary=args.binary, warmup=args.warmup, steps=args.steps,
        fp8_activation_scale=args.fp8_activation_scale,
        fp8_activation_minimum_scale=args.fp8_activation_minimum_scale,
        fp8_weight_scale=args.fp8_weight_scale,
        fp8_weight_scale_mode="output-channel-amax",
        fp8_weight_scale_scope="attention-output-only",
        fp8_activation_scale_mode="tensor-amax",
        fp8_diagnostic_mode="full",
        fp8_fp32_layers="" if layer is None else str(layer))
    return matrix.command(proxy, model, args.context, policy, logits_path)


def run_worker(args: argparse.Namespace, model: dict, policy: str,
               variant: str, layer: int | None) -> tuple[
                   dict, list[float], dict | None, dict | None]:
    layer_name = "none" if layer is None else str(layer)
    logits_path = args.output_directory / (
        f".{model['name']}-{args.context}-{variant}-{layer_name}.bin")
    boundary = f"{model['name']} T{args.context} {variant} layer={layer_name}"
    pre = matrix.require_idle(
        args.physical_gpu_index, args.max_idle_vram_percent,
        args.max_idle_use_percent, f"{boundary} pre")
    completed = subprocess.run(
        command(args, model, policy, logits_path, layer),
        capture_output=True, text=True)
    post = matrix.require_idle(
        args.physical_gpu_index, args.max_idle_vram_percent,
        args.max_idle_use_percent, f"{boundary} post")
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or "FP8 layer-search worker failed")
    output = [json.loads(line) for line in completed.stdout.splitlines()
              if line.strip()]
    if len(output) != 1 or output[0].get("status") != "pass":
        raise RuntimeError("FP8 layer-search worker output contract failed")
    expected_layers = "" if layer is None else str(layer)
    if policy == "fp8" and output[0].get("fp8_fp32_layers") != expected_layers:
        raise RuntimeError("FP8 layer-search routing identity changed")
    values = matrix.read_float32(logits_path)
    logits_path.unlink()
    return output[0], values, pre, post


def make_row(model: dict, args: argparse.Namespace, output: dict,
             values: list[float], reference: list[float], variant: str,
             layer: int | None, pre: dict | None, post: dict | None) -> dict:
    row = {
        "schema_version": 1, "status": "pass",
        "track": "official_fp8_layer_leave_one_out",
        "model": model["name"], "revision": model["revision"],
        "context": args.context, "variant": variant, "fp32_layer": layer,
        "prefill_tokens_per_second": output["prefill_tokens_per_second"],
        "resident_weight_bytes": output["resident_weight_bytes"],
        "engine_peak_bytes": output["engine_peak_bytes"],
        "fp8_linears_covered": output.get("fp8_linears_covered", 0),
        "fp8_native_shapes": output.get("fp8_native_shapes", 0),
        "fp8_software_fallback_shapes": output.get(
            "fp8_software_fallback_shapes", 0),
        "fp8_software_fallback_calls": output.get(
            "fp8_software_fallback_calls", 0),
        "fp8_output_column_scale_calls": output.get(
            "fp8_output_column_scale_calls", 0),
        "fp8_dynamic_tensor_calls": output.get("fp8_dynamic_tensor_calls", 0),
        "logit_count": len(values),
        **matrix.compare_logits(values, reference),
    }
    if pre is not None:
        row["pre_run_gpu_state"] = pre
        row["post_run_gpu_state"] = post
    return row


def add_baseline_delta(row: dict, baseline: dict) -> dict:
    result = dict(row)
    maximum = row["maximum_absolute_error"]
    rms = row["root_mean_square_error"]
    baseline_maximum = baseline["maximum_absolute_error"]
    baseline_rms = baseline["root_mean_square_error"]
    if baseline_maximum <= 0 or baseline_rms <= 0:
        raise RuntimeError("FP8 baseline errors must be positive")
    result["maximum_over_baseline"] = maximum / baseline_maximum
    result["rms_over_baseline"] = rms / baseline_rms
    result["maximum_not_worse"] = maximum <= baseline_maximum
    result["rms_not_worse"] = rms <= baseline_rms
    return result


def rank_candidates(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (
        not row["precision_gate_passed"],
        not row["top_token_equal"],
        row["root_mean_square_error"],
        row["maximum_absolute_error"],
        row["fp32_layer"],
    ))


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def main() -> int:
    args = options()
    models = matrix.load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    all_rows = []
    by_model = {}
    for model in models:
        layers = model_layer_count(Path(model["config"]))
        reference_output, reference, pre, post = run_worker(
            args, model, "fp32", "fp32", None)
        reference_row = make_row(
            model, args, reference_output, reference, reference,
            "fp32", None, pre, post)
        append_row(raw_path, reference_row)
        all_rows.append(reference_row)

        baseline_output, baseline_values, pre, post = run_worker(
            args, model, "fp8", "baseline", None)
        baseline = make_row(
            model, args, baseline_output, baseline_values, reference,
            "baseline", None, pre, post)
        append_row(raw_path, baseline)
        all_rows.append(baseline)

        candidates = []
        for layer in range(layers):
            output, values, pre, post = run_worker(
                args, model, "fp8", "leave_one_out", layer)
            row = add_baseline_delta(make_row(
                model, args, output, values, reference,
                "leave_one_out", layer, pre, post), baseline)
            append_row(raw_path, row)
            all_rows.append(row)
            candidates.append(row)
        ranked = rank_candidates(candidates)
        by_model[model["name"]] = {
            "layer_count": layers,
            "candidate_count": len(candidates),
            "baseline": baseline,
            "precision_gate_pass_count": sum(
                row["precision_gate_passed"] for row in candidates),
            "both_metrics_non_worse_count": sum(
                row["maximum_not_worse"] and row["rms_not_worse"]
                for row in candidates),
            "best_candidate": ranked[0],
            "ranked_candidates": ranked,
        }
    summary = {
        "schema_version": 1,
        "track": "official_fp8_layer_leave_one_out",
        "status": "pass",
        "execution_status": "pass",
        "models": [model["name"] for model in models],
        "context": args.context,
        "warmup": args.warmup,
        "steps": args.steps,
        "fp8_activation_scale": args.fp8_activation_scale,
        "fp8_activation_minimum_scale": args.fp8_activation_minimum_scale,
        "fp8_weight_scale": args.fp8_weight_scale,
        "candidate_count": sum(
            value["candidate_count"] for value in by_model.values()),
        "rows": all_rows,
        "by_model": by_model,
        "selection_rule": (
            "precision gate, top-token equality, RMS, maximum error, layer; "
            "screening does not select a production policy"),
        "boundary": (
            "one deterministic FP32 oracle, one retained FP8 baseline and one "
            "fresh process per single-FP32-block counterfactual; complete "
            "last-token vocabulary logits; throughput is diagnostic only"),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
