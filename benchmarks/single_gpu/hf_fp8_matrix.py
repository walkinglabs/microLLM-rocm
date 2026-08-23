#!/usr/bin/env python3
"""Official-model FP32/BF16/FP8 inference matrix with complete-logit gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import subprocess
from pathlib import Path


POLICY_ORDERS = (
    ("fp32", "bf16", "fp8"),
    ("bf16", "fp8", "fp32"),
    ("fp8", "fp32", "bf16"),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--contexts", default="8,512")
    parser.add_argument("--models")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fp8-activation-scale", type=float, default=0.025)
    parser.add_argument("--fp8-activation-minimum-scale", type=float, default=0.0001)
    parser.add_argument("--fp8-weight-scale", type=float, default=0.005)
    parser.add_argument("--fp8-weight-scale-mode",
                        choices=("fixed", "tensor-amax", "device-tensor-amax",
                                 "output-channel-amax"),
                        default="fixed")
    parser.add_argument("--fp8-activation-scale-mode",
                        choices=("fixed", "tensor-amax", "ffn-outer-row"),
                        default="fixed")
    parser.add_argument("--fp8-diagnostic-mode",
                        choices=("full", "weight-only", "activation-only",
                                 "both-roundtrip"),
                        default="full")
    parser.add_argument("--fp8-fp32-layers", default="")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    try:
        result.contexts = [int(value) for value in result.contexts.split(",")]
    except ValueError as error:
        parser.error(f"invalid contexts: {error}")
    if not result.manifest.is_file() or not result.binary.is_file() or \
            not result.contexts or any(value <= 0 for value in result.contexts) or \
            result.warmup < 0 or result.steps <= 0 or result.runs <= 0 or \
            not math.isfinite(result.fp8_activation_scale) or \
            result.fp8_activation_scale <= 0 or \
            not math.isfinite(result.fp8_activation_minimum_scale) or \
            result.fp8_activation_minimum_scale <= 0 or \
            not math.isfinite(result.fp8_weight_scale) or \
            result.fp8_weight_scale <= 0:
        parser.error("manifest, binary, contexts, runs or scales are invalid")
    result.models = result.models.split(",") if result.models else None
    return result


def load_models(path: Path, selected: list[str] | None) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models", [])
    by_name = {model.get("name"): model for model in models}
    names = selected or list(by_name)
    if not models or None in by_name or len(by_name) != len(models) or \
            not set(names) <= set(by_name):
        raise RuntimeError("model manifest is invalid")
    return [by_name[name] for name in names]


def gpu_state(index: int) -> dict:
    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmemuse", "--json"],
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("cannot query GPU state")
    card = json.loads(completed.stdout).get(f"card{index}")
    if not isinstance(card, dict):
        raise RuntimeError(f"physical GPU {index} missing")
    return {"physical_gpu_index": index,
            "gpu_use_percent": int(card["GPU use (%)"]),
            "vram_percent": int(card["GPU Memory Allocated (VRAM%)"])}


def require_idle(index: int | None, maximum_vram: int,
                 maximum_use: int, boundary: str) -> dict | None:
    if index is None:
        return None
    state = gpu_state(index)
    if state["vram_percent"] > maximum_vram or \
            state["gpu_use_percent"] > maximum_use:
        raise RuntimeError(f"GPU occupied at {boundary}: {state}")
    return state


def command(args: argparse.Namespace, model: dict, context: int,
            policy: str, logits_path: Path) -> list[str]:
    tokens = model["inference"]["token_ids"]
    prompt = [tokens[index % len(tokens)] for index in range(context)]
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", ",".join(map(str, prompt)),
        "--device", "hip", "--top-k", "1", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(args.steps),
        "--prefill-logits", "last", "--workload", "prefill",
        "--use-cache", "true", "--kv-cache-dtype", "fp32",
        "--logits-output", str(logits_path),
    ]
    if policy == "bf16":
        result.extend(["--bf16-ffn", "true", "--bf16-attention", "true"])
    elif policy == "fp8":
        diagnostic_mode = getattr(args, "fp8_diagnostic_mode", "full")
        result.extend([
            "--fp8-linear", "true",
            "--fp8-activation-scale", str(args.fp8_activation_scale),
            "--fp8-activation-minimum-scale",
            str(args.fp8_activation_minimum_scale),
            "--fp8-weight-scale", str(args.fp8_weight_scale),
            "--fp8-weight-scale-mode", args.fp8_weight_scale_mode,
            "--fp8-activation-scale-mode", args.fp8_activation_scale_mode,
            "--fp8-diagnostic-mode", diagnostic_mode,
        ])
        if args.fp8_fp32_layers:
            result.extend(["--fp8-fp32-layers", args.fp8_fp32_layers])
    return result


def read_float32(path: Path) -> list[float]:
    payload = path.read_bytes()
    if not payload or len(payload) % 4 != 0:
        raise RuntimeError("logit payload is empty or misaligned")
    return list(struct.unpack(f"{len(payload) // 4}f", payload))


def compare_logits(actual: list[float], reference: list[float]) -> dict:
    if len(actual) != len(reference):
        raise RuntimeError("logit vocabulary changed")
    differences = [left - right for left, right in zip(actual, reference)]
    maximum = max(abs(value) for value in differences)
    rms = math.sqrt(sum(value * value for value in differences) / len(differences))
    actual_top = max(range(len(actual)), key=actual.__getitem__)
    reference_top = max(range(len(reference)), key=reference.__getitem__)
    return {"maximum_absolute_error": maximum,
            "root_mean_square_error": rms,
            "top_token": actual_top,
            "reference_top_token": reference_top,
            "top_token_equal": actual_top == reference_top,
            "precision_gate_passed": maximum <= 0.2 and rms <= 0.05 and
                                     actual_top == reference_top}


def experiment_boundary(weight_scale_mode: str,
                        activation_scale_mode: str = "fixed",
                        diagnostic_mode: str = "full") -> str:
    weight_boundary = (
        "per-Tensor weight amax with one-time host scan"
        if weight_scale_mode == "tensor-amax"
        else "device per-Tensor weight amax"
        if weight_scale_mode == "device-tensor-amax"
        else "device per-output-channel weight amax"
        if weight_scale_mode == "output-channel-amax"
        else "static global weight scale")
    activation_boundary = (
        "device per-input-Tensor activation amax"
        if activation_scale_mode == "tensor-amax"
        else "FFN-only outer-row activation scales"
        if activation_scale_mode == "ffn-outer-row"
        else "fixed global activation scale")
    return (
        f"{weight_boundary}; {activation_boundary}; "
        f"diagnostic mode={diagnostic_mode}; "
        "single-representation Linear weights; FP32 Embedding/Norm/tied head; "
        "FP32 logits are the internal oracle; no PyTorch FP8 reference")


def main() -> int:
    args = options()
    models = load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for model in models:
        for context in args.contexts:
            for process_run in range(1, args.runs + 1):
                order = POLICY_ORDERS[(process_run - 1) % len(POLICY_ORDERS)]
                pending = []
                for policy in order:
                    logits_path = args.output_directory / (
                        f".{model['name']}-{context}-{process_run}-{policy}.bin")
                    pre = require_idle(
                        args.physical_gpu_index, args.max_idle_vram_percent,
                        args.max_idle_use_percent,
                        f"{model['name']} T{context} run{process_run} {policy} pre")
                    completed = subprocess.run(
                        command(args, model, context, policy, logits_path),
                        capture_output=True, text=True)
                    post = require_idle(
                        args.physical_gpu_index, args.max_idle_vram_percent,
                        args.max_idle_use_percent,
                        f"{model['name']} T{context} run{process_run} {policy} post")
                    if completed.returncode != 0:
                        raise RuntimeError(completed.stderr.strip() or "FP8 worker failed")
                    output = [json.loads(line) for line in completed.stdout.splitlines()
                              if line.strip()]
                    if len(output) != 1 or output[0].get("status") != "pass":
                        raise RuntimeError("FP8 worker output contract failed")
                    values = read_float32(logits_path)
                    logits_path.unlink()
                    pending.append((policy, output[0], values, pre, post))
                reference = next(values for policy, _, values, _, _ in pending
                                 if policy == "fp32")
                for policy, output, values, pre, post in pending:
                    comparison = compare_logits(values, reference)
                    row = {
                        "schema_version": 1, "status": "pass",
                        "track": "official_fp8_inference_matrix",
                        "model": model["name"], "revision": model["revision"],
                        "context": context, "policy": policy,
                        "process_run": process_run,
                        "policy_order": list(order),
                        "prefill_tokens_per_second": output[
                            "prefill_tokens_per_second"],
                        "resident_weight_bytes": output["resident_weight_bytes"],
                        "fp8_weight_bytes_retained": output.get(
                            "fp8_weight_bytes_retained", 0),
                        "fp8_scale_bytes_retained": output.get(
                            "fp8_scale_bytes_retained", 0),
                        "engine_peak_bytes": output["engine_peak_bytes"],
                        "weight_preparation_ms": output.get(
                            "weight_preparation_ms", 0.0),
                        "converted_tensors": output.get(
                            "fp8_converted_tensors" if policy == "fp8"
                            else "bf16_ffn_converted_tensors", 0),
                        "fp8_linears_covered": output.get(
                            "fp8_linears_covered", 0),
                        "fp8_native_shapes": output.get("fp8_native_shapes", 0),
                        "fp8_software_fallback_shapes": output.get(
                            "fp8_software_fallback_shapes", 0),
                        "fp8_software_fallback_calls": output.get(
                            "fp8_software_fallback_calls", 0),
                        "fp8_outer_row_fallback_calls": output.get(
                            "fp8_outer_row_fallback_calls", 0),
                        "fp8_outer_row_native_status": output.get(
                            "fp8_outer_row_native_status", -1),
                        "fp8_output_column_scale_calls": output.get(
                            "fp8_output_column_scale_calls", 0),
                        "fp8_dynamic_tensor_calls": output.get(
                            "fp8_dynamic_tensor_calls", 0),
                        "fp8_dynamic_row_calls": output.get(
                            "fp8_dynamic_row_calls", 0),
                        "fp8_dynamic_tensor_elements": output.get(
                            "fp8_dynamic_tensor_elements", 0),
                        "fp8_dynamic_row_elements": output.get(
                            "fp8_dynamic_row_elements", 0),
                        "fp8_dynamic_column_calls": output.get(
                            "fp8_dynamic_column_calls", 0),
                        "fp8_dynamic_column_elements": output.get(
                            "fp8_dynamic_column_elements", 0),
                        "fp8_activation_scale": args.fp8_activation_scale,
                        "fp8_activation_minimum_scale":
                            args.fp8_activation_minimum_scale,
                        "fp8_weight_scale": args.fp8_weight_scale,
                        "fp8_weight_scale_mode": args.fp8_weight_scale_mode,
                        "fp8_activation_scale_mode": args.fp8_activation_scale_mode,
                        "fp8_diagnostic_mode": args.fp8_diagnostic_mode,
                        "fp8_fp32_layers": args.fp8_fp32_layers,
                        "fp8_weight_scale_min": output.get(
                            "fp8_weight_scale_min", 0.0),
                        "fp8_weight_scale_max": output.get(
                            "fp8_weight_scale_max", 0.0),
                        "fp8_weight_bytes_scanned": output.get(
                            "fp8_weight_bytes_scanned", 0),
                        "fp8_device_weight_bytes_scanned": output.get(
                            "fp8_device_weight_bytes_scanned", 0),
                        "fp8_device_amax_tensors": output.get(
                            "fp8_device_amax_tensors", 0),
                        "fp8_host_scale_summary_available": output.get(
                            "fp8_host_scale_summary_available", True),
                        "logit_count": len(values),
                        **comparison,
                    }
                    if pre is not None:
                        row["pre_run_gpu_state"] = pre
                        row["post_run_gpu_state"] = post
                    rows.append(row)
                    with raw_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
    aggregates = []
    for model in models:
        for context in args.contexts:
            for policy in ("fp32", "bf16", "fp8"):
                selected = [row for row in rows
                            if row["model"] == model["name"] and
                            row["context"] == context and row["policy"] == policy]
                aggregates.append({
                    "model": model["name"], "context": context, "policy": policy,
                    "successful_runs": len(selected),
                    "prefill_tokens_per_second_p50": statistics.median(
                        row["prefill_tokens_per_second"] for row in selected),
                    "resident_weight_bytes": selected[0]["resident_weight_bytes"],
                    "fp8_weight_bytes_retained": selected[0][
                        "fp8_weight_bytes_retained"],
                    "fp8_scale_bytes_retained": selected[0][
                        "fp8_scale_bytes_retained"],
                    "engine_peak_bytes_p50": statistics.median(
                        row["engine_peak_bytes"] for row in selected),
                    "weight_preparation_ms_p50": statistics.median(
                        row["weight_preparation_ms"] for row in selected),
                    "maximum_absolute_error_max": max(
                        row["maximum_absolute_error"] for row in selected),
                    "root_mean_square_error_max": max(
                        row["root_mean_square_error"] for row in selected),
                    "top_token_equal_all": all(
                        row["top_token_equal"] for row in selected),
                    "precision_gate_passed_all": all(
                        row["precision_gate_passed"] for row in selected),
                    "fp8_native_shapes_max": max(
                        row["fp8_native_shapes"] for row in selected),
                    "fp8_software_fallback_shapes_max": max(
                        row["fp8_software_fallback_shapes"] for row in selected),
                    "fp8_software_fallback_calls_p50": statistics.median(
                        row["fp8_software_fallback_calls"] for row in selected),
                    "fp8_outer_row_fallback_calls_p50": statistics.median(
                        row["fp8_outer_row_fallback_calls"] for row in selected),
                    "fp8_outer_row_native_statuses": sorted(set(
                        row["fp8_outer_row_native_status"] for row in selected)),
                    "fp8_output_column_scale_calls_p50": statistics.median(
                        row["fp8_output_column_scale_calls"] for row in selected),
                    "fp8_dynamic_tensor_calls_p50": statistics.median(
                        row["fp8_dynamic_tensor_calls"] for row in selected),
                    "fp8_dynamic_row_calls_p50": statistics.median(
                        row["fp8_dynamic_row_calls"] for row in selected),
                    "fp8_dynamic_column_calls_p50": statistics.median(
                        row["fp8_dynamic_column_calls"] for row in selected),
                })
    accuracy_failures = [row for row in aggregates
                         if row["policy"] != "fp32" and
                         not row["precision_gate_passed_all"]]
    summary = {
        "schema_version": 1,
        "track": "official_fp8_inference_matrix",
        "status": "complete_with_recorded_accuracy_failures"
        if accuracy_failures else "pass",
        "execution_status": "pass",
        "models": [model["name"] for model in models],
        "contexts": args.contexts,
        "policies": ["fp32", "bf16", "fp8"],
        "runs": args.runs, "warmup": args.warmup, "steps": args.steps,
        "fp8_activation_scale": args.fp8_activation_scale,
        "fp8_activation_minimum_scale": args.fp8_activation_minimum_scale,
        "fp8_weight_scale": args.fp8_weight_scale,
        "fp8_weight_scale_mode": args.fp8_weight_scale_mode,
        "fp8_activation_scale_mode": args.fp8_activation_scale_mode,
        "fp8_diagnostic_mode": args.fp8_diagnostic_mode,
        "fp8_fp32_layers": args.fp8_fp32_layers,
        "rows": rows, "aggregates": aggregates,
        "accuracy_failure_count": len(accuracy_failures),
        "accuracy_failures": accuracy_failures,
        "boundary": experiment_boundary(
            args.fp8_weight_scale_mode, args.fp8_activation_scale_mode,
            args.fp8_diagnostic_mode),
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
