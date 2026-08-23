#!/usr/bin/env python3
"""Compare native FP8 GEMM with identical FP8 roundtrips and FP32 GEMM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
from pathlib import Path


MATRIX_SPEC = importlib.util.spec_from_file_location(
    "_microllm_hf_fp8_matrix", Path(__file__).with_name("hf_fp8_matrix.py"))
MATRIX = importlib.util.module_from_spec(MATRIX_SPEC)
assert MATRIX_SPEC.loader is not None
MATRIX_SPEC.loader.exec_module(MATRIX)

POLICY_ORDERS = (
    ("fp32", "full", "both-roundtrip"),
    ("full", "both-roundtrip", "fp32"),
    ("both-roundtrip", "fp32", "full"),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--contexts", default="8,512")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--fp8-activation-minimum-scale", type=float,
                        default=0.0001)
    parser.add_argument("--fp8-weight-scale", type=float, default=0.005)
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
            result.runs <= 0 or \
            not math.isfinite(result.fp8_activation_minimum_scale) or \
            result.fp8_activation_minimum_scale <= 0 or \
            not math.isfinite(result.fp8_weight_scale) or \
            result.fp8_weight_scale <= 0:
        parser.error("manifest, binary, contexts, runs or scales are invalid")
    result.models = result.models.split(",") if result.models else None
    return result


def worker_command(args: argparse.Namespace, model: dict, context: int,
                   policy: str, logits_path: Path) -> list[str]:
    command_args = argparse.Namespace(
        binary=args.binary,
        warmup=0,
        steps=1,
        fp8_activation_scale=0.2,
        fp8_activation_minimum_scale=args.fp8_activation_minimum_scale,
        fp8_weight_scale=args.fp8_weight_scale,
        fp8_weight_scale_mode="device-tensor-amax",
        fp8_activation_scale_mode="tensor-amax",
        fp8_diagnostic_mode=policy if policy != "fp32" else "full",
        fp8_fp32_layers="",
    )
    return MATRIX.command(command_args, model, context,
                          "fp32" if policy == "fp32" else "fp8",
                          logits_path)


def comparison_bundle(full: list[float], roundtrip: list[float],
                      reference: list[float]) -> dict:
    return {
        "full_vs_fp32": MATRIX.compare_logits(full, reference),
        "both_roundtrip_vs_fp32": MATRIX.compare_logits(roundtrip, reference),
        "full_vs_both_roundtrip": MATRIX.compare_logits(full, roundtrip),
    }


def main() -> int:
    args = options()
    models = MATRIX.load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    pairs_path = args.output_directory / "pairs.jsonl"
    raw_path.write_text("", encoding="utf-8")
    pairs_path.write_text("", encoding="utf-8")
    workers = []
    pairs = []
    for model in models:
        for context in args.contexts:
            for process_run in range(1, args.runs + 1):
                order = POLICY_ORDERS[(process_run - 1) % len(POLICY_ORDERS)]
                values_by_policy = {}
                outputs_by_policy = {}
                for policy in order:
                    logits_path = args.output_directory / (
                        f".{model['name']}-{context}-{process_run}-{policy}.bin")
                    pre = MATRIX.require_idle(
                        args.physical_gpu_index, args.max_idle_vram_percent,
                        args.max_idle_use_percent,
                        f"{model['name']} T{context} run{process_run} {policy} pre")
                    completed = subprocess.run(
                        worker_command(args, model, context, policy, logits_path),
                        capture_output=True, text=True)
                    post = MATRIX.require_idle(
                        args.physical_gpu_index, args.max_idle_vram_percent,
                        args.max_idle_use_percent,
                        f"{model['name']} T{context} run{process_run} {policy} post")
                    if completed.returncode != 0:
                        raise RuntimeError(completed.stderr.strip() or
                                           "native-roundtrip worker failed")
                    output_rows = [json.loads(line)
                                   for line in completed.stdout.splitlines()
                                   if line.strip()]
                    if len(output_rows) != 1 or \
                            output_rows[0].get("status") != "pass":
                        raise RuntimeError(
                            "native-roundtrip worker output contract failed")
                    values = MATRIX.read_float32(logits_path)
                    logits_path.unlink()
                    output = output_rows[0]
                    if policy != "fp32" and \
                            output.get("fp8_diagnostic_mode") != policy:
                        raise RuntimeError("diagnostic worker mode changed")
                    if policy == "both-roundtrip" and (
                            output.get("fp8_native_shapes", 0) != 0 or
                            output.get("fp8_software_fallback_calls", 0) != 0):
                        raise RuntimeError(
                            "both-roundtrip unexpectedly dispatched FP8 GEMM")
                    row = {
                        "schema_version": 1,
                        "status": "pass",
                        "record_type": "fp8_native_roundtrip_worker",
                        "model": model["name"],
                        "revision": model["revision"],
                        "context": context,
                        "process_run": process_run,
                        "policy": policy,
                        "policy_order": list(order),
                        "logit_count": len(values),
                        "fp8_linears_covered": output.get(
                            "fp8_linears_covered", 0),
                        "fp8_converted_tensors": output.get(
                            "fp8_converted_tensors", 0),
                        "fp8_dynamic_tensor_calls": output.get(
                            "fp8_dynamic_tensor_calls", 0),
                        "fp8_native_shapes": output.get("fp8_native_shapes", 0),
                        "fp8_software_fallback_calls": output.get(
                            "fp8_software_fallback_calls", 0),
                        "compute_dtype": output.get("compute_dtype", ""),
                        "throughput_is_performance_evidence": False,
                    }
                    if pre is not None:
                        row["pre_run_gpu_state"] = pre
                        row["post_run_gpu_state"] = post
                    workers.append(row)
                    values_by_policy[policy] = values
                    outputs_by_policy[policy] = output
                    with raw_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
                comparisons = comparison_bundle(
                    values_by_policy["full"],
                    values_by_policy["both-roundtrip"],
                    values_by_policy["fp32"])
                pair = {
                    "schema_version": 1,
                    "status": "pass",
                    "record_type": "fp8_native_roundtrip_comparison",
                    "model": model["name"],
                    "revision": model["revision"],
                    "context": context,
                    "process_run": process_run,
                    "logit_count": len(values_by_policy["fp32"]),
                    "same_weight_conversion_count":
                        outputs_by_policy["full"].get("fp8_converted_tensors") ==
                        outputs_by_policy["both-roundtrip"].get(
                            "fp8_converted_tensors"),
                    "same_dynamic_activation_count":
                        outputs_by_policy["full"].get(
                            "fp8_dynamic_tensor_calls") ==
                        outputs_by_policy["both-roundtrip"].get(
                            "fp8_dynamic_tensor_calls"),
                    "comparisons": comparisons,
                    "throughput_is_performance_evidence": False,
                }
                pairs.append(pair)
                with pairs_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(pair, sort_keys=True) + "\n")
                print(json.dumps(pair, sort_keys=True), flush=True)
    failures = [pair for pair in pairs
                if not pair["comparisons"][
                    "full_vs_both_roundtrip"]["precision_gate_passed"]]
    summary = {
        "schema_version": 1,
        "status": "complete_with_recorded_native_roundtrip_differences"
        if failures else "pass",
        "execution_status": "pass",
        "track": "fp8_native_vs_both_roundtrip",
        "models": [model["name"] for model in models],
        "contexts": args.contexts,
        "runs": args.runs,
        "warmup": 0,
        "steps": 1,
        "worker_rows": len(workers),
        "comparison_rows": len(pairs),
        "native_roundtrip_failure_count": len(failures),
        "throughput_is_performance_evidence": False,
        "boundary": (
            "identical device Tensor-amax weights and activations; full uses native FP8 GEMM; "
            "both-roundtrip dequantizes both operands and uses FP32 GEMM; complete logits only"),
        "workers": workers,
        "comparisons": pairs,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
