#!/usr/bin/env python3
"""Numerical-only clipped activation fraction pilot with one FP32 oracle per case."""

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


def parse_fractions(text: str) -> list[float]:
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as error:
        raise ValueError(f"invalid fraction: {error}") from error
    if not values or any(not math.isfinite(value) or not 0 < value <= 1
                         for value in values) or len(set(values)) != len(values) or \
            1.0 not in values:
        raise ValueError("fractions must be unique values in (0,1] and include 1.0")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models")
    parser.add_argument("--contexts", default="8,512")
    parser.add_argument("--fractions", default="1,0.75,0.5,0.25")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    args = parser.parse_args()
    try:
        args.contexts = [int(value) for value in args.contexts.split(",")]
        args.fractions = parse_fractions(args.fractions)
    except ValueError as error:
        parser.error(str(error))
    if not args.manifest.is_file() or not args.binary.is_file() or \
            not args.contexts or any(value <= 0 for value in args.contexts):
        parser.error("manifest, binary or contexts are invalid")
    args.models = args.models.split(",") if args.models else None
    return args


def worker_command(args: argparse.Namespace, model: dict, context: int,
                   fraction: float | None, logits_path: Path) -> list[str]:
    command_args = argparse.Namespace(
        binary=args.binary,
        warmup=0,
        steps=1,
        fp8_activation_scale=0.2,
        fp8_activation_minimum_scale=0.0001,
        fp8_activation_amax_fraction=1.0 if fraction is None else fraction,
        fp8_weight_scale=0.0001,
        fp8_weight_scale_mode="output-channel-amax",
        fp8_weight_scale_scope="attention-output-only",
        fp8_activation_scale_mode="tensor-amax",
        fp8_diagnostic_mode="full",
        fp8_fp32_layers="",
    )
    return MATRIX.command(command_args, model, context,
                          "fp32" if fraction is None else "fp8", logits_path)


def select_fraction(aggregates: list[dict]) -> dict:
    by_fraction = {row["fraction"]: row for row in aggregates}
    if 1.0 not in by_fraction:
        raise ValueError("fraction aggregates require the 1.0 control")
    eligible = [row for row in aggregates if row["top_token_equal_all"]]
    if not eligible:
        return {"selected_fraction": None,
                "reason": "every fraction changes at least one top token"}
    best = min(eligible, key=lambda row: (
        row["worst_normalized_rms"], row["worst_normalized_max"],
        -row["fraction"]))
    control = by_fraction[1.0]
    if best["fraction"] != 1.0 and \
            best["worst_normalized_rms"] < control["worst_normalized_rms"]:
        return {"selected_fraction": best["fraction"],
                "reason": "strictly lower worst normalized RMS with stable top tokens"}
    return {"selected_fraction": 1.0,
            "reason": "no clipped candidate strictly improves worst normalized RMS"}


def main() -> int:
    args = options()
    models = MATRIX.load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for model in models:
        for context in args.contexts:
            values_by_fraction = {}
            for fraction in [None, *args.fractions]:
                policy_name = "fp32" if fraction is None else f"fraction-{fraction:g}"
                logits_path = args.output_directory / (
                    f".{model['name']}-{context}-{policy_name}.bin")
                pre = MATRIX.require_idle(
                    args.physical_gpu_index, args.max_idle_vram_percent,
                    args.max_idle_use_percent,
                    f"{model['name']} T{context} {policy_name} pre")
                completed = subprocess.run(
                    worker_command(args, model, context, fraction, logits_path),
                    capture_output=True, text=True)
                post = MATRIX.require_idle(
                    args.physical_gpu_index, args.max_idle_vram_percent,
                    args.max_idle_use_percent,
                    f"{model['name']} T{context} {policy_name} post")
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or
                                       "fraction pilot worker failed")
                output_rows = [json.loads(line)
                               for line in completed.stdout.splitlines()
                               if line.strip()]
                if len(output_rows) != 1 or output_rows[0].get("status") != "pass":
                    raise RuntimeError("fraction pilot worker output contract failed")
                output = output_rows[0]
                values = MATRIX.read_float32(logits_path)
                logits_path.unlink()
                if fraction is not None:
                    expected_clipped = 0 if fraction == 1.0 else \
                        output.get("fp8_dynamic_tensor_calls", 0)
                    if output.get("fp8_dynamic_clipped_tensor_calls", 0) != \
                            expected_clipped:
                        raise RuntimeError("clipped activation call count changed")
                row = {
                    "schema_version": 1,
                    "status": "pass",
                    "record_type": "fp8_activation_fraction_pilot_worker",
                    "model": model["name"],
                    "revision": model["revision"],
                    "context": context,
                    "policy": policy_name,
                    "fraction": fraction,
                    "logit_count": len(values),
                    "fp8_dynamic_tensor_calls": output.get(
                        "fp8_dynamic_tensor_calls", 0),
                    "fp8_dynamic_clipped_tensor_calls": output.get(
                        "fp8_dynamic_clipped_tensor_calls", 0),
                    "fp8_output_column_scale_calls": output.get(
                        "fp8_output_column_scale_calls", 0),
                    "throughput_is_performance_evidence": False,
                }
                if pre is not None:
                    row["pre_run_gpu_state"] = pre
                    row["post_run_gpu_state"] = post
                rows.append(row)
                values_by_fraction[fraction] = values
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
            reference = values_by_fraction[None]
            for fraction in args.fractions:
                comparison = MATRIX.compare_logits(
                    values_by_fraction[fraction], reference)
                comparison_row = {
                    "schema_version": 1,
                    "status": "pass",
                    "record_type": "fp8_activation_fraction_comparison",
                    "model": model["name"],
                    "revision": model["revision"],
                    "context": context,
                    "fraction": fraction,
                    "logit_count": len(reference),
                    **comparison,
                }
                rows.append(comparison_row)
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(comparison_row, sort_keys=True) + "\n")
                print(json.dumps(comparison_row, sort_keys=True), flush=True)
    comparisons = [row for row in rows
                   if row["record_type"] == "fp8_activation_fraction_comparison"]
    aggregates = []
    for fraction in args.fractions:
        selected = [row for row in comparisons if row["fraction"] == fraction]
        aggregates.append({
            "fraction": fraction,
            "cases": len(selected),
            "worst_normalized_rms": max(
                row["root_mean_square_error"] / 0.05 for row in selected),
            "worst_normalized_max": max(
                row["maximum_absolute_error"] / 0.2 for row in selected),
            "top_token_equal_all": all(row["top_token_equal"] for row in selected),
            "precision_gate_pass_count": sum(
                row["precision_gate_passed"] for row in selected),
        })
    selection = select_fraction(aggregates)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "execution_status": "pass",
        "track": "fp8_clipped_activation_fraction_pilot",
        "models": [model["name"] for model in models],
        "contexts": args.contexts,
        "fractions": args.fractions,
        "worker_rows": sum(row["record_type"].endswith("worker") for row in rows),
        "comparison_rows": len(comparisons),
        "throughput_is_performance_evidence": False,
        "selection": selection,
        "aggregates": aggregates,
        "comparisons": comparisons,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
