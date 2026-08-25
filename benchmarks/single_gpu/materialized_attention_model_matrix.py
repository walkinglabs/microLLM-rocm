#!/usr/bin/env python3
"""Expand exact-order materialized Attention across official model shapes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def names(value: str) -> list[str]:
    result = [item for item in value.split(",") if item]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("names must be unique and non-empty")
    return result


def positive_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result) or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--models", type=names,
        default=["qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"])
    parser.add_argument("--contexts", type=positive_ints, default=[512, 2048])
    parser.add_argument("--batches", type=positive_ints, default=[1, 2])
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--cache-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--minimum-sequence", type=int, default=512)
    parser.add_argument(
        "--candidate-policy", choices=("materialized", "auto"),
        default="materialized")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    result = parser.parse_args()
    if (result.decode_tokens <= 0 or result.minimum_sequence <= 0 or
            result.warmup < 0 or result.steps <= 0 or result.runs <= 0 or
            result.timeout_seconds <= 0):
        parser.error("matrix measurement counts are invalid")
    for path in (result.comparison_runner, result.manifest, result.binary):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def run_case(args: argparse.Namespace, model: str, context: int,
             batch: int, output: Path) -> dict:
    completed = subprocess.run([
        sys.executable, str(args.comparison_runner),
        "--manifest", str(args.manifest), "--binary", str(args.binary),
        "--output-directory", str(output), "--model", model,
        "--candidate-policy", args.candidate_policy,
        "--context", str(context), "--batch", str(batch),
        "--decode-tokens", str(args.decode_tokens),
        "--cache-dtype", args.cache_dtype,
        "--minimum-sequence", str(args.minimum_sequence),
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--runs", str(args.runs), "--maximum-logit-error", "0",
        "--maximum-logit-rms", "0",
        "--timeout-seconds", str(args.timeout_seconds),
    ], text=True, capture_output=True, timeout=args.timeout_seconds * 2)
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"{model} T{context} B{batch} emitted no summary: "
            f"{completed.stderr.strip() or completed.stdout.strip()}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("model") != model or summary.get("context") != context or \
            summary.get("batch") != batch or \
            summary.get("candidate_policy") != args.candidate_policy:
        raise ValueError("child comparison summary identity changed")
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"{model} T{context} B{batch} runner crashed: {completed.stderr}")
    return {
        "model": model,
        "revision": summary.get("revision"),
        "context": context,
        "batch": batch,
        "decode_tokens": args.decode_tokens,
        "cache_dtype": args.cache_dtype,
        "candidate_policy": args.candidate_policy,
        "runs_per_policy": args.runs,
        "child_status": summary.get("status"),
        "accuracy_gate_passed": summary.get("accuracy_gate_passed") is True,
        "performance_gate_passed": summary.get("performance_gate_passed") is True,
        "all_generated_tokens_equal": summary.get("all_generated_tokens_equal") is True,
        "maximum_logit_error": summary.get("maximum_logit_error"),
        "maximum_logit_rms_error": summary.get("maximum_logit_rms_error"),
        "current_throughput_tokens_per_second":
            summary.get("median_current_throughput_tokens_per_second"),
        "materialized_throughput_tokens_per_second":
            summary.get("median_split_throughput_tokens_per_second"),
        "throughput_speedup": summary.get("median_throughput_speedup"),
        "paired_speedups": summary.get("paired_speedups"),
        "leave_one_pair_out_speedups": summary.get("leave_one_pair_out_speedups"),
        "peak_bytes_delta": summary.get("median_peak_bytes_delta"),
        "allocation_calls_delta": summary.get("median_allocation_calls_delta"),
        "backend_allocation_calls_delta":
            summary.get("median_backend_allocation_calls_delta"),
        "relative_directory": output.name,
    }


def policy_boundary(cases: list[dict], models: list[str],
                    contexts: list[int], batches: list[int]) -> int:
    ordered = sorted(contexts)
    for minimum in ordered:
        relevant = [case for case in cases if case["context"] >= minimum]
        expected = (len(models) * len([value for value in ordered if value >= minimum]) *
                    len(batches))
        if len(relevant) == expected and all(
                case["accuracy_gate_passed"] and
                case["performance_gate_passed"] for case in relevant):
            return minimum
    return 0


def svg(cases: list[dict]) -> str:
    width, height = 1250, 125 + len(cases) * 64
    maximum = max(1.1, max(float(case["throughput_speedup"]) for case in cases) * 1.08)
    left, plot = 390, 690
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:700}'
        '.muted{fill:#94a3b8}</style>',
        '<text x="28" y="34" class="title">Materialized-score official model boundary</text>',
        '<text x="28" y="58" class="small muted">paired fresh-process throughput · green requires exact logits and performance gate</text>',
    ]
    gate_x = left + plot * 1.05 / maximum
    parts.append(f'<line x1="{gate_x:.2f}" y1="75" x2="{gate_x:.2f}" '
                 f'y2="{height - 25}" stroke="#22c55e" stroke-dasharray="5 5"/>')
    for index, case in enumerate(cases):
        y = 88 + index * 64
        label = f'{case["model"]} · T{case["context"]} B{case["batch"]}'
        value = float(case["throughput_speedup"])
        bar = plot * value / maximum
        passed = case["accuracy_gate_passed"] and case["performance_gate_passed"]
        color = "#22c55e" if passed else "#ef4444"
        parts.extend([
            f'<text x="28" y="{y + 21}" class="label">{label}</text>',
            f'<rect x="{left}" y="{y}" width="{bar:.2f}" height="24" rx="4" fill="{color}"/>',
            f'<text x="{left + bar + 10:.2f}" y="{y + 18}" class="label">{value:.3f}x</text>',
            f'<text x="1125" y="{y + 18}" class="small">logit {float(case["maximum_logit_error"]):.1e}</text>',
        ])
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = arguments()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for model in args.models:
        for context in args.contexts:
            for batch in args.batches:
                name = f"{model}-t{context}-b{batch}"
                case = run_case(
                    args, model, context, batch,
                    args.output_directory / name)
                cases.append(case)
                print(json.dumps(case, sort_keys=True), flush=True)
    boundary = policy_boundary(cases, args.models, args.contexts, args.batches)
    summary = {
        "schema_version": 1,
        "record_type": "materialized_attention_model_matrix",
        "status": "pass",
        "matrix_complete": True,
        "models": args.models,
        "contexts": args.contexts,
        "batches": args.batches,
        "decode_tokens": args.decode_tokens,
        "cache_dtype": args.cache_dtype,
        "candidate_policy": args.candidate_policy,
        "runs_per_policy": args.runs,
        "case_count": len(cases),
        "all_accuracy_gates_passed": all(
            case["accuracy_gate_passed"] for case in cases),
        "all_performance_gates_passed": all(
            case["performance_gate_passed"] for case in cases),
        "minimum_default_sequence": boundary,
        "minimum_speedup": min(float(case["throughput_speedup"]) for case in cases),
        "maximum_speedup": max(float(case["throughput_speedup"]) for case in cases),
        "cases": cases,
    }
    (args.output_directory / "cases.jsonl").write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "matrix.svg").write_text(
        svg(cases), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"materialized_attention_model_matrix: {error}", file=sys.stderr)
        raise SystemExit(2) from error
