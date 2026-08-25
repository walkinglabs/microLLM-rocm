#!/usr/bin/env python3
"""Pair current and split-sequence cached Attention on one official model."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--cache-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--splits", type=int, default=32)
    parser.add_argument(
        "--candidate-policy", choices=("split", "materialized"),
        default="split")
    parser.add_argument("--minimum-sequence", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--maximum-logit-error", type=float, default=1.0e-3)
    parser.add_argument("--maximum-logit-rms", type=float, default=1.0e-4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    result = parser.parse_args()
    if (result.context <= 0 or result.batch <= 0 or result.decode_tokens <= 0 or
            result.splits <= 0 or result.splits > 32 or
            result.minimum_sequence <= 0 or result.warmup < 0 or
            result.steps <= 0 or result.runs <= 0 or
            result.maximum_logit_error < 0 or result.maximum_logit_rms < 0 or
            result.timeout_seconds <= 0):
        parser.error("model comparison options are outside the measured contract")
    for path in (result.manifest, result.binary):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def model_entry(path: Path, name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list):
        raise ValueError("manifest is not schema-version-1")
    matches = [model for model in models if model.get("name") == name]
    if len(matches) != 1:
        raise ValueError("requested model is missing or duplicated")
    model = matches[0]
    required = {"revision", "parameter_count", "config", "weights", "inference"}
    if required - model.keys() or not model["inference"].get("token_ids"):
        raise ValueError("model manifest entry is incomplete")
    return model


def expanded_tokens(seed: list[int], context: int) -> list[int]:
    return [int(seed[index % len(seed)]) for index in range(context)]


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model process emitted no JSON object")


def read_float32(path: Path) -> list[float]:
    values = array("f")
    with path.open("rb") as stream:
        values.frombytes(stream.read())
    return list(values)


def command(args: argparse.Namespace, model: dict, policy: str,
            logits_path: Path) -> list[str]:
    tokens = expanded_tokens(model["inference"]["token_ids"], args.context)
    is_candidate = policy == "split"
    splits = (args.splits if is_candidate and
              args.candidate_policy == "split" else 0)
    materialized = is_candidate and args.candidate_policy == "materialized"
    return [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "1", "--batch", str(args.batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", args.cache_dtype,
        "--cache-capacity", str(args.context + args.decode_tokens),
        "--new-tokens", str(args.decode_tokens),
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(args.steps),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--workload", "decode", "--cache-logits-output", str(logits_path),
        "--cached-attention-splits", str(splits),
        "--cached-attention-minimum-sequence", str(args.minimum_sequence),
        "--cached-attention-materialized", str(materialized).lower(),
    ]


def validate_record(record: dict, args: argparse.Namespace, model: dict,
                    policy: str) -> None:
    is_candidate = policy == "split"
    splits = (args.splits if is_candidate and
              args.candidate_policy == "split" else 0)
    materialized = is_candidate and args.candidate_policy == "materialized"
    required = {
        "parameter_count": model["parameter_count"],
        "token_count": args.context,
        "batch": args.batch,
        "warmup": args.warmup,
        "steps": args.steps,
        "use_cache": True,
        "cache_prefill_mode": "full",
        "decode_mode": "steady",
        "decode_step_semantics": "one_model_forward_per_measured_token",
        "kv_cache_dtype": args.cache_dtype,
        "requested_cache_capacity": args.context + args.decode_tokens,
        "kv_cache_capacity_tokens": args.context + args.decode_tokens,
        "kv_cache_active_tokens": args.context + args.decode_tokens,
        "cached_attention_splits": splits,
        "cached_attention_minimum_sequence": args.minimum_sequence,
        "cached_attention_materialized_scores": materialized,
        "cached_attention_materialized_policy": (
            "explicit-on" if materialized else "explicit-off"),
        "cached_attention_materialized_auto_eligible": False,
    }
    for field, expected in required.items():
        if record.get(field) != expected:
            raise ValueError(
                f"{policy} {field} expected {expected!r}, got {record.get(field)!r}")
    if (record.get("measured_tokens") !=
            args.batch * args.decode_tokens * args.steps or
            record.get("measured_forward_steps") !=
            args.batch * args.decode_tokens * args.steps or
            len(record.get("generated_tokens", [])) != args.decode_tokens or
            not math.isfinite(float(record.get("decode_tokens_per_second", math.nan))) or
            float(record["decode_tokens_per_second"]) <= 0 or
            int(record.get("engine_peak_bytes", 0)) <= 0 or
            int(record.get("kv_cache_actual_bytes", 0)) <= 0 or
            record.get("kv_cache_actual_bytes") != record.get("kv_cache_active_bytes") or
            int(record.get("engine_allocation_calls", 0)) <= 0 or
            int(record.get("engine_backend_allocation_calls", -1)) < 0):
        raise ValueError(f"{policy} timing/memory/token contract changed")


def run_policy(args: argparse.Namespace, model: dict, policy: str,
               run: int, order: str, temporary: Path) -> tuple[dict, list[float]]:
    logits_path = temporary / f"run-{run}-{policy}.bin"
    completed = subprocess.run(
        command(args, model, policy, logits_path), text=True,
        capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{policy} failed: {completed.stderr.strip() or completed.stdout.strip()}")
    record = last_json(completed.stdout)
    validate_record(record, args, model, policy)
    logits = read_float32(logits_path)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    expected = args.batch * int(config["vocab_size"])
    if len(logits) != expected or any(not math.isfinite(value) for value in logits):
        raise ValueError(f"{policy} cached logits are incomplete or non-finite")
    record.update({
        "schema_version": 1,
        "record_type": "cached_attention_split_model_measurement",
        "status": "pass",
        "model": args.model,
        "revision": model["revision"],
        "context": args.context,
        "policy": policy,
        "candidate_policy": args.candidate_policy,
        "process_run": run,
        "pair_order": order,
        "complete_logit_elements": len(logits),
    })
    return record, logits


def compare_pair(args: argparse.Namespace, current: tuple[dict, list[float]],
                 split: tuple[dict, list[float]], run: int, order: str) -> dict:
    current_record, current_logits = current
    split_record, split_logits = split
    differences = [abs(left - right)
                   for left, right in zip(current_logits, split_logits)]
    rms = math.sqrt(sum(value * value for value in differences) / len(differences))
    maximum = max(differences)
    tokens_equal = (current_record["generated_tokens"] ==
                    split_record["generated_tokens"])
    accuracy = (tokens_equal and maximum <= args.maximum_logit_error and
                rms <= args.maximum_logit_rms)
    return {
        "schema_version": 1,
        "record_type": "cached_attention_split_model_pair",
        "status": "pass" if accuracy else "failed",
        "model": args.model,
        "revision": current_record["revision"],
        "context": args.context,
        "batch": args.batch,
        "decode_tokens": args.decode_tokens,
        "cache_dtype": args.cache_dtype,
        "candidate_policy": args.candidate_policy,
        "splits": args.splits if args.candidate_policy == "split" else 0,
        "minimum_sequence": args.minimum_sequence,
        "process_run": run,
        "pair_order": order,
        "complete_logit_elements": len(differences),
        "maximum_logit_error": maximum,
        "logit_rms_error": rms,
        "generated_tokens_equal": tokens_equal,
        "current_tokens": current_record["generated_tokens"],
        "split_tokens": split_record["generated_tokens"],
        "current_throughput_tokens_per_second":
            current_record["decode_tokens_per_second"],
        "split_throughput_tokens_per_second":
            split_record["decode_tokens_per_second"],
        "throughput_speedup": (
            split_record["decode_tokens_per_second"] /
            current_record["decode_tokens_per_second"]),
        "current_peak_bytes": current_record["engine_peak_bytes"],
        "split_peak_bytes": split_record["engine_peak_bytes"],
        "peak_bytes_delta": (split_record["engine_peak_bytes"] -
                             current_record["engine_peak_bytes"]),
        "current_allocation_calls": current_record["engine_allocation_calls"],
        "split_allocation_calls": split_record["engine_allocation_calls"],
        "allocation_calls_delta": (split_record["engine_allocation_calls"] -
                                   current_record["engine_allocation_calls"]),
        "current_backend_allocation_calls":
            current_record["engine_backend_allocation_calls"],
        "split_backend_allocation_calls":
            split_record["engine_backend_allocation_calls"],
        "backend_allocation_calls_delta": (
            split_record["engine_backend_allocation_calls"] -
            current_record["engine_backend_allocation_calls"]),
        "kv_cache_bytes": current_record["kv_cache_actual_bytes"],
    }


def leave_one(values: list[float]) -> list[float]:
    return [statistics.median(values[:index] + values[index + 1:])
            for index in range(len(values))]


def svg(summary: dict) -> str:
    width, height = 1180, 620
    pairs = summary["pairs"]
    maximum = max(max(pair["current_throughput_tokens_per_second"],
                      pair["split_throughput_tokens_per_second"])
                  for pair in pairs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.small{font-size:13px}.label{font-size:16px}.title{font-size:23px;font-weight:700}'
        '.muted{fill:#94a3b8}</style>',
        '<text x="32" y="40" class="title">Official model · current vs split cached Attention</text>',
        f'<text x="32" y="67" class="small muted">{summary["model"]} · T{summary["context"]} · '
        f'B{summary["batch"]} · N{summary["decode_tokens"]} · '
        f'{summary["candidate_policy"]}</text>',
    ]
    for index, pair in enumerate(pairs):
        y = 115 + index * 115
        current_width = 690 * pair["current_throughput_tokens_per_second"] / maximum
        split_width = 690 * pair["split_throughput_tokens_per_second"] / maximum
        parts.extend([
            f'<text x="32" y="{y + 20}" class="label">pair {index + 1} · {pair["pair_order"]}</text>',
            f'<rect x="250" y="{y}" width="{current_width:.2f}" height="25" rx="4" fill="#64748b"/>',
            f'<rect x="250" y="{y + 34}" width="{split_width:.2f}" height="25" rx="4" fill="#22c55e"/>',
            f'<text x="{260 + current_width:.2f}" y="{y + 18}" class="small">current {pair["current_throughput_tokens_per_second"]:.2f}</text>',
            f'<text x="{260 + split_width:.2f}" y="{y + 52}" class="small">split {pair["split_throughput_tokens_per_second"]:.2f}</text>',
            f'<text x="1080" y="{y + 38}" class="label" text-anchor="end">{pair["throughput_speedup"]:.3f}x</text>',
        ])
    parts.extend([
        '<rect x="32" y="475" width="1116" height="110" rx="14" fill="#111827" stroke="#334155"/>',
        f'<text x="58" y="515" class="label">median speedup  {summary["median_throughput_speedup"]:.4f}x</text>',
        f'<text x="58" y="548" class="small">logit Max/RMS  {summary["maximum_logit_error"]:.3e} / {summary["maximum_logit_rms_error"]:.3e}</text>',
        f'<text x="620" y="515" class="label">peak delta  {summary["median_peak_bytes_delta"]:+,} bytes</text>',
        f'<text x="620" y="548" class="small">tokens exact  {str(summary["all_generated_tokens_equal"]).lower()}</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> int:
    args = arguments()
    model = model_entry(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    pairs = []
    with tempfile.TemporaryDirectory(prefix="microllm-split-model-") as directory:
        temporary = Path(directory)
        for run in range(1, args.runs + 1):
            order = "current-first" if run % 2 else "split-first"
            measured = {}
            for policy in (("current", "split") if order == "current-first"
                           else ("split", "current")):
                measured[policy] = run_policy(
                    args, model, policy, run, order, temporary)
                records.append(measured[policy][0])
            pair = compare_pair(
                args, measured["current"], measured["split"], run, order)
            pairs.append(pair)
            print(json.dumps(pair, sort_keys=True), flush=True)
    speedups = [float(pair["throughput_speedup"]) for pair in pairs]
    summary = {
        "schema_version": 1,
        "record_type": "cached_attention_split_model_summary",
        "status": "pass" if all(pair["status"] == "pass" for pair in pairs)
                  else "failed",
        "model": args.model,
        "revision": model["revision"],
        "context": args.context,
        "batch": args.batch,
        "decode_tokens": args.decode_tokens,
        "cache_dtype": args.cache_dtype,
        "candidate_policy": args.candidate_policy,
        "splits": args.splits if args.candidate_policy == "split" else 0,
        "minimum_sequence": args.minimum_sequence,
        "runs_per_policy": args.runs,
        "process_rows": len(records),
        "pair_rows": len(pairs),
        "warmup": args.warmup,
        "steps": args.steps,
        "maximum_logit_error": max(pair["maximum_logit_error"] for pair in pairs),
        "maximum_logit_rms_error": max(pair["logit_rms_error"] for pair in pairs),
        "all_generated_tokens_equal": all(
            pair["generated_tokens_equal"] for pair in pairs),
        "median_current_throughput_tokens_per_second": statistics.median(
            pair["current_throughput_tokens_per_second"] for pair in pairs),
        "median_split_throughput_tokens_per_second": statistics.median(
            pair["split_throughput_tokens_per_second"] for pair in pairs),
        "median_throughput_speedup": statistics.median(speedups),
        "paired_speedups": speedups,
        "leave_one_pair_out_speedups": leave_one(speedups),
        "median_peak_bytes_delta": statistics.median(
            pair["peak_bytes_delta"] for pair in pairs),
        "median_allocation_calls_delta": statistics.median(
            pair["allocation_calls_delta"] for pair in pairs),
        "median_backend_allocation_calls_delta": statistics.median(
            pair["backend_allocation_calls_delta"] for pair in pairs),
        "accuracy_gate_passed": all(pair["status"] == "pass" for pair in pairs),
        "performance_gate_passed": (
            statistics.median(speedups) >= 1.05 and
            min(leave_one(speedups)) >= 1.01),
        "pairs": pairs,
    }
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8")
    (args.output_directory / "pairs.jsonl").write_text(
        "".join(json.dumps(pair, sort_keys=True) + "\n" for pair in pairs),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "comparison.svg").write_text(
        svg(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"compare_cached_attention_split_models: {error}", file=sys.stderr)
        raise SystemExit(2) from error
