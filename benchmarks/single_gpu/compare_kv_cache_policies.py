#!/usr/bin/env python3
"""Same-binary paired comparison for two microLLM KV-cache policies."""

import argparse
import json
from pathlib import Path
import statistics

from hf_inference_shape_matrix import (
    load_models, micro_command, normalize_micro, positive_int_list,
    run_one_json, validate_measurement,
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--micro-binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models", default="qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--contexts", default="32,512,2048")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--candidate-fp32-layers", required=True)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    result = parser.parse_args()
    result.models = [name for name in result.models.split(",") if name]
    result.contexts = positive_int_list(result.contexts, "contexts")
    result.batches = positive_int_list(result.batches, "batches")
    if not result.manifest.is_file() or not result.micro_binary.is_file():
        parser.error("manifest and micro binary must exist")
    if result.decode_tokens <= 0 or result.warmup < 0 or result.steps <= 0 or \
            result.runs <= 0 or result.timeout_seconds <= 0:
        parser.error("decode-tokens/steps/runs/timeout must be positive; warmup nonnegative")
    return result


def runner_args(args: argparse.Namespace, fp32_layers: str) -> argparse.Namespace:
    return argparse.Namespace(
        micro_binary=args.micro_binary,
        decode_tokens=args.decode_tokens,
        warmup=args.warmup,
        steps=args.steps,
        micro_batch_argmax_mode="device",
        micro_kv_cache_dtype="bf16",
        micro_kv_cache_fp32_layers=fp32_layers,
    )


def summarize(records: list[dict], models: list[dict], contexts: list[int],
              batches: list[int], runs: int) -> dict:
    rows = []
    for model in models:
        for context in contexts:
            for batch in batches:
                selected = [record for record in records
                            if record["model"] == model["name"] and
                            record["context"] == context and
                            record["batch"] == batch]
                row = {"model": model["name"], "revision": model["revision"],
                       "context": context, "batch": batch}
                policy_records = {}
                for policy in ("uniform", "candidate"):
                    measured = [record for record in selected
                                if record["policy"] == policy and
                                record["status"] == "pass"]
                    if len(measured) != runs:
                        row[f"{policy}_status"] = "incomplete"
                        continue
                    policy_records[policy] = measured
                    row[f"{policy}_status"] = "pass"
                    for field in ("throughput_tokens_per_second",
                                  "mean_cache_prepare_ms",
                                  "mean_end_to_end_generation_ms",
                                  "peak_bytes", "kv_cache_actual_bytes"):
                        row[f"{policy}_{field}"] = statistics.median(
                            float(record[field]) for record in measured)
                    row[f"{policy}_generated_tokens"] = measured[0]["generated_tokens"]
                if len(policy_records) == 2:
                    row["throughput_ratio_candidate_over_uniform"] = \
                        row["candidate_throughput_tokens_per_second"] / \
                        row["uniform_throughput_tokens_per_second"]
                    row["prepare_speedup"] = row["uniform_mean_cache_prepare_ms"] / \
                        row["candidate_mean_cache_prepare_ms"]
                    row["end_to_end_speedup"] = \
                        row["uniform_mean_end_to_end_generation_ms"] / \
                        row["candidate_mean_end_to_end_generation_ms"]
                    row["peak_ratio"] = row["candidate_peak_bytes"] / row["uniform_peak_bytes"]
                    row["tokens_equal"] = row["candidate_generated_tokens"] == \
                        row["uniform_generated_tokens"]
                    row["status"] = "pass"
                else:
                    row["status"] = "incomplete"
                rows.append(row)
    return {"schema_version": 1, "track": "same_binary_kv_policy_comparison",
            "runs_per_policy": runs,
            "status": "pass" if all(row["status"] == "pass" for row in rows)
                      else "incomplete",
            "rows": rows}


def main() -> int:
    args = options()
    models = load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    records = []
    for model in models:
        for context in args.contexts:
            for batch in args.batches:
                for process_run in range(1, args.runs + 1):
                    order = ("uniform", "candidate") if process_run % 2 else \
                        ("candidate", "uniform")
                    for policy in order:
                        fp32_layers = "" if policy == "uniform" else \
                            args.candidate_fp32_layers
                        policy_args = runner_args(args, fp32_layers)
                        raw = run_one_json(micro_command(
                            policy_args, model, context, batch, "decode", "cached"),
                            args.timeout_seconds)
                        record = normalize_micro(raw, model, context, batch,
                                                 "decode", "cached", policy_args)
                        validate_measurement(record, model, "microllm", context, batch,
                                             "decode", "cached", args.warmup,
                                             args.steps, args.decode_tokens)
                        record.update({"policy": policy, "process_run": process_run,
                                       "pair_order": list(order)})
                        records.append(record)
                        with raw_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record, sort_keys=True) + "\n")
                        print(json.dumps(record, sort_keys=True), flush=True)
    summary = summarize(records, models, args.contexts, args.batches, args.runs)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
