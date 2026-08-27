#!/usr/bin/env python3
"""Five-case current-vs-up-FP32 performance gate for official Qwen3-0.6B."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


POLICIES = ("current", "up-fp32")
RUNS = 3
WARMUP = 2
STEPS = 5
RESIDENT_DELTA_BYTES = 176_160_768
CASES = (
    {"name": "cached_T1_B1_N1", "workload": "decode", "context": 1,
     "batch": 1, "decode_tokens": 1},
    {"name": "cached_T32_B1_N4", "workload": "decode", "context": 32,
     "batch": 1, "decode_tokens": 4},
    {"name": "cached_T128_B2_N32", "workload": "decode", "context": 128,
     "batch": 2, "decode_tokens": 32},
    {"name": "prefill_T512_B2", "workload": "prefill", "context": 512,
     "batch": 2, "decode_tokens": 0},
    {"name": "cached_T512_B2_N32", "workload": "decode", "context": 512,
     "batch": 2, "decode_tokens": 32},
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    for path in (args.manifest, args.binary):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def load_model(path: Path, name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list) or not models:
        raise RuntimeError("manifest must contain a non-empty schema-version-1 model list")
    selected = [model for model in models if model.get("name") == name]
    if len(selected) != 1:
        raise RuntimeError(f"manifest must contain exactly one {name} model")
    model = selected[0]
    for field in ("config", "weights", "revision", "parameter_count", "inference"):
        if field not in model:
            raise RuntimeError(f"model is missing {field}")
    for field in ("config", "weights"):
        if not Path(model[field]).is_file():
            raise RuntimeError(f"model {field} does not exist: {model[field]}")
    tokens = model["inference"].get("token_ids")
    if not isinstance(tokens, list) or not tokens or any(
            type(token) is not int or token < 0 for token in tokens):
        raise RuntimeError("model needs nonnegative inference token_ids")
    return model


def expanded_tokens(seed: list[int], context: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(context)]


def command(args: argparse.Namespace, model: dict, case: dict,
            policy: str) -> list[str]:
    workload = case["workload"]
    cached = workload == "decode"
    tokens = expanded_tokens(model["inference"]["token_ids"], case["context"])
    result = [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "1", "--batch", str(case["batch"]),
        "--use-cache", str(cached).lower(), "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--new-tokens", str(case["decode_tokens"]),
        "--warmup", str(WARMUP), "--steps", str(STEPS),
        "--prefill-warmup", str(WARMUP), "--prefill-steps", str(STEPS),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--workload", workload,
    ]
    if cached:
        result.extend(["--cache-capacity",
                       str(case["context"] + case["decode_tokens"])])
    if policy == "up-fp32":
        # gate/down stay BF16; up is the only FP32 FFN projection.
        result.extend(["--bf16-ffn-weight-scope", "gate-down"])
    return result


def output_signature(source: dict, workload: str) -> list[int]:
    if workload == "decode":
        tokens = source.get("generated_tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RuntimeError("decode process omitted generated tokens")
        return tokens
    top = source.get("top_logits")
    if not isinstance(top, list) or len(top) != 1 or type(top[0].get("token")) is not int:
        raise RuntimeError("prefill process omitted its top token")
    return [top[0]["token"]]


def run_one(args: argparse.Namespace, model: dict, case: dict, policy: str,
            process_run: int, order: tuple[str, ...]) -> dict:
    completed = subprocess.run(
        command(args, model, case, policy), capture_output=True, text=True,
        timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"{case['name']} {policy} emitted {len(lines)} JSON lines")
    source = json.loads(lines[0])
    expected_scope = "gate-down" if policy == "up-fp32" else "all"
    expected_ffn = 56 if policy == "up-fp32" else 84
    expected_forwards = case["batch"] * case["decode_tokens"] * STEPS
    required = {
        "status": "pass", "parameter_count": model["parameter_count"],
        "token_count": case["context"], "batch": case["batch"],
        "workload": case["workload"], "warmup": WARMUP, "steps": STEPS,
        "bf16_ffn_weight_scope": expected_scope,
        "bf16_ffn_converted_tensors": expected_ffn,
        "bf16_attention_converted_tensors": 112,
    }
    if any(source.get(field) != value for field, value in required.items()):
        raise RuntimeError(f"{case['name']} {policy} changed the formal contract")
    if case["workload"] == "decode":
        capacity = case["context"] + case["decode_tokens"]
        theoretical_cache_bytes = (
            2 * int(source.get("kv_cache_layers", 0)) *
            int(source.get("kv_cache_heads", 0)) *
            int(source.get("kv_cache_head_dimension", 0)) * case["batch"] *
            capacity * int(source.get("kv_cache_element_bytes", 0)))
        if (source.get("decode_tokens") != case["decode_tokens"] or
                source.get("measured_forward_steps") != expected_forwards or
                source.get("measured_tokens") != expected_forwards or
                source.get("kv_cache_capacity_tokens") != capacity or
                source.get("kv_cache_active_tokens") != capacity or
                source.get("kv_cache_actual_bytes") != theoretical_cache_bytes or
                source.get("kv_cache_active_bytes") != theoretical_cache_bytes):
            raise RuntimeError(f"{case['name']} {policy} changed decode/KV accounting")
        throughput = float(source["decode_tokens_per_second"])
        latency_ms = float(source["mean_generation_ms"])
    else:
        if source.get("decode_tokens") != 0:
            raise RuntimeError(f"{case['name']} {policy} unexpectedly decoded tokens")
        throughput = float(source["prefill_tokens_per_second"])
        latency_ms = float(source["forward_ms"])
    if not math.isfinite(throughput) or throughput <= 0.0 or \
            not math.isfinite(latency_ms) or latency_ms <= 0.0:
        raise RuntimeError(f"{case['name']} {policy} produced invalid timing")
    resident = int(source["resident_weight_bytes"])
    peak = int(source["engine_peak_bytes"])
    if resident <= 0 or peak < resident:
        raise RuntimeError(f"{case['name']} {policy} produced invalid engine memory")
    return {
        "schema_version": 1,
        "record_type": "qwen3_up_fp32_performance_sample",
        "status": "pass", "model": model["name"], "revision": model["revision"],
        "case": case["name"], "workload": case["workload"],
        "context": case["context"], "batch": case["batch"],
        "decode_tokens": case["decode_tokens"], "policy": policy,
        "process_run": process_run, "pair_order": list(order),
        "warmup": WARMUP, "steps": STEPS,
        "throughput_tokens_per_second": throughput, "latency_ms": latency_ms,
        "resident_weight_bytes": resident, "engine_peak_bytes": peak,
        "engine_incremental_peak_bytes": peak - resident,
        "preparation_peak_bytes": int(source["preparation_peak_bytes"]),
        "kv_cache_actual_bytes": int(source.get("kv_cache_actual_bytes", 0)),
        "kv_cache_capacity_tokens": int(source.get("kv_cache_capacity_tokens", 0)),
        "output_signature": output_signature(source, case["workload"]),
    }


def median_fields(rows: list[dict]) -> dict:
    return {
        field: statistics.median(row[field] for row in rows)
        for field in ("throughput_tokens_per_second", "latency_ms",
                      "resident_weight_bytes", "engine_peak_bytes",
                      "engine_incremental_peak_bytes", "preparation_peak_bytes")
    }


def summarize(records: list[dict], model: dict) -> dict:
    case_rows = []
    ratios = []
    for case in CASES:
        grouped = {
            policy: [row for row in records
                     if row["case"] == case["name"] and row["policy"] == policy]
            for policy in POLICIES
        }
        if any(len(rows) != RUNS for rows in grouped.values()):
            raise RuntimeError(f"{case['name']} needs {RUNS} samples per policy")
        current = median_fields(grouped["current"])
        candidate = median_fields(grouped["up-fp32"])
        throughput_ratio = (candidate["throughput_tokens_per_second"] /
                            current["throughput_tokens_per_second"])
        latency_ratio = candidate["latency_ms"] / current["latency_ms"]
        ratios.append(throughput_ratio)
        current_outputs = {
            tuple(row["output_signature"]) for row in grouped["current"]}
        candidate_outputs = {
            tuple(row["output_signature"]) for row in grouped["up-fp32"]}
        outputs_equal_across_policies = current_outputs == candidate_outputs
        incremental_tolerance = max(
            8 * 1024 * 1024, 0.05 * current["engine_incremental_peak_bytes"])
        gates = {
            "throughput_at_least_point95": throughput_ratio >= 0.95,
            "latency_at_most_1_05": latency_ratio <= 1.05,
            "resident_delta_exact":
                candidate["resident_weight_bytes"] -
                current["resident_weight_bytes"] == RESIDENT_DELTA_BYTES,
            "incremental_peak_within_tolerance":
                candidate["engine_incremental_peak_bytes"] <=
                current["engine_incremental_peak_bytes"] + incremental_tolerance,
            "current_output_deterministic": len(current_outputs) == 1,
            "candidate_output_deterministic": len(candidate_outputs) == 1,
        }
        case_rows.append({
            **case, "runs": RUNS,
            "current": current, "up_fp32": candidate,
            "candidate_over_current_throughput": throughput_ratio,
            "candidate_over_current_latency": latency_ratio,
            "outputs_equal_across_policies": outputs_equal_across_policies,
            "incremental_peak_tolerance_bytes": incremental_tolerance,
            "gates": gates, "status": "pass" if all(gates.values()) else "reject",
        })
    geometric_mean = math.prod(ratios) ** (1.0 / len(ratios))
    gates = {
        "all_case_gates_pass": all(row["status"] == "pass" for row in case_rows),
        "throughput_geometric_mean_at_least_point97": geometric_mean >= 0.97,
    }
    return {
        "schema_version": 1,
        "record_type": "qwen3_up_fp32_performance_gate",
        "status": "pass_performance" if all(gates.values()) else "reject_performance",
        "model": model["name"], "revision": model["revision"],
        "candidate": "FFN up FP32; FFN gate/down and all Attention BF16; BF16 Cache",
        "runs": RUNS, "warmup": WARMUP, "steps": STEPS,
        "pairing": "fresh processes; policy order alternates by process run",
        "candidate_over_current_throughput_geometric_mean": geometric_mean,
        "gates": gates, "cases": case_rows,
        "boundary": (
            "performance-only gate against the current mixed-BF16 implementation; "
            "each policy must be deterministic, while cross-policy output changes are "
            "accepted or rejected only by the separate FP32-oracle shape gate"),
    }


def main() -> int:
    args = options()
    model = load_model(args.manifest, args.model)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    records = []
    for case in CASES:
        for process_run in range(1, RUNS + 1):
            order = POLICIES if process_run % 2 else tuple(reversed(POLICIES))
            for policy in order:
                record = run_one(args, model, case, policy, process_run, order)
                records.append(record)
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
    summary = summarize(records, model)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "pass_performance" else 2


if __name__ == "__main__":
    raise SystemExit(main())
