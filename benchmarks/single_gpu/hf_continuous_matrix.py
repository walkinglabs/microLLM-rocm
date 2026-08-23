#!/usr/bin/env python3
"""Fresh-process official-model continuous-serving matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


SUITES = {
    "smoke": {
        "short_s2": {"slots": 2, "prompts": [8, 8], "outputs": [2, 3]},
    },
    "standard": {
        "short_s2": {"slots": 2, "prompts": [8, 8, 32, 32],
                     "outputs": [8, 16, 8, 16]},
        "short_s4": {"slots": 4, "prompts": [8, 8, 16, 16, 32, 32, 64, 64],
                     "outputs": [8, 16, 8, 16, 8, 16, 8, 16]},
        "long_s2": {"slots": 2, "prompts": [512, 512, 2048, 2048],
                    "outputs": [8, 16, 8, 16]},
        "long_s4": {"slots": 4,
                    "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
                    "outputs": [8, 8, 8, 8, 16, 16, 16, 16]},
    },
    "slot-sweep": {
        **{
            f"short_s{slots}": {
                "slots": slots,
                "group": "short",
                "prompts": [8, 8, 16, 16, 32, 32, 64, 64],
                "outputs": [8, 16, 8, 16, 8, 16, 8, 16],
            }
            for slots in (1, 2, 4, 8)
        },
        **{
            f"long_s{slots}": {
                "slots": slots,
                "group": "long",
                "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
                "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
            }
            for slots in (1, 2, 4, 8)
        },
    },
    "length-buckets": {
        "long_uniform_s8": {
            "slots": 8,
            "policy": "uniform",
            "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
            "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
        },
        "long_bucketed_s8": {
            "slots": 8,
            "policy": "length_bucketed",
            "buckets": [
                {"max_sequence_length": 264, "max_slots": 2},
                {"max_sequence_length": 520, "max_slots": 2},
                {"max_sequence_length": 1040, "max_slots": 2},
                {"max_sequence_length": 2064, "max_slots": 2},
            ],
            "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
            "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
        },
    },
    "bucket-sweep": {
        "long_buckets_1": {
            "slots": 8,
            "bucket_count": 1,
            "policy": "uniform",
            "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
            "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
        },
        "long_buckets_2": {
            "slots": 8,
            "bucket_count": 2,
            "policy": "length_bucketed",
            "buckets": [
                {"max_sequence_length": 520, "max_slots": 4},
                {"max_sequence_length": 2064, "max_slots": 4},
            ],
            "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
            "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
        },
        "long_buckets_4": {
            "slots": 8,
            "bucket_count": 4,
            "policy": "length_bucketed",
            "buckets": [
                {"max_sequence_length": 264, "max_slots": 2},
                {"max_sequence_length": 520, "max_slots": 2},
                {"max_sequence_length": 1040, "max_slots": 2},
                {"max_sequence_length": 2064, "max_slots": 2},
            ],
            "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
            "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
        },
    },
    "traffic-skew": {
        "short_heavy_uniform": {
            "slots": 8, "group": "short_heavy", "policy": "uniform",
            "focus_indices": [0, 1, 2, 3, 4, 5],
            "prompts": [256, 256, 512, 512, 256, 512, 1024, 2048],
            "outputs": [8, 8, 8, 8, 8, 8, 16, 16],
        },
        "short_heavy_bucket2": {
            "slots": 8, "group": "short_heavy", "policy": "bucketed",
            "focus_indices": [0, 1, 2, 3, 4, 5],
            "buckets": [
                {"max_sequence_length": 520, "max_slots": 4},
                {"max_sequence_length": 2064, "max_slots": 4},
            ],
            "prompts": [256, 256, 512, 512, 256, 512, 1024, 2048],
            "outputs": [8, 8, 8, 8, 8, 8, 16, 16],
        },
        "long_heavy_uniform": {
            "slots": 8, "group": "long_heavy", "policy": "uniform",
            "focus_indices": [2, 3, 4, 5, 6, 7],
            "prompts": [256, 512, 1024, 1024, 2048, 2048, 1024, 2048],
            "outputs": [8, 8, 16, 16, 16, 16, 16, 16],
        },
        "long_heavy_bucket2": {
            "slots": 8, "group": "long_heavy", "policy": "bucketed",
            "focus_indices": [2, 3, 4, 5, 6, 7],
            "buckets": [
                {"max_sequence_length": 520, "max_slots": 4},
                {"max_sequence_length": 2064, "max_slots": 4},
            ],
            "prompts": [256, 512, 1024, 1024, 2048, 2048, 1024, 2048],
            "outputs": [8, 8, 16, 16, 16, 16, 16, 16],
        },
        "delayed_uniform": {
            "slots": 8, "group": "delayed", "policy": "uniform",
            "focus_indices": [4, 5, 6, 7],
            "arrivals": [0, 0, 0, 0, 4, 4, 4, 4],
            "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
            "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
        },
        "delayed_bucket2": {
            "slots": 8, "group": "delayed", "policy": "bucketed",
            "focus_indices": [4, 5, 6, 7],
            "arrivals": [0, 0, 0, 0, 4, 4, 4, 4],
            "buckets": [
                {"max_sequence_length": 520, "max_slots": 4},
                {"max_sequence_length": 2064, "max_slots": 4},
            ],
            "prompts": [256, 256, 512, 512, 1024, 1024, 2048, 2048],
            "outputs": [8, 8, 8, 8, 16, 16, 16, 16],
        },
    },
}

# Reuse the exact Experiment 116 traffic axes. The new suite adds one policy
# dimension without copying or silently changing prompts, arrivals, or focus rows.
SUITES["traffic-overflow"] = {}
for _case_name, _case in SUITES["traffic-skew"].items():
    if _case["policy"] == "uniform":
        SUITES["traffic-overflow"][_case_name] = dict(_case)
        continue
    _fixed_name = _case_name.replace("_bucket2", "_fixed")
    SUITES["traffic-overflow"][_fixed_name] = {
        **_case, "policy": "fixed", "overflow": False,
    }
    _overflow_name = _case_name.replace("_bucket2", "_overflow")
    _overflow_case = {**_case, "policy": "overflow", "overflow": True}
    if _case["group"] == "short_heavy":
        _overflow_case["expected_routes"] = [0, 0, 0, 0, 1, 1, 1, 1]
        _overflow_case["expected_overflow"] = 2
    else:
        _overflow_case["expected_routes"] = (
            [0, 0, 1, 1, 1, 1, 1, 1] if _case["group"] == "long_heavy"
            else [0, 0, 0, 0, 1, 1, 1, 1])
        _overflow_case["expected_overflow"] = 0
    SUITES["traffic-overflow"][_overflow_name] = _overflow_case


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--suite", choices=tuple(SUITES), default="standard")
    parser.add_argument("--models")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--max-idle-vram-percent", type=int, default=5)
    parser.add_argument("--max-idle-use-percent", type=int, default=10)
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    if result.warmup < 0 or result.steps <= 0 or result.runs <= 0 or \
            result.timeout_seconds <= 0 or \
            not 0 <= result.max_idle_vram_percent <= 100 or \
            not 0 <= result.max_idle_use_percent <= 100:
        parser.error("warmup must be nonnegative and steps/runs/timeout positive")
    if result.physical_gpu_index is not None and result.physical_gpu_index < 0:
        parser.error("physical GPU index must be nonnegative")
    result.models = result.models.split(",") if result.models else None
    return result


def load_models(path: Path, selected: list[str] | None = None) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list) or not models:
        raise RuntimeError("manifest needs schema-version-1 models")
    by_name = {model.get("name"): model for model in models}
    names = selected or list(by_name)
    if None in by_name or len(by_name) != len(models) or not set(names) <= set(by_name):
        raise RuntimeError("manifest model names are missing, duplicate, or unknown")
    return [by_name[name] for name in names]


def parse_gpu_state(text: str, physical_index: int) -> dict:
    document = json.loads(text)
    card = document.get(f"card{physical_index}")
    if not isinstance(card, dict):
        raise RuntimeError(f"rocm-smi did not report physical GPU {physical_index}")
    try:
        return {
            "physical_gpu_index": physical_index,
            "gpu_use_percent": int(card["GPU use (%)"]),
            "vram_percent": int(card["GPU Memory Allocated (VRAM%)"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("rocm-smi idle-state schema changed") from error


def require_idle_gpu(physical_index: int | None, maximum_vram: int,
                     maximum_use: int, boundary: str) -> dict | None:
    if physical_index is None:
        return None
    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmemuse", "--json"],
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot verify {boundary} GPU state: " +
            (completed.stderr.strip() or completed.stdout.strip()))
    state = parse_gpu_state(completed.stdout, physical_index)
    if state["vram_percent"] > maximum_vram:
        raise RuntimeError(
            f"physical GPU {physical_index} is externally occupied at {boundary}: "
            f"VRAM {state['vram_percent']}% exceeds {maximum_vram}%")
    if state["gpu_use_percent"] > maximum_use:
        raise RuntimeError(
            f"physical GPU {physical_index} is externally busy at {boundary}: "
            f"use {state['gpu_use_percent']}% exceeds {maximum_use}%")
    return state


def model_cache_shape(config_path: str | Path) -> tuple[int, int, int]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dimension = int(config.get("head_dim", hidden // heads))
    return layers, kv_heads, head_dimension


def theoretical_cache_bytes(model: dict, case: dict, element_bytes: int = 2) -> int:
    layers, kv_heads, head_dimension = model_cache_shape(model["config"])
    if case.get("buckets"):
        token_slots = sum(int(bucket["max_sequence_length"]) *
                          int(bucket["max_slots"])
                          for bucket in case["buckets"])
    else:
        capacity = max(prompt + output for prompt, output in
                       zip(case["prompts"], case["outputs"]))
        token_slots = int(case["slots"]) * capacity
    return 2 * layers * kv_heads * head_dimension * token_slots * element_bytes


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= quantile <= 1.0:
        raise RuntimeError("percentile input is invalid")
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def command(binary: Path, model: dict, case: dict,
            warmup: int, steps: int) -> list[str]:
    tokens = model["inference"]["token_ids"]
    result = [
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "1", "--new-tokens", "0",
        "--warmup", str(warmup), "--steps", str(steps),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--workload", "continuous", "--kv-cache-dtype", "bf16",
        "--continuous-slots", str(case["slots"]),
        "--continuous-prompt-lengths", ",".join(map(str, case["prompts"])),
        "--continuous-new-token-lengths", ",".join(map(str, case["outputs"])),
    ]
    if case.get("buckets"):
        result.extend([
            "--continuous-cache-buckets",
            ",".join(
                f"{bucket['max_sequence_length']}:{bucket['max_slots']}"
                for bucket in case["buckets"]),
        ])
    if case.get("arrivals"):
        result.extend([
            "--continuous-arrival-steps",
            ",".join(map(str, case["arrivals"])),
        ])
    if case.get("overflow"):
        result.extend(["--continuous-bucket-overflow", "true"])
    return result


def validate(record: dict, model: dict, case_name: str, case: dict,
             warmup: int, steps: int) -> dict:
    expected_tokens = sum(case["outputs"]) * steps
    expected_cache = theoretical_cache_bytes(model, case)
    required = {
        "status": "pass",
        "record_type": "official_continuous_serving_measurement",
        "request_count": len(case["prompts"]),
        "continuous_slots": case["slots"],
        "warmup": warmup,
        "steps": steps,
        "measured_tokens": expected_tokens,
        "prompt_lengths": case["prompts"],
        "new_token_lengths": case["outputs"],
        "deterministic_across_steps": True,
        "bucketed_cache": bool(case.get("buckets")),
        "arrival_steps": case.get("arrivals", [0] * len(case["prompts"])),
        "continuous_bucket_overflow": bool(case.get("overflow")),
        "overflow_routed_requests": int(case.get("expected_overflow", 0)),
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"{model['name']} {case_name} changed its serving contract")
    if int(record.get("allocated_cache_bytes", -1)) != expected_cache or \
            not 0 < int(record.get("peak_active_cache_bytes", 0)) <= expected_cache or \
            not 0.0 < float(record.get("kv_cache_byte_utilization", 0.0)) <= 1.0 or \
            float(record.get("tokens_per_second", 0.0)) <= 0.0 or \
            int(record.get("engine_peak_bytes", 0)) <= 0 or \
            int(record.get("resident_weight_bytes", 0)) <= 0:
        raise RuntimeError(f"{model['name']} {case_name} has invalid memory/timing evidence")
    expected_buckets = case.get("buckets", [])
    expected_routes = []
    if expected_buckets:
        if sum(int(bucket["max_slots"]) for bucket in expected_buckets) != \
                int(case["slots"]):
            raise RuntimeError("length bucket slots changed the total concurrency")
        for prompt, output in zip(case["prompts"], case["outputs"]):
            required_tokens = prompt + output
            expected_routes.append(next(
                index for index, bucket in enumerate(expected_buckets)
                if required_tokens <= int(bucket["max_sequence_length"])))
    expected_routes = case.get("expected_routes", expected_routes)
    if record.get("continuous_cache_buckets") != expected_buckets or \
            record.get("request_bucket_indices") != expected_routes:
        raise RuntimeError(f"{model['name']} {case_name} has invalid bucket routing")
    ttft = record.get("request_ttft_ms", [])
    completion = record.get("request_completion_ms", [])
    if len(ttft) != len(case["prompts"]) or len(completion) != len(ttft) or \
            any(float(first) < 0 or float(last) < float(first)
                for first, last in zip(ttft, completion)) or \
            abs(float(record.get("request_ttft_p50_ms", -1)) -
                percentile(ttft, 0.50)) > 1.0e-5 or \
            abs(float(record.get("request_ttft_p95_ms", -1)) -
                percentile(ttft, 0.95)) > 1.0e-5 or \
            abs(float(record.get("request_completion_p50_ms", -1)) -
                percentile(completion, 0.50)) > 1.0e-5 or \
            abs(float(record.get("request_completion_p95_ms", -1)) -
                percentile(completion, 0.95)) > 1.0e-5:
        raise RuntimeError(f"{model['name']} {case_name} has invalid request latency evidence")
    return {**record, "model": model["name"], "revision": model["revision"],
            "case": case_name, "expected_cache_bytes": expected_cache}


def token_difference(reference: list[list[int]], actual: list[list[int]]) -> dict:
    differing_requests = []
    first = None
    for request, (left, right) in enumerate(zip(reference, actual)):
        if left == right:
            continue
        differing_requests.append(request)
        if first is None:
            limit = min(len(left), len(right))
            token = next((index for index in range(limit)
                          if left[index] != right[index]), limit)
            first = {"request": request, "token": token}
    if len(reference) != len(actual):
        differing_requests.extend(range(min(len(reference), len(actual)),
                                          max(len(reference), len(actual))))
        if first is None:
            first = {"request": min(len(reference), len(actual)), "token": 0}
    return {
        "exact": reference == actual,
        "differing_request_count": len(differing_requests),
        "differing_requests": differing_requests,
        "first_difference": first,
    }


def focused_policy_summary(selected: list[dict], aggregate: dict,
                           focus: list[int]) -> dict:
    return {
        "raw": selected,
        "aggregate": aggregate,
        "focus_ttft_p50_ms": statistics.median(
            percentile([row["request_ttft_ms"][index] for index in focus], 0.50)
            for row in selected),
        "focus_ttft_p95_ms": statistics.median(
            percentile([row["request_ttft_ms"][index] for index in focus], 0.95)
            for row in selected),
        "focus_completion_p50_ms": statistics.median(
            percentile([row["request_completion_ms"][index]
                        for index in focus], 0.50) for row in selected),
        "focus_completion_p95_ms": statistics.median(
            percentile([row["request_completion_ms"][index]
                        for index in focus], 0.95) for row in selected),
    }


def main() -> int:
    args = options()
    models = load_models(args.manifest, args.models)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    rows = []
    for model in models:
        for case_name, case in SUITES[args.suite].items():
            for process_run in range(1, args.runs + 1):
                pre_gpu_state = require_idle_gpu(
                    args.physical_gpu_index, args.max_idle_vram_percent,
                    args.max_idle_use_percent,
                    f"{model['name']} {case_name} run {process_run} pre")
                completed = subprocess.run(
                    command(args.binary, model, case, args.warmup, args.steps),
                    capture_output=True, text=True, timeout=args.timeout_seconds)
                post_gpu_state = require_idle_gpu(
                    args.physical_gpu_index, args.max_idle_vram_percent,
                    args.max_idle_use_percent,
                    f"{model['name']} {case_name} run {process_run} post")
                if completed.returncode == 0:
                    lines = [line for line in completed.stdout.splitlines() if line.strip()]
                    if len(lines) != 1:
                        raise RuntimeError("continuous worker must emit one JSON line")
                    record = validate(json.loads(lines[0]), model, case_name, case,
                                      args.warmup, args.steps)
                else:
                    text = completed.stderr.strip() or completed.stdout.strip()
                    status = "oom" if "out of memory" in text.lower() else "failed"
                    record = {"schema_version": 1, "status": status,
                              "record_type": "official_continuous_serving_measurement",
                              "model": model["name"], "revision": model["revision"],
                              "case": case_name, "error": text}
                record["process_run"] = process_run
                if pre_gpu_state is not None:
                    record["pre_run_gpu_state"] = pre_gpu_state
                    record["post_run_gpu_state"] = post_gpu_state
                rows.append(record)
                with raw_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
    aggregates = []
    for model in models:
        for case_name in SUITES[args.suite]:
            selected = [row for row in rows if row.get("model") == model["name"] and
                        row.get("case") == case_name and row.get("status") == "pass"]
            aggregate = {"model": model["name"], "case": case_name,
                         "successful_runs": len(selected), "required_runs": args.runs}
            if len(selected) == args.runs:
                checksums = {int(row["token_checksum"]) for row in selected}
                if len(checksums) != 1:
                    raise RuntimeError(f"{model['name']} {case_name} checksum changed across runs")
                throughput = sorted(float(row["tokens_per_second"]) for row in selected)
                peak = sorted(int(row["engine_peak_bytes"]) for row in selected)
                ttft_p50 = sorted(float(row["request_ttft_p50_ms"])
                                  for row in selected)
                ttft_p95 = sorted(float(row["request_ttft_p95_ms"])
                                  for row in selected)
                completion_p50 = sorted(float(row["request_completion_p50_ms"])
                                        for row in selected)
                completion_p95 = sorted(float(row["request_completion_p95_ms"])
                                        for row in selected)
                aggregate.update({
                    "status": "pass",
                    "tokens_per_second_min": throughput[0],
                    "tokens_per_second_p50": statistics.median(throughput),
                    "tokens_per_second_max": throughput[-1],
                    "engine_peak_bytes_min": peak[0],
                    "engine_peak_bytes_p50": statistics.median(peak),
                    "engine_peak_bytes_max": peak[-1],
                    "request_ttft_p50_ms_p50": statistics.median(ttft_p50),
                    "request_ttft_p95_ms_p50": statistics.median(ttft_p95),
                    "request_completion_p50_ms_p50": statistics.median(completion_p50),
                    "request_completion_p95_ms_p50": statistics.median(completion_p95),
                    "token_checksum": checksums.pop(),
                })
            else:
                aggregate["status"] = "limited"
            aggregates.append(aggregate)
    slot_sweeps = []
    if args.suite == "slot-sweep":
        for model in models:
            for group in ("short", "long"):
                selected = [row for row in aggregates
                            if row["model"] == model["name"] and
                            SUITES[args.suite][row["case"]]["group"] == group]
                selected.sort(key=lambda row: SUITES[args.suite][row["case"]]["slots"])
                if len(selected) != 4 or any(row["status"] != "pass" for row in selected):
                    raise RuntimeError(f"incomplete slot sweep: {model['name']} {group}")
                baseline_tps = float(selected[0]["tokens_per_second_p50"])
                baseline_peak = int(selected[0]["engine_peak_bytes_p50"])
                group_rows = []
                expected_tokens = None
                group_exact = True
                for aggregate in selected:
                    case = SUITES[args.suite][aggregate["case"]]
                    slots = int(case["slots"])
                    raw = [row for row in rows if row.get("model") == model["name"] and
                           row.get("case") == aggregate["case"]]
                    current_tokens = raw[0]["generated_tokens"]
                    if any(row["generated_tokens"] != current_tokens for row in raw):
                        raise RuntimeError("generated tokens changed within a slot case")
                    if expected_tokens is None:
                        expected_tokens = current_tokens
                    difference = token_difference(expected_tokens, current_tokens)
                    group_exact = group_exact and difference["exact"]
                    tps = float(aggregate["tokens_per_second_p50"])
                    peak = int(aggregate["engine_peak_bytes_p50"])
                    group_rows.append({
                        "slots": slots,
                        "tokens_per_second_p50": tps,
                        "speedup_vs_s1": tps / baseline_tps,
                        "parallel_efficiency_vs_s1": tps / (baseline_tps * slots),
                        "engine_peak_bytes_p50": peak,
                        "engine_peak_growth_vs_s1": peak / baseline_peak,
                        "allocated_cache_bytes": raw[0]["allocated_cache_bytes"],
                        "peak_active_cache_bytes": raw[0]["peak_active_cache_bytes"],
                        "kv_cache_byte_utilization": raw[0]["kv_cache_byte_utilization"],
                        "slot_utilization": raw[0]["slot_utilization"],
                        "generated_tokens_equal_to_s1": difference["exact"],
                        "token_difference_vs_s1": difference,
                    })
                slot_sweeps.append({
                    "model": model["name"],
                    "group": group,
                    "request_count": 8,
                    "slots": group_rows,
                    "generated_tokens_equal_across_slots": group_exact,
                })
    bucket_comparisons = []
    if args.suite == "length-buckets":
        for model in models:
            uniform = next(row for row in rows
                           if row.get("model") == model["name"] and
                           row.get("case") == "long_uniform_s8")
            bucketed = next(row for row in rows
                            if row.get("model") == model["name"] and
                            row.get("case") == "long_bucketed_s8")
            uniform_aggregate = next(row for row in aggregates
                                     if row.get("model") == model["name"] and
                                     row.get("case") == "long_uniform_s8")
            bucketed_aggregate = next(row for row in aggregates
                                      if row.get("model") == model["name"] and
                                      row.get("case") == "long_bucketed_s8")
            if uniform.get("status") != "pass" or bucketed.get("status") != "pass":
                continue
            difference = token_difference(
                uniform["generated_tokens"], bucketed["generated_tokens"])
            bucket_comparisons.append({
                "model": model["name"],
                "token_difference": difference,
                "allocated_cache_ratio": (
                    bucketed["allocated_cache_bytes"] /
                    uniform["allocated_cache_bytes"]),
                "tokens_per_second_ratio": (
                    bucketed_aggregate["tokens_per_second_p50"] /
                    uniform_aggregate["tokens_per_second_p50"]),
                "request_ttft_p50_ratio": (
                    bucketed_aggregate["request_ttft_p50_ms_p50"] /
                    uniform_aggregate["request_ttft_p50_ms_p50"]),
                "request_completion_p50_ratio": (
                    bucketed_aggregate["request_completion_p50_ms_p50"] /
                    uniform_aggregate["request_completion_p50_ms_p50"]),
            })
    bucket_sweeps = []
    if args.suite == "bucket-sweep":
        for model in models:
            selected = []
            baseline_tokens = None
            baseline_cache = None
            baseline_tps = None
            all_exact = True
            for case_name, case in SUITES[args.suite].items():
                raw = [row for row in rows
                       if row.get("model") == model["name"] and
                       row.get("case") == case_name]
                aggregate = next(row for row in aggregates
                                 if row.get("model") == model["name"] and
                                 row.get("case") == case_name)
                if len(raw) != args.runs or aggregate.get("status") != "pass":
                    continue
                current_tokens = raw[0]["generated_tokens"]
                if baseline_tokens is None:
                    baseline_tokens = current_tokens
                    baseline_cache = int(raw[0]["allocated_cache_bytes"])
                    baseline_tps = float(aggregate["tokens_per_second_p50"])
                difference = token_difference(baseline_tokens, current_tokens)
                all_exact = all_exact and difference["exact"]
                selected.append({
                    "bucket_count": int(case["bucket_count"]),
                    "allocated_cache_bytes": int(raw[0]["allocated_cache_bytes"]),
                    "allocated_cache_ratio": (
                        int(raw[0]["allocated_cache_bytes"]) / baseline_cache),
                    "tokens_per_second_p50": aggregate["tokens_per_second_p50"],
                    "tokens_per_second_ratio": (
                        float(aggregate["tokens_per_second_p50"]) / baseline_tps),
                    "request_ttft_p50_ms": aggregate["request_ttft_p50_ms_p50"],
                    "request_ttft_p95_ms": aggregate["request_ttft_p95_ms_p50"],
                    "request_completion_p50_ms": (
                        aggregate["request_completion_p50_ms_p50"]),
                    "request_completion_p95_ms": (
                        aggregate["request_completion_p95_ms_p50"]),
                    "engine_peak_bytes_p50": aggregate["engine_peak_bytes_p50"],
                    "token_difference_vs_one_bucket": difference,
                })
            selected.sort(key=lambda row: row["bucket_count"])
            bucket_sweeps.append({
                "model": model["name"],
                "request_count": 8,
                "rows": selected,
                "generated_tokens_equal_across_bucket_counts": all_exact,
            })
    traffic_comparisons = []
    if args.suite == "traffic-skew":
        for model in models:
            for group in ("short_heavy", "long_heavy", "delayed"):
                cases = [(name, case) for name, case in SUITES[args.suite].items()
                         if case["group"] == group]
                by_policy = {}
                for case_name, case in cases:
                    selected = [row for row in rows
                                if row.get("model") == model["name"] and
                                row.get("case") == case_name and
                                row.get("status") == "pass"]
                    aggregate = next(row for row in aggregates
                                     if row.get("model") == model["name"] and
                                     row.get("case") == case_name)
                    focus = case["focus_indices"]
                    by_policy[case["policy"]] = focused_policy_summary(
                        selected, aggregate, focus)
                uniform = by_policy["uniform"]
                bucketed = by_policy["bucketed"]
                difference = token_difference(
                    uniform["raw"][0]["generated_tokens"],
                    bucketed["raw"][0]["generated_tokens"])
                traffic_comparisons.append({
                    "model": model["name"],
                    "group": group,
                    "focus_indices": next(case["focus_indices"] for _, case in cases),
                    "token_difference": difference,
                    "bucketed_over_uniform_tps": (
                        bucketed["aggregate"]["tokens_per_second_p50"] /
                        uniform["aggregate"]["tokens_per_second_p50"]),
                    "bucketed_over_uniform_focus_ttft": (
                        bucketed["focus_ttft_p50_ms"] /
                        uniform["focus_ttft_p50_ms"]),
                    "bucketed_over_uniform_focus_ttft_p95": (
                        bucketed["focus_ttft_p95_ms"] /
                        uniform["focus_ttft_p95_ms"]),
                    "bucketed_over_uniform_focus_completion": (
                        bucketed["focus_completion_p50_ms"] /
                        uniform["focus_completion_p50_ms"]),
                    "bucketed_over_uniform_focus_completion_p95": (
                        bucketed["focus_completion_p95_ms"] /
                        uniform["focus_completion_p95_ms"]),
                    "uniform_focus_ttft_p50_ms": uniform["focus_ttft_p50_ms"],
                    "bucketed_focus_ttft_p50_ms": bucketed["focus_ttft_p50_ms"],
                    "uniform_focus_ttft_p95_ms": uniform["focus_ttft_p95_ms"],
                    "bucketed_focus_ttft_p95_ms": bucketed["focus_ttft_p95_ms"],
                    "uniform_focus_completion_p50_ms":
                        uniform["focus_completion_p50_ms"],
                    "bucketed_focus_completion_p50_ms":
                        bucketed["focus_completion_p50_ms"],
                    "uniform_focus_completion_p95_ms":
                        uniform["focus_completion_p95_ms"],
                    "bucketed_focus_completion_p95_ms":
                        bucketed["focus_completion_p95_ms"],
                })
    overflow_comparisons = []
    if args.suite == "traffic-overflow":
        for model in models:
            for group in ("short_heavy", "long_heavy", "delayed"):
                cases = [(name, case) for name, case in SUITES[args.suite].items()
                         if case["group"] == group]
                by_policy = {}
                for case_name, case in cases:
                    selected = [row for row in rows
                                if row.get("model") == model["name"] and
                                row.get("case") == case_name and
                                row.get("status") == "pass"]
                    aggregate = next(row for row in aggregates
                                     if row.get("model") == model["name"] and
                                     row.get("case") == case_name)
                    by_policy[case["policy"]] = focused_policy_summary(
                        selected, aggregate, case["focus_indices"])
                uniform = by_policy["uniform"]
                fixed = by_policy["fixed"]
                overflow = by_policy["overflow"]
                fixed_difference = token_difference(
                    fixed["raw"][0]["generated_tokens"],
                    overflow["raw"][0]["generated_tokens"])
                uniform_difference = token_difference(
                    uniform["raw"][0]["generated_tokens"],
                    overflow["raw"][0]["generated_tokens"])
                overflow_comparisons.append({
                    "model": model["name"],
                    "group": group,
                    "focus_indices": next(case["focus_indices"]
                                          for _, case in cases),
                    "token_difference_vs_fixed": fixed_difference,
                    "token_difference_vs_uniform": uniform_difference,
                    "overflow_routes": overflow["raw"][0][
                        "request_bucket_indices"],
                    "overflow_routed_requests": overflow["raw"][0][
                        "overflow_routed_requests"],
                    "overflow_over_fixed_tps": (
                        overflow["aggregate"]["tokens_per_second_p50"] /
                        fixed["aggregate"]["tokens_per_second_p50"]),
                    "overflow_over_fixed_focus_ttft_p50": (
                        overflow["focus_ttft_p50_ms"] /
                        fixed["focus_ttft_p50_ms"]),
                    "overflow_over_fixed_focus_ttft_p95": (
                        overflow["focus_ttft_p95_ms"] /
                        fixed["focus_ttft_p95_ms"]),
                    "overflow_over_fixed_focus_completion_p50": (
                        overflow["focus_completion_p50_ms"] /
                        fixed["focus_completion_p50_ms"]),
                    "overflow_over_fixed_focus_completion_p95": (
                        overflow["focus_completion_p95_ms"] /
                        fixed["focus_completion_p95_ms"]),
                    "overflow_over_uniform_tps": (
                        overflow["aggregate"]["tokens_per_second_p50"] /
                        uniform["aggregate"]["tokens_per_second_p50"]),
                    "overflow_over_uniform_focus_ttft_p95": (
                        overflow["focus_ttft_p95_ms"] /
                        uniform["focus_ttft_p95_ms"]),
                    "overflow_over_uniform_focus_completion_p95": (
                        overflow["focus_completion_p95_ms"] /
                        uniform["focus_completion_p95_ms"]),
                    "fixed_focus_ttft_p95_ms": fixed["focus_ttft_p95_ms"],
                    "overflow_focus_ttft_p95_ms": overflow[
                        "focus_ttft_p95_ms"],
                    "uniform_focus_ttft_p95_ms": uniform[
                        "focus_ttft_p95_ms"],
                    "fixed_focus_completion_p95_ms": fixed[
                        "focus_completion_p95_ms"],
                    "overflow_focus_completion_p95_ms": overflow[
                        "focus_completion_p95_ms"],
                    "uniform_focus_completion_p95_ms": uniform[
                        "focus_completion_p95_ms"],
                })
    execution_status = "pass" if all(row["status"] == "pass" for row in rows) \
        else "complete_with_recorded_limits"
    accuracy_failures = (
        args.suite == "slot-sweep" and any(
            not row["generated_tokens_equal_across_slots"]
            for row in slot_sweeps)) or (
        args.suite == "bucket-sweep" and any(
            not row["generated_tokens_equal_across_bucket_counts"]
            for row in bucket_sweeps)) or (
        args.suite == "traffic-skew" and any(
            row["token_difference"]["exact"] is not True
            for row in traffic_comparisons)) or (
        args.suite == "traffic-overflow" and any(
            row["token_difference_vs_fixed"]["exact"] is not True or
            row["token_difference_vs_uniform"]["exact"] is not True
            for row in overflow_comparisons))
    summary = {
        "schema_version": 1,
        "track": "official_continuous_serving_matrix",
        "suite": args.suite,
        "warmup": args.warmup,
        "steps": args.steps,
        "runs": args.runs,
        "models": [model["name"] for model in models],
        "cases": SUITES[args.suite],
        "status": "complete_with_recorded_accuracy_failures"
        if execution_status == "pass" and accuracy_failures else execution_status,
        "execution_status": execution_status,
        "rows": rows,
        "aggregates": aggregates,
        "slot_sweeps": slot_sweeps,
        "bucket_comparisons": bucket_comparisons,
        "bucket_sweeps": bucket_sweeps,
        "traffic_comparisons": traffic_comparisons,
        "overflow_comparisons": overflow_comparisons,
        "pytorch_boundary": "not measured; no variable-position PyTorch serving oracle",
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
