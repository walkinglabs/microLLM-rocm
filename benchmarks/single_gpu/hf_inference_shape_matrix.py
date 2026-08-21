#!/usr/bin/env python3
"""Paired official-model inference matrix with explicit KV-cache evidence.

Every timed row runs in a fresh process.  The parent alternates framework order,
keeps failures/OOM/unsupported shapes as data, and computes medians only from
complete measured pairs.  PyTorch worker mode lives in this file so its exact
shape, cache and memory contract cannot silently drift from the orchestrator.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


MATRIX_SUITES = {
    "smoke": {"contexts": [8, 128], "batches": [1, 2]},
    "standard": {"contexts": [8, 32, 128, 512, 2048],
                 "batches": [1, 2, 4, 8]},
    "extended": {"contexts": [1, 8, 32, 128, 512, 1024, 2048, 4096],
                 "batches": [1, 2, 4, 8, 16]},
}


def positive_int_list(text: str, name: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    if not values or any(value <= 0 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{name} must contain unique positive integers")
    return values


def name_list(text: str) -> list[str]:
    values = text.split(",")
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("models must be unique non-empty names")
    return values


def case_list(text: str) -> list[str]:
    values = name_list(text)
    allowed = {"prefill", "cached", "uncached"}
    if not set(values) <= allowed:
        raise argparse.ArgumentTypeError(
            "cases must contain prefill, cached and/or uncached")
    return values


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--micro-binary", required=True, type=Path)
    parser.add_argument("--pytorch-python", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--models", type=name_list)
    parser.add_argument("--suite", choices=tuple(MATRIX_SUITES), default="standard")
    parser.add_argument("--contexts")
    parser.add_argument("--batches")
    parser.add_argument("--cases", type=case_list,
                        default=case_list("prefill,cached,uncached"))
    parser.add_argument("--micro-batch-argmax-mode", choices=("host", "device"),
                        default="device")
    parser.add_argument("--micro-kv-cache-dtype", choices=("fp32", "bf16"),
                        default="fp32")
    parser.add_argument("--micro-kv-cache-fp32-layers", default="")
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument("--worker-context", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-workload", choices=("prefill", "decode"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-cache", choices=("cached", "uncached"),
                        help=argparse.SUPPRESS)
    result = parser.parse_args()
    selected = MATRIX_SUITES[result.suite]
    result.contexts = (positive_int_list(result.contexts, "contexts")
                       if result.contexts else list(selected["contexts"]))
    result.batches = (positive_int_list(result.batches, "batches")
                      if result.batches else list(selected["batches"]))
    for path in (result.manifest, result.micro_binary, result.pytorch_python):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if result.decode_tokens <= 0 or result.warmup < 0 or result.steps <= 0 or \
            result.runs <= 0 or result.timeout_seconds <= 0:
        parser.error("decode-tokens/steps/runs/timeout must be positive; warmup nonnegative")
    return result


def load_models(path: Path, selected: list[str] | None = None) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list) or not models:
        raise RuntimeError("manifest must contain a non-empty schema-version-1 models list")
    by_name = {model.get("name"): model for model in models}
    if None in by_name or len(by_name) != len(models):
        raise RuntimeError("model names must be present and unique")
    names = selected or list(by_name)
    unknown = set(names) - set(by_name)
    if unknown:
        raise RuntimeError(f"unknown selected models: {sorted(unknown)}")
    for name in names:
        model = by_name[name]
        missing = {"revision", "config", "weights", "parameter_count", "inference"} - model.keys()
        if missing:
            raise RuntimeError(f"{name} is missing fields: {sorted(missing)}")
        tokens = model["inference"].get("token_ids")
        if not isinstance(tokens, list) or not tokens or any(int(token) < 0 for token in tokens):
            raise RuntimeError(f"{name} needs nonnegative inference.token_ids")
    return [by_name[name] for name in names]


def expanded_tokens(seed: list[int], context: int) -> list[int]:
    return [int(seed[index % len(seed)]) for index in range(context)]


def theoretical_kv_cache_bytes(layers: int, kv_heads: int, head_dimension: int,
                               batch: int, tokens: int, element_bytes: int) -> int:
    return 2 * layers * kv_heads * head_dimension * batch * tokens * element_bytes


def classify_failure(error: Exception | str) -> str:
    text = str(error).lower()
    if "out of memory" in text or "memory allocation" in text:
        return "oom"
    if "currently supports batch 1" in text or "unsupported" in text:
        return "unsupported"
    return "failed"


def run_one_json(command: list[str], timeout: int) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"exit {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"command emitted {len(lines)} JSON lines instead of one")
    return json.loads(lines[0])


def validate_measurement(record: dict, model: dict, framework: str, context: int,
                         batch: int, workload: str, cache: str,
                         warmup: int, steps: int, decode_tokens: int) -> None:
    required_equal = {
        "status": "pass", "model": model["name"], "parameter_count": model["parameter_count"],
        "context": context, "batch": batch, "workload": workload,
        "cache_mode": cache, "warmup": warmup, "steps": steps,
    }
    if any(record.get(key) != value for key, value in required_equal.items()):
        raise RuntimeError(f"{model['name']} {framework} violated inference shape contract")
    expected_precision = ("mixed_bf16_weights_fp32_activations"
                          if framework == "microllm" else "full_bf16_model")
    if record.get("precision") != expected_precision:
        raise RuntimeError(f"{model['name']} {framework} did not report its precision policy")
    if int(record.get("peak_bytes", 0)) <= 0 or \
            int(record.get("device_total_bytes", 0)) <= 0 or \
            int(record.get("resident_weight_bytes", 0)) <= 0 or \
            not math.isfinite(float(record.get("throughput_tokens_per_second", math.nan))) or \
            float(record["throughput_tokens_per_second"]) <= 0:
        raise RuntimeError(f"{model['name']} {framework} has invalid timing/memory evidence")
    if workload == "decode":
        if int(record.get("measured_tokens", -1)) != batch * decode_tokens * steps:
            raise RuntimeError(f"{model['name']} {framework} decode token accounting changed")
        expected = int(record.get("kv_cache_theoretical_bytes", -1))
        actual = int(record.get("kv_cache_actual_bytes", -1))
        active = int(record.get("kv_cache_active_bytes", -1))
        utilization = float(record.get("kv_cache_utilization", 0.0))
        if cache == "cached" and (expected <= 0 or actual <= 0 or expected != actual or
                                   active <= 0 or active > actual or
                                   not (0.0 < utilization <= 1.0)):
            raise RuntimeError(f"{model['name']} {framework} omitted cached KV bytes")
        if cache == "uncached" and (expected != 0 or actual != 0):
            raise RuntimeError(f"{model['name']} {framework} allocated a disabled KV cache")
    else:
        top = record.get("top_logits", [])
        if not top or not math.isfinite(float(top[0].get("logit", math.nan))):
            raise RuntimeError(f"{model['name']} {framework} omitted prefill logit evidence")


def micro_command(args: argparse.Namespace, model: dict, context: int, batch: int,
                  workload: str, cache: str) -> list[str]:
    ids = expanded_tokens(model["inference"]["token_ids"], context)
    command = [
        str(args.micro_binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in ids), "--device", "hip",
        "--top-k", "1", "--batch", str(batch), "--use-cache", str(cache == "cached").lower(),
        "--cache-prefill-mode", "full",
        "--batch-argmax-mode", args.micro_batch_argmax_mode,
        "--kv-cache-dtype", args.micro_kv_cache_dtype,
        "--new-tokens", str(args.decode_tokens if workload == "decode" else 0),
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--prefill-warmup", str(args.warmup), "--prefill-steps", str(args.steps),
        "--bf16-ffn", "true", "--bf16-attention", "true", "--workload", workload,
    ]
    fp32_layers = getattr(args, "micro_kv_cache_fp32_layers", "")
    if fp32_layers:
        command.extend(["--kv-cache-fp32-layers",
                        fp32_layers])
    return command


def normalize_micro(raw: dict, model: dict, context: int, batch: int,
                    workload: str, cache: str, args: argparse.Namespace) -> dict:
    throughput = (raw["prefill_tokens_per_second"] if workload == "prefill"
                  else raw["decode_tokens_per_second"])
    actual = int(raw.get("kv_cache_actual_bytes", 0)) if cache == "cached" else 0
    element_bytes = int(raw.get("kv_cache_element_bytes", 0)) if cache == "cached" else 0
    capacity_tokens = int(raw.get("kv_cache_capacity_tokens", 0)) if cache == "cached" else 0
    if cache == "cached" and element_bytes == 0:
        theoretical = (2 * int(raw.get("kv_cache_heads", 0)) *
                       int(raw.get("kv_cache_head_dimension", 0)) * batch *
                       capacity_tokens *
                       (4 * int(raw.get("kv_cache_fp32_layers", 0)) +
                        2 * int(raw.get("kv_cache_bf16_layers", 0))))
    else:
        theoretical = theoretical_kv_cache_bytes(
            int(raw.get("kv_cache_layers", 0)), int(raw.get("kv_cache_heads", 0)),
            int(raw.get("kv_cache_head_dimension", 0)), batch, capacity_tokens,
            element_bytes) if cache == "cached" else 0
    return {
        **raw, "schema_version": 1, "record_type": "official_inference_shape_measurement",
        "framework": "microllm", "model": model["name"], "revision": model["revision"],
        "status": "pass", "context": context, "batch": batch, "workload": workload,
        "cache_mode": cache, "precision": "mixed_bf16_weights_fp32_activations",
        "kv_cache_dtype": raw.get(
            "kv_cache_dtype", getattr(args, "micro_kv_cache_dtype", "fp32")),
        "kv_cache_fp32_layer_policy": raw.get("kv_cache_fp32_layer_policy", ""),
        "kv_cache_fp32_layers": int(raw.get("kv_cache_fp32_layers", 0)),
        "kv_cache_bf16_layers": int(raw.get("kv_cache_bf16_layers", 0)),
        "precision_policy": raw.get("inference_weight_policy"),
        "warmup": args.warmup,
        "steps": args.steps, "decode_tokens": args.decode_tokens,
        "throughput_tokens_per_second": throughput,
        "latency_ms": raw["forward_ms"] if workload == "prefill" else raw["mean_generation_ms"],
        "mean_cache_prepare_ms": float(raw.get("mean_cache_prepare_ms", 0.0)),
        "mean_end_to_end_generation_ms": float(
            raw.get("mean_end_to_end_generation_ms",
                    raw.get("forward_ms" if workload == "prefill"
                            else "mean_generation_ms", 0.0))),
        "peak_bytes": raw["engine_peak_bytes"],
        "device_total_bytes": raw["device_total_bytes"],
        "peak_memory_share_of_device": raw["engine_peak_share_of_device"],
        "kv_cache_actual_bytes": actual, "kv_cache_theoretical_bytes": theoretical,
        "kv_cache_allocation_efficiency": theoretical / actual if actual else 0.0,
        "kv_cache_utilization": float(raw.get("kv_cache_utilization", 0.0)),
        "kv_cache_share_of_peak": actual / raw["engine_peak_bytes"] if actual else 0.0,
        "measured_tokens": (context * batch * args.steps if workload == "prefill"
                            else raw["measured_tokens"]),
    }


def torch_device(allow_fallback: bool):
    import torch
    workaround = "none"
    if torch.version.hip and torch.cuda.device_count() == 0 and torch._C._cuda_getDeviceCount() > 0:
        if not allow_fallback:
            raise RuntimeError("AMDSMI reports zero devices; enable the explicit fallback")
        torch.cuda._device_count_amdsmi = lambda: -1
        workaround = "amdsmi_zero_fallback_to_hip_runtime"
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device unavailable")
    return torch.device("cuda:0"), workaround


def cache_tensors(past) -> list:
    if past is None:
        return []
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    return [tensor for layer in past for tensor in layer[:2]]


def pytorch_worker(args: argparse.Namespace, model: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    device, workaround = torch_device(args.allow_amdsmi_fallback)
    context, batch = args.worker_context, args.worker_batch
    workload, cache = args.worker_workload, args.worker_cache
    ids = expanded_tokens(model["inference"]["token_ids"], context)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device).repeat(batch, 1)
    gc.collect()
    torch.cuda.empty_cache()
    loaded = AutoModelForCausalLM.from_pretrained(
        Path(model["config"]).parent, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").to(device).eval()
    torch.cuda.synchronize(device)
    parameter_count = sum(parameter.numel() for parameter in loaded.parameters())
    if parameter_count != model["parameter_count"]:
        raise RuntimeError("PyTorch parameter count changed")

    def prefill_once():
        return loaded(input_ids=input_ids, use_cache=False).logits

    def decode_once(use_cache: bool) -> tuple[list[int], object | None]:
        suffix = []
        if use_cache:
            prepared = loaded(input_ids=input_ids, use_cache=True)
            logits = prepared.logits[:, -1, :]
            past = prepared.past_key_values
            for generated in range(args.decode_tokens):
                selected = torch.argmax(logits, dim=-1)
                suffix.append(selected)
                if generated + 1 < args.decode_tokens:
                    current = loaded(
                        input_ids=selected[:, None], past_key_values=past,
                        use_cache=True)
                    logits = current.logits[:, -1, :]
                    past = current.past_key_values
            stacked = torch.stack(suffix, dim=1)
            rows = stacked.tolist()
            if any(row != rows[0] for row in rows):
                raise RuntimeError("identical PyTorch batch rows generated different tokens")
            return rows[0], past
        sequence = input_ids
        for _ in range(args.decode_tokens):
            logits = loaded(input_ids=sequence, use_cache=False).logits[:, -1, :]
            selected = torch.argmax(logits, dim=-1)
            suffix.append(selected)
            sequence = torch.cat((sequence, selected[:, None]), dim=1)
        stacked = torch.stack(suffix, dim=1)
        rows = stacked.tolist()
        if any(row != rows[0] for row in rows):
            raise RuntimeError("identical PyTorch batch rows generated different tokens")
        return rows[0], None

    with torch.inference_mode():
        for _ in range(args.warmup):
            if workload == "prefill":
                prefill_once()
            else:
                decode_once(cache == "cached")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        final = None
        final_past = None
        first_suffix = None
        elapsed_ms = 0.0
        cache_prepare_ms = 0.0
        for _ in range(args.steps):
            if workload == "prefill":
                start = time.perf_counter()
                final = prefill_once()
                torch.cuda.synchronize(device)
                elapsed_ms += (time.perf_counter() - start) * 1000.0
            elif cache == "cached":
                # Build the prompt cache outside the decode timer, matching the
                # microLLM prefill/decode phase boundary.
                prepare_start = time.perf_counter()
                prepared = loaded(input_ids=input_ids, use_cache=True)
                logits = prepared.logits[:, -1, :]
                past = prepared.past_key_values
                torch.cuda.synchronize(device)
                cache_prepare_ms += (time.perf_counter() - prepare_start) * 1000.0
                start = time.perf_counter()
                suffix_tensors = []
                for generated in range(args.decode_tokens):
                    selected = torch.argmax(logits, dim=-1)
                    suffix_tensors.append(selected)
                    if generated + 1 < args.decode_tokens:
                        current = loaded(
                            input_ids=selected[:, None], past_key_values=past,
                            use_cache=True)
                        logits = current.logits[:, -1, :]
                        past = current.past_key_values
                torch.cuda.synchronize(device)
                elapsed_ms += (time.perf_counter() - start) * 1000.0
                stacked = torch.stack(suffix_tensors, dim=1)
                suffixes = stacked.tolist()
                final_past = past
                if any(row != suffixes[0] for row in suffixes):
                    raise RuntimeError("identical PyTorch batch rows generated different tokens")
                if first_suffix is not None and suffixes[0] != first_suffix:
                    raise RuntimeError("PyTorch generation changed across measured steps")
                first_suffix = suffixes[0]
            else:
                start = time.perf_counter()
                suffix, final_past = decode_once(False)
                torch.cuda.synchronize(device)
                elapsed_ms += (time.perf_counter() - start) * 1000.0
                if first_suffix is not None and suffix != first_suffix:
                    raise RuntimeError("PyTorch generation changed across measured steps")
                first_suffix = suffix
    measured = (context * batch * args.steps if workload == "prefill"
                else args.decode_tokens * batch * args.steps)
    tensors = cache_tensors(final_past) \
        if workload == "decode" and cache == "cached" else []
    actual = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if tensors:
        active_tokens = int(tensors[0].shape[-2])
        element_bytes = tensors[0].element_size()
    else:
        active_tokens = 0
        element_bytes = 0
    config = loaded.config
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dimension = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    theoretical = theoretical_kv_cache_bytes(
        config.num_hidden_layers, kv_heads, head_dimension, batch,
        active_tokens, element_bytes) if tensors else 0
    peak = torch.cuda.max_memory_allocated(device)
    device_total = int(torch.cuda.get_device_properties(device).total_memory)
    top_logits = []
    if workload == "prefill":
        value, token = torch.max(final[0, -1].float(), dim=0)
        top_logits = [{"token": int(token.item()), "logit": float(value.item())}]
    return {
        "schema_version": 1, "record_type": "official_inference_shape_measurement",
        "framework": "pytorch", "status": "pass", "model": model["name"],
        "revision": model["revision"], "parameter_count": parameter_count,
        "device": str(device), "device_discovery_workaround": workaround,
        "precision": "full_bf16_model", "precision_policy": "full_model_bf16",
        "resident_weight_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in loaded.parameters()),
        "context": context, "batch": batch, "workload": workload, "cache_mode": cache,
        "warmup": args.warmup, "steps": args.steps, "decode_tokens": args.decode_tokens,
        "latency_ms": elapsed_ms / args.steps,
        "mean_cache_prepare_ms": cache_prepare_ms / args.steps,
        "mean_end_to_end_generation_ms":
            (cache_prepare_ms + elapsed_ms) / args.steps,
        "measured_tokens": measured,
        "throughput_tokens_per_second": measured * 1000.0 / elapsed_ms,
        "peak_bytes": peak, "device_total_bytes": device_total,
        "peak_memory_share_of_device": peak / device_total,
        "kv_cache_actual_bytes": actual, "kv_cache_theoretical_bytes": theoretical,
        "kv_cache_active_bytes": actual,
        "kv_cache_capacity_tokens": active_tokens,
        "kv_cache_active_tokens": active_tokens,
        "kv_cache_element_bytes": element_bytes,
        "kv_cache_allocation_efficiency": theoretical / actual if actual else 0.0,
        "kv_cache_utilization": 1.0 if actual else 0.0,
        "kv_cache_share_of_peak": actual / peak if actual else 0.0,
        "top_logits": top_logits,
        "generated_tokens": first_suffix or [],
    }


def summarize(records: list[dict], models: list[dict], contexts: list[int],
              batches: list[int], runs: int,
              cases: list[str] | tuple[str, ...] = ("prefill", "cached", "uncached"),
              micro_kv_cache_dtype: str = "fp32",
              micro_kv_cache_fp32_layers: str = "") -> dict:
    rows = []
    case_pairs = {
        "prefill": ("prefill", "uncached"),
        "cached": ("decode", "cached"),
        "uncached": ("decode", "uncached"),
    }
    for model in models:
        for context in contexts:
            for batch in batches:
                for case in cases:
                    workload, cache = case_pairs[case]
                    selected = [record for record in records
                                if record.get("model") == model["name"] and
                                record.get("context") == context and
                                record.get("batch") == batch and
                                record.get("workload") == workload and
                                record.get("cache_mode") == cache]
                    row = {"model": model["name"], "revision": model["revision"],
                           "context": context, "batch": batch, "workload": workload,
                           "cache_mode": cache}
                    per_framework = {}
                    for framework in ("microllm", "pytorch"):
                        measured = [record for record in selected
                                    if record.get("framework") == framework and
                                    record.get("status") == "pass"]
                        failures = [record for record in selected
                                    if record.get("framework") == framework and
                                    record.get("status") != "pass"]
                        if len(measured) == runs:
                            per_framework[framework] = measured
                            row[f"{framework}_status"] = "pass"
                            for field in ("throughput_tokens_per_second", "latency_ms",
                                          "peak_bytes", "resident_weight_bytes",
                                          "device_total_bytes",
                                          "peak_memory_share_of_device",
                                          "kv_cache_actual_bytes",
                                          "kv_cache_theoretical_bytes",
                                          "kv_cache_allocation_efficiency",
                                          "kv_cache_utilization",
                                          "kv_cache_share_of_peak",
                                          "mean_cache_prepare_ms",
                                          "mean_end_to_end_generation_ms"):
                                row[f"{framework}_{field}"] = statistics.median(
                                    float(record.get(field, 0.0)) for record in measured)
                            if workload == "decode":
                                row[f"{framework}_generated_tokens"] = measured[0][
                                    "generated_tokens"]
                            peak = row[f"{framework}_peak_bytes"]
                            throughput = row[f"{framework}_throughput_tokens_per_second"]
                            row[f"{framework}_peak_bytes_per_request"] = peak / batch
                            row[f"{framework}_kv_cache_bytes_per_request"] = \
                                row[f"{framework}_kv_cache_actual_bytes"] / batch
                            row[f"{framework}_throughput_per_peak_gib"] = \
                                throughput / (peak / (1024.0 ** 3))
                        else:
                            statuses = sorted({record.get("status", "failed")
                                               for record in failures})
                            row[f"{framework}_status"] = statuses[0] if len(statuses) == 1 \
                                else "incomplete"
                    if len(per_framework) == 2:
                        micro_tps = row["microllm_throughput_tokens_per_second"]
                        torch_tps = row["pytorch_throughput_tokens_per_second"]
                        row["throughput_ratio_microllm_over_pytorch"] = micro_tps / torch_tps
                        row["peak_memory_ratio_microllm_over_pytorch"] = \
                            row["microllm_peak_bytes"] / row["pytorch_peak_bytes"]
                        row["resident_weight_ratio_microllm_over_pytorch"] = \
                            row["microllm_resident_weight_bytes"] / \
                            row["pytorch_resident_weight_bytes"]
                        if workload == "decode":
                            row["cross_framework_tokens_equal"] = \
                                row["microllm_generated_tokens"] == row["pytorch_generated_tokens"]
                        else:
                            micro_top = per_framework["microllm"][0].get("top_logits", [])
                            torch_top = per_framework["pytorch"][0].get("top_logits", [])
                            if micro_top and torch_top:
                                row["prefill_top_token_equal"] = \
                                    micro_top[0]["token"] == torch_top[0]["token"]
                                row["prefill_top_logit_abs_difference"] = abs(
                                    float(micro_top[0]["logit"]) -
                                    float(torch_top[0]["logit"]))
                    row["status"] = "pass" if len(per_framework) == 2 else "limited"
                    rows.append(row)

    for framework in ("microllm", "pytorch"):
        batch_one = {
            (row["model"], row["context"], row["workload"], row["cache_mode"]): row
            for row in rows if row["batch"] == 1 and
            row.get(f"{framework}_status") == "pass"
        }
        for row in rows:
            if row.get(f"{framework}_status") != "pass":
                continue
            baseline = batch_one.get((row["model"], row["context"], row["workload"],
                                      row["cache_mode"]))
            if baseline is None:
                continue
            scaling = row[f"{framework}_throughput_tokens_per_second"] / \
                baseline[f"{framework}_throughput_tokens_per_second"]
            row[f"{framework}_batch_throughput_scaling"] = scaling
            row[f"{framework}_batch_efficiency"] = scaling / row["batch"]
            row[f"{framework}_peak_memory_scaling"] = \
                row[f"{framework}_peak_bytes"] / baseline[f"{framework}_peak_bytes"]

        by_key = {(row["model"], row["context"], row["batch"], row["cache_mode"]): row
                  for row in rows if row["workload"] == "decode" and
                  row.get(f"{framework}_status") == "pass"}
        for row in rows:
            if row["workload"] != "decode" or row["cache_mode"] != "cached":
                continue
            other = by_key.get((row["model"], row["context"], row["batch"], "uncached"))
            if other is not None and row.get(f"{framework}_status") == "pass":
                row[f"{framework}_cache_tokens_equal"] = \
                    row[f"{framework}_generated_tokens"] == other[f"{framework}_generated_tokens"]
                row[f"{framework}_cache_speedup"] = \
                    row[f"{framework}_throughput_tokens_per_second"] / \
                    other[f"{framework}_throughput_tokens_per_second"]
    return {
        "schema_version": 1, "track": "official_inference_shape_matrix",
        "axes": {"contexts": contexts, "batches": batches,
                 "cases": list(cases)},
        "precision_boundary": {
            "microllm": "mixed_bf16_weights_fp32_activations; cache=" +
                         micro_kv_cache_dtype + "; fp32_layers=" +
                         micro_kv_cache_fp32_layers,
            "pytorch": "full_bf16_model"
        }, "runs_per_framework": runs,
        "pairing": "fresh processes; framework order alternates by run",
        "status": "pass" if all(row["status"] == "pass" for row in rows)
        else "complete_with_recorded_limits", "rows": rows,
    }


def main() -> int:
    args = options()
    models = load_models(args.manifest, args.models)
    if args.worker_model is not None:
        model = next((item for item in models if item["name"] == args.worker_model), None)
        if model is None or None in (args.worker_context, args.worker_batch,
                                     args.worker_workload, args.worker_cache):
            raise RuntimeError("worker mode requires a known model and complete case")
        print(json.dumps(pytorch_worker(args, model), sort_keys=True))
        return 0

    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    records = []

    def save(record: dict) -> None:
        records.append(record)
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    for model in models:
        for context in args.contexts:
            for batch in args.batches:
                case_pairs = {
                    "prefill": ("prefill", "uncached"),
                    "cached": ("decode", "cached"),
                    "uncached": ("decode", "uncached"),
                }
                for case in args.cases:
                    workload, cache = case_pairs[case]
                    for process_run in range(1, args.runs + 1):
                        order = ("microllm", "pytorch") if process_run % 2 else \
                            ("pytorch", "microllm")
                        for framework in order:
                            base = {
                                "schema_version": 1,
                                "record_type": "official_inference_shape_measurement",
                                "framework": framework, "model": model["name"],
                                "revision": model["revision"], "context": context,
                                "batch": batch, "workload": workload, "cache_mode": cache,
                                "precision": "bf16", "process_run": process_run,
                                "pair_order": list(order),
                            }
                            try:
                                if context + (args.decode_tokens if workload == "decode" else 0) > \
                                        int(json.loads(Path(model["config"]).read_text(
                                            encoding="utf-8"))["max_position_embeddings"]):
                                    raise RuntimeError("unsupported: sequence exceeds model context")
                                if framework == "microllm":
                                    raw = run_one_json(micro_command(
                                        args, model, context, batch, workload, cache),
                                        args.timeout_seconds)
                                    record = normalize_micro(
                                        raw, model, context, batch, workload, cache, args)
                                else:
                                    command = [
                                        str(args.pytorch_python), str(Path(__file__).resolve()),
                                        "--manifest", str(args.manifest),
                                        "--micro-binary", str(args.micro_binary),
                                        "--pytorch-python", str(args.pytorch_python),
                                        "--output-directory", str(args.output_directory),
                                        "--models", model["name"], "--contexts", str(context),
                                        "--batches", str(batch), "--decode-tokens",
                                        str(args.decode_tokens), "--warmup", str(args.warmup),
                                        "--steps", str(args.steps), "--runs", str(args.runs),
                                        "--timeout-seconds", str(args.timeout_seconds),
                                        "--worker-model", model["name"],
                                        "--worker-context", str(context),
                                        "--worker-batch", str(batch),
                                        "--worker-workload", workload,
                                        "--worker-cache", cache,
                                    ]
                                    if args.allow_amdsmi_fallback:
                                        command.append("--allow-amdsmi-fallback")
                                    record = run_one_json(command, args.timeout_seconds)
                                validate_measurement(record, model, framework, context, batch,
                                                     workload, cache, args.warmup, args.steps,
                                                     args.decode_tokens)
                                record.update({"process_run": process_run,
                                               "pair_order": list(order)})
                            except Exception as error:
                                record = {**base, "status": classify_failure(error),
                                          "error": str(error)}
                            save(record)
    summary = summarize(records, models, args.contexts, args.batches, args.runs,
                        args.cases, args.micro_kv_cache_dtype,
                        args.micro_kv_cache_fp32_layers)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
