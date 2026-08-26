#!/usr/bin/env python3
"""Gate invariant FP32 Q/KV solutions on DeepSeek cache, logits, and speed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "fp32_qkv_model_gate_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

CACHE_SPEC = importlib.util.spec_from_file_location(
    "fp32_qkv_model_gate_cache",
    Path(__file__).with_name("audit_prefill_cache_prefix.py"))
CACHE = importlib.util.module_from_spec(CACHE_SPEC)
assert CACHE_SPEC.loader is not None
CACHE_SPEC.loader.exec_module(CACHE)

BATCHES = (1, 2, 4, 8)
POLICIES = ("default", "invariant-qkv")
Q_SOLUTION = 296100
KV_SOLUTION = 292135


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--performance-warmup", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context <= 0 or args.runs != 2 or
            args.performance_warmup < 1 or args.timeout_seconds <= 0):
        parser.error("FP32 QKV model-gate inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, policy: str, batch: int,
            warmup: int, cache_output: Path | None = None,
            logits_output: Path | None = None) -> list[str]:
    result = [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", COMMON.expanded(model["inference"]["token_ids"], args.context),
        "--device", "hip", "--top-k", "1", "--batch", str(batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", str(args.context + 1), "--new-tokens", "1",
        "--warmup", str(warmup), "--steps", "1",
        "--prefill-warmup", str(warmup), "--prefill-steps", "1",
        "--bf16-ffn", "false", "--bf16-attention", "false",
        "--workload", "decode",
    ]
    if policy == "invariant-qkv":
        result.extend([
            "--fp32-prefill-q-solution-index", str(Q_SOLUTION),
            "--fp32-prefill-kv-solution-index", str(KV_SOLUTION),
        ])
    if cache_output is not None:
        result.extend([
            "--prefill-cache-output", str(cache_output),
            "--prefill-cache-layer", "0",
        ])
    if logits_output is not None:
        result.extend([
            "--cache-logits-output", str(logits_output),
            "--cache-logits-step", "0",
        ])
    return result


def cache_metric(left: list[float], right: list[float]) -> dict:
    return CACHE.difference(left, right)


def logit_metric(left: list[float], right: list[float]) -> dict:
    maximum, rms, bitwise = COMMON.error(left, right)
    return {"elements": len(left), "maximum": maximum, "rms": rms,
            "bitwise_equal": bitwise}


def require_route(record: dict, policy: str, batch: int, context: int,
                  warmup: int) -> None:
    candidate = policy == "invariant-qkv"
    expected = {
        "status": "pass", "batch": batch, "token_count": context,
        "decode_tokens": 1, "kv_cache_dtype": "bf16",
        "cached_attention_materialized_policy": "auto-enabled",
        "fp32_prefill_q_solution_index": Q_SOLUTION if candidate else -1,
        "fp32_prefill_kv_solution_index": KV_SOLUTION if candidate else -1,
        "fp32_solution_registered_entries": 2 if candidate else 0,
        "fp32_solution_cached_algorithms": 2 if candidate else 0,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"{policy} B{batch} {name} expected {wanted!r}, got {record.get(name)!r}")
    expected_dispatches = 84 * (warmup + 1)
    if candidate and (record.get("fp32_solution_registry_hits") !=
                          expected_dispatches or
                      record.get("fp32_solution_cache_misses") != 2 or
                      record.get("fp32_solution_cache_hits") !=
                          expected_dispatches - 2 or
                      record.get("fp32_solution_dispatches") !=
                          expected_dispatches):
        raise ValueError(f"{policy} B{batch} projection registry counts changed")


def precision_phase(args: argparse.Namespace, model: dict,
                    vocabulary: int, temporary: Path) -> list[dict]:
    rows = []
    for run in range(1, args.runs + 1):
        default_data = {}
        for policy in POLICIES:
            references = None
            for batch in BATCHES:
                cache_path = temporary / f"precision-{policy}-b{batch}-r{run}.cache"
                logits_path = temporary / f"precision-{policy}-b{batch}-r{run}.logits"
                completed = subprocess.run(
                    command(args, model, policy, batch, 0,
                            cache_path, logits_path),
                    text=True, capture_output=True, timeout=args.timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(
                        completed.stderr.strip() or completed.stdout.strip())
                record = COMMON.last_json(completed.stdout)
                require_route(record, policy, batch, args.context, 0)
                header, raw_cache, values = CACHE.load(cache_path)
                logits = COMMON.read_logits(logits_path, batch, vocabulary)
                logit_rows = [logits[index * vocabulary:(index + 1) * vocabulary]
                              for index in range(batch)]
                current = {
                    "key": values["key"][0], "value": values["value"][0],
                    "logits": logit_rows[0],
                }
                if references is None:
                    references = current
                if policy == "default":
                    default_data[batch] = current
                default = default_data.get(batch, current)
                key_within = raw_cache["key"] == [raw_cache["key"][0]] * batch
                value_within = raw_cache["value"] == [raw_cache["value"][0]] * batch
                logit_within = all(row == logit_rows[0] for row in logit_rows)
                host_tokens = [COMMON.argmax(row) for row in logit_rows]
                device_token = int(record["generated_tokens"][0])
                row = {
                    "schema_version": 1,
                    "record_type": "fp32_qkv_model_precision_process",
                    "status": "pass", "model": args.model,
                    "revision": model["revision"], "context": args.context,
                    "policy": policy, "batch": batch, "process_run": run,
                    "cache_shape": header["shape"],
                    "key_cross_batch": cache_metric(
                        references["key"], current["key"]),
                    "value_cross_batch": cache_metric(
                        references["value"], current["value"]),
                    "logits_cross_batch": logit_metric(
                        references["logits"], current["logits"]),
                    "key_vs_default": cache_metric(
                        default["key"], current["key"]),
                    "value_vs_default": cache_metric(
                        default["value"], current["value"]),
                    "logits_vs_default": logit_metric(
                        default["logits"], current["logits"]),
                    "key_within_batch_bitwise_equal": key_within,
                    "value_within_batch_bitwise_equal": value_within,
                    "logits_within_batch_bitwise_equal": logit_within,
                    "host_argmax_tokens": host_tokens,
                    "device_argmax_token": device_token,
                    "host_device_argmax_equal": all(
                        token == device_token for token in host_tokens),
                    "fp32_solution_registry_hits":
                        record["fp32_solution_registry_hits"],
                    "fp32_solution_registry_misses":
                        record["fp32_solution_registry_misses"],
                }
                rows.append(row)
                print(json.dumps({"phase": "precision", "policy": policy,
                                  "batch": batch, "process_run": run,
                                  "status": "pass"}, sort_keys=True), flush=True)
                cache_path.unlink()
                logits_path.unlink()
                del values, logits, logit_rows, current
    return rows


def performance_phase(args: argparse.Namespace, model: dict) -> list[dict]:
    rows = []
    for run in range(1, args.runs + 1):
        policy_order = list(POLICIES) if run % 2 else list(reversed(POLICIES))
        batch_order = list(BATCHES) if run % 2 else list(reversed(BATCHES))
        for policy in policy_order:
            for batch in batch_order:
                completed = subprocess.run(
                    command(args, model, policy, batch,
                            args.performance_warmup),
                    text=True, capture_output=True, timeout=args.timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(
                        completed.stderr.strip() or completed.stdout.strip())
                record = COMMON.last_json(completed.stdout)
                require_route(record, policy, batch, args.context,
                              args.performance_warmup)
                rows.append({
                    "schema_version": 1,
                    "record_type": "fp32_qkv_model_performance_process",
                    "status": "pass", "model": args.model,
                    "revision": model["revision"], "context": args.context,
                    "policy": policy, "batch": batch, "process_run": run,
                    "decode_prepare_ms": float(record["decode_prepare_ms"]),
                    "decode_tokens_per_second":
                        float(record["decode_tokens_per_second"]),
                    "engine_peak_bytes": int(record["engine_peak_bytes"]),
                    "generated_tokens": record["generated_tokens"],
                    "fp32_solution_registry_hits":
                        record["fp32_solution_registry_hits"],
                    "fp32_solution_registry_misses":
                        record["fp32_solution_registry_misses"],
                })
                print(json.dumps({"phase": "performance", "policy": policy,
                                  "batch": batch, "process_run": run,
                                  "status": "pass"}, sort_keys=True), flush=True)
    return rows


def summarize(precision: list[dict], performance: list[dict]) -> dict:
    precision_cases = []
    policy_summaries = []
    for policy in POLICIES:
        selected_cases = []
        for batch in BATCHES:
            rows = [row for row in precision
                    if row["policy"] == policy and row["batch"] == batch]
            if len(rows) != 2:
                raise ValueError("FP32 QKV precision process count changed")
            metric_names = (
                "key_cross_batch", "value_cross_batch", "logits_cross_batch",
                "key_vs_default", "value_vs_default", "logits_vs_default")
            if any(rows[0][name] != rows[1][name] for name in metric_names):
                raise ValueError(f"{policy} B{batch} precision metrics are not deterministic")
            case = {
                "policy": policy, "batch": batch, "runs": 2,
                **{name: rows[0][name] for name in metric_names},
                "key_within_batch_bitwise_equal": all(
                    row["key_within_batch_bitwise_equal"] for row in rows),
                "value_within_batch_bitwise_equal": all(
                    row["value_within_batch_bitwise_equal"] for row in rows),
                "logits_within_batch_bitwise_equal": all(
                    row["logits_within_batch_bitwise_equal"] for row in rows),
                "host_device_argmax_equal": all(
                    row["host_device_argmax_equal"] for row in rows),
                "device_argmax_tokens": [row["device_argmax_token"] for row in rows],
            }
            precision_cases.append(case)
            selected_cases.append(case)
        policy_summaries.append({
            "policy": policy,
            "maximum_key_cross_batch_error": max(
                row["key_cross_batch"]["maximum"] for row in selected_cases),
            "maximum_value_cross_batch_error": max(
                row["value_cross_batch"]["maximum"] for row in selected_cases),
            "maximum_logit_cross_batch_error": max(
                row["logits_cross_batch"]["maximum"] for row in selected_cases),
            "maximum_logit_cross_batch_rms_error": max(
                row["logits_cross_batch"]["rms"] for row in selected_cases),
            "cache_bitwise_case_count": sum(
                row["key_cross_batch"]["bitwise_equal"] and
                row["value_cross_batch"]["bitwise_equal"]
                for row in selected_cases),
            "logit_bitwise_case_count": sum(
                row["logits_cross_batch"]["bitwise_equal"]
                for row in selected_cases),
        })
    performance_cases = []
    for batch in BATCHES:
        values = {}
        for policy in POLICIES:
            rows = [row for row in performance
                    if row["policy"] == policy and row["batch"] == batch]
            if len(rows) != 2:
                raise ValueError("FP32 QKV performance process count changed")
            values[policy] = {
                "prefill_ms": statistics.median(
                    row["decode_prepare_ms"] for row in rows),
                "decode_tokens_per_second": statistics.median(
                    row["decode_tokens_per_second"] for row in rows),
                "peak_bytes": max(row["engine_peak_bytes"] for row in rows),
                "tokens": [row["generated_tokens"] for row in rows],
            }
        performance_cases.append({
            "batch": batch,
            "default": values["default"],
            "candidate": values["invariant-qkv"],
            "prefill_speedup": values["default"]["prefill_ms"] /
                values["invariant-qkv"]["prefill_ms"],
            "decode_throughput_ratio":
                values["invariant-qkv"]["decode_tokens_per_second"] /
                values["default"]["decode_tokens_per_second"],
            "peak_delta_bytes": values["invariant-qkv"]["peak_bytes"] -
                values["default"]["peak_bytes"],
        })
    return {
        "schema_version": 1,
        "record_type": "fp32_qkv_complete_model_gate",
        "status": "pass", "context": 2048,
        "batches": list(BATCHES), "policies": list(POLICIES),
        "q_solution_index": Q_SOLUTION, "kv_solution_index": KV_SOLUTION,
        "precision_process_rows": len(precision),
        "performance_process_rows": len(performance),
        "all_precision_repeat_metrics_equal": True,
        "all_host_device_argmax_equal": all(
            row["host_device_argmax_equal"] for row in precision),
        "policy_summaries": policy_summaries,
        "precision_cases": precision_cases,
        "performance_cases": performance_cases,
    }


def render(summary: dict) -> str:
    width, height = 1500, 610
    policies = {row["policy"]: row for row in summary["policy_summaries"]}
    maximum = max(row["maximum_logit_cross_batch_error"]
                  for row in policies.values())
    scale = 700.0 / maximum if maximum else 1.0
    colors = {"default": "#f97316", "invariant-qkv": "#22c55e"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:12px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">FP32 invariant Q/KV complete-model gate</text>',
        '<text x="30" y="62" class="muted">DeepSeek T2048 · raw BF16 cache + '
        '151,936 logits + no-export performance</text>',
    ]
    for policy_index, policy in enumerate(POLICIES):
        y0 = 105 + policy_index * 170
        parts.append(f'<text x="30" y="{y0 + 18}" class="label">{policy}</text>')
        rows = [row for row in summary["precision_cases"]
                if row["policy"] == policy]
        for index, row in enumerate(rows):
            y = y0 + index * 32
            length = max(2.0, row["logits_cross_batch"]["maximum"] * scale)
            parts.extend((
                f'<text x="190" y="{y + 18}" class="label">B{row["batch"]}</text>',
                f'<rect x="230" y="{y}" width="{length:.2f}" height="22" rx="4" '
                f'fill="{colors[policy]}"/>',
                f'<text x="{245 + length:.2f}" y="{y + 17}" class="label">'
                f'logit Max {row["logits_cross_batch"]["maximum"]:.3e} · '
                f'K {row["key_cross_batch"]["maximum"]:.3e} · '
                f'V {row["value_cross_batch"]["maximum"]:.3e}</text>',
            ))
    y = 480
    for index, row in enumerate(summary["performance_cases"]):
        x = 230 + index * 280
        parts.append(f'<text x="{x}" y="{y}" class="label">B{row["batch"]} '
                     f'prefill {row["prefill_speedup"]:.3f}x · '
                     f'decode {row["decode_throughput_ratio"]:.3f}x · '
                     f'peak {row["peak_delta_bytes"]}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="microllm-fp32-qkv-model-gate-") as root:
        precision = precision_phase(args, model, vocabulary, Path(root))
    performance = performance_phase(args, model)
    summary = summarize(precision, performance)
    (args.output_directory / "precision-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in precision),
        encoding="utf-8")
    (args.output_directory / "performance-raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in performance),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "model-gate.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_qkv_model_gate: {error}", file=sys.stderr)
        raise SystemExit(2) from error
