#!/usr/bin/env python3
"""Gate batch-selective near-default prefill Attention solutions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


QKV = load_module("fp32_attention_selective_qkv", "fp32_qkv_model_gate.py")
COMMON = QKV.COMMON
CACHE = QKV.CACHE

POLICIES = ("upstream-exact", "batch-selective")
BATCHES = (1, 2, 4, 8)
Q_SOLUTION = QKV.Q_SOLUTION
KV_SOLUTION = QKV.KV_SOLUTION
SELECTIVE = {
    1: {"qk": -1, "pv": -1},
    2: {"qk": -1, "pv": 295716},
    4: {"qk": 311274, "pv": 295716},
    8: {"qk": 311303, "pv": 292462},
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--performance-warmup", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context != 2048 or args.runs != 2 or
            args.performance_warmup != 1 or args.timeout_seconds <= 0):
        parser.error("batch-selective Attention inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, policy: str,
            batch: int, warmup: int,
            cache_output: Path | None = None,
            logits_output: Path | None = None) -> list[str]:
    if policy not in POLICIES or batch not in SELECTIVE:
        raise ValueError("unknown batch-selective policy or batch")
    result = QKV.command(
        args, model, "invariant-qkv", batch, warmup,
        cache_output, logits_output)
    if policy == "batch-selective":
        indices = SELECTIVE[batch]
        if indices["qk"] >= 0:
            result.extend([
                "--fp32-prefill-attention-qk-solution-index",
                str(indices["qk"]),
            ])
        if indices["pv"] >= 0:
            result.extend([
                "--fp32-prefill-attention-pv-solution-index",
                str(indices["pv"]),
            ])
    return result


def require_route(record: dict, policy: str, batch: int,
                  context: int, warmup: int) -> None:
    indices = SELECTIVE[batch] if policy == "batch-selective" else {
        "qk": -1, "pv": -1}
    attention_entries = int(indices["qk"] >= 0) + int(indices["pv"] >= 0)
    registered = 2 + attention_entries
    hits_per_prefill = 84 + 28 * attention_entries
    expected = {
        "status": "pass",
        "batch": batch,
        "token_count": context,
        "decode_tokens": 1,
        "kv_cache_dtype": "bf16",
        "fp32_prefill_q_solution_index": Q_SOLUTION,
        "fp32_prefill_kv_solution_index": KV_SOLUTION,
        "fp32_prefill_attention_qk_solution_index": indices["qk"],
        "fp32_prefill_attention_pv_solution_index": indices["pv"],
        "fp32_solution_registered_entries": registered,
        "fp32_solution_cached_algorithms": registered,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"{policy} B{batch} {name} expected {wanted!r}, "
                f"got {record.get(name)!r}")
    dispatches = hits_per_prefill * (warmup + 1)
    if (record.get("fp32_solution_registry_hits") != dispatches or
            record.get("fp32_solution_cache_misses") != registered or
            record.get("fp32_solution_cache_hits") != dispatches - registered or
            record.get("fp32_solution_dispatches") != dispatches):
        raise ValueError(f"{policy} B{batch} registry counts changed")


def logit_metric(left: list[float], right: list[float]) -> dict:
    maximum, rms, bitwise = COMMON.error(left, right)
    return {"elements": len(left), "maximum": maximum, "rms": rms,
            "bitwise_equal": bitwise}


def precision_phase(args: argparse.Namespace, model: dict,
                    vocabulary: int, temporary: Path) -> list[dict]:
    rows = []
    for run in range(1, args.runs + 1):
        upstream_by_batch = {}
        for policy in POLICIES:
            references = None
            for batch in BATCHES:
                cache_path = temporary / f"{policy}-b{batch}-r{run}.cache"
                logits_path = temporary / f"{policy}-b{batch}-r{run}.logits"
                completed = subprocess.run(
                    command(args, model, policy, batch, 0,
                            cache_path, logits_path),
                    text=True, capture_output=True,
                    timeout=args.timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(
                        completed.stderr.strip() or completed.stdout.strip())
                record = COMMON.last_json(completed.stdout)
                require_route(record, policy, batch, args.context, 0)
                header, raw_cache, values = CACHE.load(cache_path)
                logits = COMMON.read_logits(logits_path, batch, vocabulary)
                logit_rows = [
                    logits[index * vocabulary:(index + 1) * vocabulary]
                    for index in range(batch)]
                current = {
                    "key": values["key"][0],
                    "value": values["value"][0],
                    "logits": logit_rows[0],
                }
                if references is None:
                    references = current
                if policy == "upstream-exact":
                    upstream_by_batch[batch] = current
                upstream = upstream_by_batch[batch]
                device_token = int(record["generated_tokens"][0])
                rows.append({
                    "schema_version": 1,
                    "record_type": "selective_attention_precision_process",
                    "status": "pass",
                    "model": args.model,
                    "revision": model["revision"],
                    "policy": policy,
                    "context": args.context,
                    "batch": batch,
                    "process_run": run,
                    "qk_solution_index":
                        SELECTIVE[batch]["qk"] if policy == "batch-selective" else -1,
                    "pv_solution_index":
                        SELECTIVE[batch]["pv"] if policy == "batch-selective" else -1,
                    "cache_shape": header["shape"],
                    "key_cross_batch": CACHE.difference(
                        references["key"], current["key"]),
                    "value_cross_batch": CACHE.difference(
                        references["value"], current["value"]),
                    "logits_cross_batch": logit_metric(
                        references["logits"], current["logits"]),
                    "key_vs_upstream": CACHE.difference(
                        upstream["key"], current["key"]),
                    "value_vs_upstream": CACHE.difference(
                        upstream["value"], current["value"]),
                    "logits_vs_upstream": logit_metric(
                        upstream["logits"], current["logits"]),
                    "key_within_batch_bitwise_equal":
                        raw_cache["key"] == [raw_cache["key"][0]] * batch,
                    "value_within_batch_bitwise_equal":
                        raw_cache["value"] == [raw_cache["value"][0]] * batch,
                    "logits_within_batch_bitwise_equal":
                        all(row == logit_rows[0] for row in logit_rows),
                    "host_argmax_tokens": [COMMON.argmax(row) for row in logit_rows],
                    "device_argmax_token": device_token,
                    "host_device_argmax_equal": all(
                        COMMON.argmax(row) == device_token for row in logit_rows),
                    "fp32_solution_registry_hits":
                        record["fp32_solution_registry_hits"],
                    "fp32_solution_registry_misses":
                        record["fp32_solution_registry_misses"],
                })
                cache_path.unlink()
                logits_path.unlink()
                print(json.dumps({
                    "phase": "precision", "policy": policy,
                    "batch": batch, "process_run": run, "status": "pass",
                }, sort_keys=True), flush=True)
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
                    text=True, capture_output=True,
                    timeout=args.timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(
                        completed.stderr.strip() or completed.stdout.strip())
                record = COMMON.last_json(completed.stdout)
                require_route(
                    record, policy, batch, args.context,
                    args.performance_warmup)
                prefill_ms = float(record["mean_decode_prepare_ms"])
                rows.append({
                    "schema_version": 1,
                    "record_type": "selective_attention_performance_process",
                    "status": "pass",
                    "model": args.model,
                    "revision": model["revision"],
                    "policy": policy,
                    "context": args.context,
                    "batch": batch,
                    "process_run": run,
                    "qk_solution_index":
                        SELECTIVE[batch]["qk"] if policy == "batch-selective" else -1,
                    "pv_solution_index":
                        SELECTIVE[batch]["pv"] if policy == "batch-selective" else -1,
                    "prefill_ms": prefill_ms,
                    "prefill_tokens_per_second":
                        batch * args.context * 1000.0 / prefill_ms,
                    "engine_peak_bytes": int(record["engine_peak_bytes"]),
                    "engine_backend_allocation_calls": int(
                        record["engine_backend_allocation_calls"]),
                    "generated_tokens": record["generated_tokens"],
                })
                print(json.dumps({
                    "phase": "performance", "policy": policy,
                    "batch": batch, "process_run": run, "status": "pass",
                }, sort_keys=True), flush=True)
    return rows


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ((ordered[middle - 1] + ordered[middle]) / 2.0
            if len(ordered) % 2 == 0 else ordered[middle])


def summarize(precision: list[dict], performance: list[dict]) -> dict:
    cases = []
    policy_summaries = []
    for policy in POLICIES:
        policy_cases = []
        for batch in BATCHES:
            p_rows = [row for row in precision
                      if row["policy"] == policy and row["batch"] == batch]
            t_rows = [row for row in performance
                      if row["policy"] == policy and row["batch"] == batch]
            if len(p_rows) != 2 or len(t_rows) != 2:
                raise ValueError(f"{policy} B{batch} rows are incomplete")
            stable = (
                "qk_solution_index", "pv_solution_index",
                "key_cross_batch", "value_cross_batch", "logits_cross_batch",
                "key_vs_upstream", "value_vs_upstream", "logits_vs_upstream",
                "key_within_batch_bitwise_equal",
                "value_within_batch_bitwise_equal",
                "logits_within_batch_bitwise_equal")
            if any(p_rows[0][field] != p_rows[1][field] for field in stable):
                raise ValueError(f"{policy} B{batch} precision is not repeatable")
            case = {
                "policy": policy,
                "batch": batch,
                "runs": 2,
                **{field: p_rows[0][field] for field in stable},
                "prefill_ms_median": median(
                    [row["prefill_ms"] for row in t_rows]),
                "prefill_tokens_per_second_median": median(
                    [row["prefill_tokens_per_second"] for row in t_rows]),
                "engine_peak_bytes_maximum": max(
                    row["engine_peak_bytes"] for row in t_rows),
                "engine_backend_allocation_calls_maximum": max(
                    row["engine_backend_allocation_calls"] for row in t_rows),
                "generated_tokens_equal":
                    t_rows[0]["generated_tokens"] == t_rows[1]["generated_tokens"],
            }
            cases.append(case)
            policy_cases.append(case)
        policy_summaries.append({
            "policy": policy,
            "maximum_logit_cross_batch_error": max(
                row["logits_cross_batch"]["maximum"] for row in policy_cases),
            "maximum_logit_cross_batch_rms_error": max(
                row["logits_cross_batch"]["rms"] for row in policy_cases),
            "maximum_logit_vs_upstream_error": max(
                row["logits_vs_upstream"]["maximum"] for row in policy_cases),
            "all_cache_cross_batch_bitwise_equal": all(
                row["key_cross_batch"]["bitwise_equal"] and
                row["value_cross_batch"]["bitwise_equal"]
                for row in policy_cases),
            "all_within_batch_logits_bitwise_equal": all(
                row["logits_within_batch_bitwise_equal"] for row in policy_cases),
            "cases": policy_cases,
        })
    upstream = {row["batch"]: row for row in cases
                if row["policy"] == "upstream-exact"}
    candidate = {row["batch"]: row for row in cases
                 if row["policy"] == "batch-selective"}
    speedups = {str(batch): upstream[batch]["prefill_ms_median"] /
                            candidate[batch]["prefill_ms_median"]
                for batch in BATCHES}
    base_summary, candidate_summary = policy_summaries
    max_improved = (candidate_summary["maximum_logit_cross_batch_error"] <=
                    base_summary["maximum_logit_cross_batch_error"] * 0.9)
    rms_improved = (candidate_summary["maximum_logit_cross_batch_rms_error"] <=
                    base_summary["maximum_logit_cross_batch_rms_error"] * 0.9)
    performance_passed = all(value >= 0.95 for value in speedups.values())
    return {
        "schema_version": 1,
        "record_type": "batch_selective_attention_model_gate",
        "status": "pass",
        "context": 2048,
        "batches": list(BATCHES),
        "runs_per_case": 2,
        "precision_process_rows": len(precision),
        "performance_process_rows": len(performance),
        "q_solution_index": Q_SOLUTION,
        "kv_solution_index": KV_SOLUTION,
        "selective_indices": {str(key): value for key, value in SELECTIVE.items()},
        "candidate_prefill_speedup_by_batch": speedups,
        "candidate_minimum_prefill_speedup": min(speedups.values()),
        "robust_logit_max_improvement": max_improved,
        "robust_logit_rms_improvement": rms_improved,
        "performance_gate_passed": performance_passed,
        "candidate_admitted": max_improved and rms_improved and performance_passed,
        "policy_summaries": policy_summaries,
        "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1540, 650
    base = next(row for row in summary["policy_summaries"]
                if row["policy"] == "upstream-exact")
    candidate = next(row for row in summary["policy_summaries"]
                     if row["policy"] == "batch-selective")
    maximum = max(base["maximum_logit_cross_batch_error"],
                  candidate["maximum_logit_cross_batch_error"], 1.0e-12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:24px;font-weight:700}.sub{font-size:12px;fill:#94a3b8}'
        '.label{font-size:12px}</style>',
        '<text x="36" y="42" class="title">Batch-selective prefill Attention · model gate</text>',
        '<text x="36" y="66" class="sub">B1 default · B2 PV · B4/B8 local QK/PV winners · two fresh processes</text>',
    ]
    for index, batch in enumerate(BATCHES):
        x = 170 + index * 330
        base_case = next(row for row in summary["cases"]
                         if row["policy"] == "upstream-exact" and
                         row["batch"] == batch)
        candidate_case = next(row for row in summary["cases"]
                              if row["policy"] == "batch-selective" and
                              row["batch"] == batch)
        base_h = 210 * base_case["logits_cross_batch"]["maximum"] / maximum
        cand_h = 210 * candidate_case["logits_cross_batch"]["maximum"] / maximum
        indices = summary["selective_indices"][str(batch)]
        parts.extend((
            f'<text x="{x + 65}" y="118" class="label" text-anchor="middle">B{batch}</text>',
            f'<rect x="{x}" y="{380 - base_h:.2f}" width="52" height="{max(2, base_h):.2f}" fill="#64748b"/>',
            f'<rect x="{x + 78}" y="{380 - cand_h:.2f}" width="52" height="{max(2, cand_h):.2f}" fill="#38bdf8"/>',
            f'<text x="{x + 65}" y="415" class="sub" text-anchor="middle">{summary["candidate_prefill_speedup_by_batch"][str(batch)]:.3f}x</text>',
            f'<text x="{x + 65}" y="440" class="sub" text-anchor="middle">QK {indices["qk"]} · PV {indices["pv"]}</text>',
        ))
    parts.extend((
        f'<text x="80" y="520" class="label">max improved: {str(summary["robust_logit_max_improvement"]).lower()}</text>',
        f'<text x="420" y="520" class="label">RMS improved: {str(summary["robust_logit_rms_improvement"]).lower()}</text>',
        f'<text x="760" y="520" class="label">min speed: {summary["candidate_minimum_prefill_speedup"]:.3f}x</text>',
        f'<text x="1100" y="520" class="label">admitted: {str(summary["candidate_admitted"]).lower()}</text>',
        '<rect x="60" y="570" width="1420" height="50" rx="10" fill="#111827" stroke="#334155"/>',
        '<text x="90" y="601" class="sub">Both global logit Max/RMS must improve ≥10% and every batch must keep ≥0.95x prefill.</text>',
        '</svg>',
    ))
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="microllm-selective-attention-") as root:
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
    (args.output_directory / "selective-gate.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"cases", "policy_summaries"}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_prefill_attention_selective_gate: {error}", file=sys.stderr)
        raise SystemExit(2) from error
