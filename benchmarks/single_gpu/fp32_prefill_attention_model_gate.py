#!/usr/bin/env python3
"""Gate scoped exact prefill Attention solutions on the complete DeepSeek model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
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


QKV = load_module("fp32_prefill_attention_qkv", "fp32_qkv_model_gate.py")
CORE = load_module("fp32_prefill_attention_core", "audit_prefill_attention_core.py")
COMMON = QKV.COMMON
CACHE = QKV.CACHE

POLICIES = ("upstream-exact", "attention-exact")
BATCHES = (1, 2, 4, 8)
Q_SOLUTION = QKV.Q_SOLUTION
KV_SOLUTION = QKV.KV_SOLUTION
QK_SOLUTION = 304681
PV_SOLUTION = 295716


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
        parser.error("prefill Attention model-gate inputs are outside the contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, policy: str, batch: int,
            warmup: int, cache_output: Path | None = None,
            logits_output: Path | None = None,
            trace_output: Path | None = None,
            binary_directory: Path | None = None) -> list[str]:
    if policy not in POLICIES:
        raise ValueError("unknown prefill Attention model policy")
    result = QKV.command(
        args, model, "invariant-qkv", batch, warmup,
        cache_output, logits_output)
    if policy == "attention-exact":
        result.extend([
            "--fp32-prefill-attention-qk-solution-index", str(QK_SOLUTION),
            "--fp32-prefill-attention-pv-solution-index", str(PV_SOLUTION),
        ])
    if (trace_output is None) != (binary_directory is None):
        raise ValueError("trace and binary paths must be provided together")
    if trace_output is not None:
        result.extend([
            "--trace-output", str(trace_output),
            "--trace-max-elements", "1",
            "--trace-value-filter", ",".join(CORE.STAGES),
            "--trace-binary-directory", str(binary_directory),
        ])
    return result


def require_route(record: dict, policy: str, batch: int,
                  context: int, warmup: int) -> None:
    candidate = policy == "attention-exact"
    expected = {
        "status": "pass",
        "batch": batch,
        "token_count": context,
        "decode_tokens": 1,
        "kv_cache_dtype": "bf16",
        "fp32_prefill_q_solution_index": Q_SOLUTION,
        "fp32_prefill_kv_solution_index": KV_SOLUTION,
        "fp32_prefill_attention_qk_solution_index":
            QK_SOLUTION if candidate else -1,
        "fp32_prefill_attention_pv_solution_index":
            PV_SOLUTION if candidate else -1,
        "fp32_solution_registered_entries": 4 if candidate else 2,
        "fp32_solution_cached_algorithms": 4 if candidate else 2,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(
                f"{policy} B{batch} {name} expected {wanted!r}, "
                f"got {record.get(name)!r}")
    expected_dispatches = (140 if candidate else 84) * (warmup + 1)
    expected_entries = 4 if candidate else 2
    if (record.get("fp32_solution_registry_hits") != expected_dispatches or
            record.get("fp32_solution_cache_misses") != expected_entries or
            record.get("fp32_solution_cache_hits") !=
            expected_dispatches - expected_entries or
            record.get("fp32_solution_dispatches") != expected_dispatches):
        raise ValueError(f"{policy} B{batch} scoped registry counts changed")


def exact_or_difference(left: Path, right: Path, elements: int,
                        left_offset: int = 0, right_offset: int = 0) -> dict:
    byte_count = elements * 4
    with left.open("rb") as left_file, right.open("rb") as right_file:
        left_file.seek(left_offset)
        right_file.seek(right_offset)
        remaining = byte_count
        while remaining:
            size = min(4 * 1024 * 1024, remaining)
            if left_file.read(size) != right_file.read(size):
                return CORE.difference_binary(
                    left, right, elements, left_offset, right_offset)
            remaining -= size
    return {
        "elements": elements,
        "bitwise_equal": True,
        "first_bitwise_index": None,
        "first_numeric_index": None,
        "maximum": 0.0,
        "rms": 0.0,
        "relative_l2": 0.0,
    }


def compare_core(reference: dict[str, dict], actual: dict[str, dict],
                 batch: int) -> list[dict]:
    stages = []
    for name in CORE.STAGES:
        left = reference[name]
        right = actual[name]
        row_elements = math.prod(left["shape"])
        if math.prod(right["shape"]) != row_elements * min(batch, 2):
            raise ValueError(f"candidate core row shape changed: {name}")
        cross = exact_or_difference(
            left["binary_path"], right["binary_path"], row_elements)
        within = (exact_or_difference(
            right["binary_path"], right["binary_path"], row_elements,
            0, row_elements * 4) if batch > 1 else {
                "elements": row_elements,
                "bitwise_equal": True,
                "first_bitwise_index": None,
                "first_numeric_index": None,
                "maximum": 0.0,
                "rms": 0.0,
                "relative_l2": 0.0,
            })
        stages.append({
            "name": name,
            "shape_b1": left["shape"],
            "shape_actual": right["shape"],
            "b1_vs_batch_row0": cross,
            "batch_row0_vs_row1": within,
        })
    return stages


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
            core_reference = None
            retained_reference_directory = None
            for batch in BATCHES:
                process = temporary / f"precision-{policy}-b{batch}-r{run}"
                process.mkdir()
                cache_path = process / "cache.bin"
                logits_path = process / "logits.bin"
                trace_path = process / "trace.jsonl"
                binary_directory = process / "values"
                candidate = policy == "attention-exact"
                completed = subprocess.run(
                    command(
                        args, model, policy, batch, 0, cache_path, logits_path,
                        trace_path if candidate else None,
                        binary_directory if candidate else None),
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
                core_stages = []
                if candidate:
                    selected = CORE.load_binary_records(
                        trace_path, binary_directory, batch, args.context)
                    if (record.get("trace_record_count") != 54 or
                            record.get("trace_binary_record_count") != 3 or
                            record.get("trace_binary_bytes") != sum(
                                stage["binary_values"]["bytes"]
                                for stage in selected.values())):
                        raise ValueError(
                            f"candidate B{batch} core trace route changed")
                    if core_reference is None:
                        core_reference = selected
                        retained_reference_directory = process
                    core_stages = compare_core(core_reference, selected, batch)
                key_within = raw_cache["key"] == [raw_cache["key"][0]] * batch
                value_within = raw_cache["value"] == [raw_cache["value"][0]] * batch
                logit_within = all(row == logit_rows[0] for row in logit_rows)
                rows.append({
                    "schema_version": 1,
                    "record_type": "prefill_attention_model_precision_process",
                    "status": "pass",
                    "model": args.model,
                    "revision": model["revision"],
                    "policy": policy,
                    "context": args.context,
                    "batch": batch,
                    "process_run": run,
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
                    "key_within_batch_bitwise_equal": key_within,
                    "value_within_batch_bitwise_equal": value_within,
                    "logits_within_batch_bitwise_equal": logit_within,
                    "host_argmax_tokens": [COMMON.argmax(row) for row in logit_rows],
                    "device_argmax_token": int(record["generated_tokens"][0]),
                    "host_device_argmax_equal": all(
                        COMMON.argmax(row) == int(record["generated_tokens"][0])
                        for row in logit_rows),
                    "core_stages": core_stages,
                    "fp32_solution_registry_hits":
                        record["fp32_solution_registry_hits"],
                    "fp32_solution_registry_misses":
                        record["fp32_solution_registry_misses"],
                })
                print(json.dumps({
                    "phase": "precision", "policy": policy,
                    "batch": batch, "process_run": run, "status": "pass",
                }, sort_keys=True), flush=True)
                if process != retained_reference_directory:
                    shutil.rmtree(process)
            if retained_reference_directory is not None:
                shutil.rmtree(retained_reference_directory)
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
                    "record_type": "prefill_attention_model_performance_process",
                    "status": "pass",
                    "model": args.model,
                    "revision": model["revision"],
                    "policy": policy,
                    "context": args.context,
                    "batch": batch,
                    "process_run": run,
                    "prefill_ms": prefill_ms,
                    "prefill_tokens_per_second":
                        batch * args.context * 1000.0 / prefill_ms,
                    "engine_peak_bytes": int(record["engine_peak_bytes"]),
                    "engine_backend_allocation_calls": int(
                        record["engine_backend_allocation_calls"]),
                    "generated_tokens": record["generated_tokens"],
                    "fp32_solution_registry_hits":
                        record["fp32_solution_registry_hits"],
                    "fp32_solution_registry_misses":
                        record["fp32_solution_registry_misses"],
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
    policy_summaries = []
    cases = []
    for policy in POLICIES:
        policy_cases = []
        for batch in BATCHES:
            p_rows = [row for row in precision
                      if row["policy"] == policy and row["batch"] == batch]
            t_rows = [row for row in performance
                      if row["policy"] == policy and row["batch"] == batch]
            if len(p_rows) != 2 or len(t_rows) != 2:
                raise ValueError(f"{policy} B{batch} model rows are incomplete")
            stable_fields = (
                "key_cross_batch", "value_cross_batch", "logits_cross_batch",
                "key_vs_upstream", "value_vs_upstream", "logits_vs_upstream",
                "key_within_batch_bitwise_equal",
                "value_within_batch_bitwise_equal",
                "logits_within_batch_bitwise_equal", "core_stages")
            if any(p_rows[0][field] != p_rows[1][field]
                   for field in stable_fields):
                raise ValueError(f"{policy} B{batch} precision is not repeatable")
            case = {
                "policy": policy,
                "batch": batch,
                "runs": 2,
                **{field: p_rows[0][field] for field in stable_fields},
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
            policy_cases.append(case)
            cases.append(case)
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
            "all_core_cross_batch_bitwise_equal": (
                all(all(stage["b1_vs_batch_row0"]["bitwise_equal"] and
                        stage["batch_row0_vs_row1"]["bitwise_equal"]
                        for stage in row["core_stages"])
                    for row in policy_cases)
                if policy == "attention-exact" else None),
            "all_within_batch_logits_bitwise_equal": all(
                row["logits_within_batch_bitwise_equal"]
                for row in policy_cases),
            "cases": policy_cases,
        })
    upstream = {row["batch"]: row for row in cases
                if row["policy"] == "upstream-exact"}
    candidate = {row["batch"]: row for row in cases
                 if row["policy"] == "attention-exact"}
    speedups = {
        str(batch): (upstream[batch]["prefill_ms_median"] /
                     candidate[batch]["prefill_ms_median"])
        for batch in BATCHES
    }
    upstream_summary = policy_summaries[0]
    candidate_summary = policy_summaries[1]
    max_improved = (candidate_summary["maximum_logit_cross_batch_error"] <=
                    upstream_summary["maximum_logit_cross_batch_error"] * 0.9)
    rms_improved = (candidate_summary["maximum_logit_cross_batch_rms_error"] <=
                    upstream_summary["maximum_logit_cross_batch_rms_error"] * 0.9)
    performance_passed = all(value >= 0.95 for value in speedups.values())
    core_exact = candidate_summary["all_core_cross_batch_bitwise_equal"]
    return {
        "schema_version": 1,
        "record_type": "prefill_attention_model_gate",
        "status": "pass",
        "context": 2048,
        "batches": list(BATCHES),
        "runs_per_case": 2,
        "precision_process_rows": len(precision),
        "performance_process_rows": len(performance),
        "q_solution_index": Q_SOLUTION,
        "kv_solution_index": KV_SOLUTION,
        "qk_solution_index": QK_SOLUTION,
        "pv_solution_index": PV_SOLUTION,
        "candidate_prefill_speedup_by_batch": speedups,
        "candidate_minimum_prefill_speedup": min(speedups.values()),
        "candidate_core_bitwise_equal": core_exact,
        "robust_logit_max_improvement": max_improved,
        "robust_logit_rms_improvement": rms_improved,
        "performance_gate_passed": performance_passed,
        "candidate_admitted": core_exact and max_improved and rms_improved and
                              performance_passed,
        "policy_summaries": policy_summaries,
        "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1580, 700
    upstream = next(row for row in summary["policy_summaries"]
                    if row["policy"] == "upstream-exact")
    candidate = next(row for row in summary["policy_summaries"]
                     if row["policy"] == "attention-exact")
    maximum = max(
        upstream["maximum_logit_cross_batch_error"],
        candidate["maximum_logit_cross_batch_error"], 1.0e-12)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:24px;font-weight:700}.sub{font-size:12px;fill:#94a3b8}'
        '.label{font-size:12px}</style>',
        '<text x="36" y="42" class="title">Scoped exact Attention · complete model gate</text>',
        '<text x="36" y="66" class="sub">DeepSeek T2048 · two fresh processes · '
        'upstream Q/K/V fixed in both policies</text>',
    ]
    for index, batch in enumerate(BATCHES):
        x = 170 + index * 340
        base_case = next(row for row in summary["cases"]
                         if row["policy"] == "upstream-exact" and
                         row["batch"] == batch)
        candidate_case = next(row for row in summary["cases"]
                              if row["policy"] == "attention-exact" and
                              row["batch"] == batch)
        base_height = 220 * base_case["logits_cross_batch"]["maximum"] / maximum
        candidate_height = 220 * candidate_case["logits_cross_batch"]["maximum"] / maximum
        parts.extend((
            f'<text x="{x + 65}" y="120" class="label" text-anchor="middle">B{batch}</text>',
            f'<rect x="{x}" y="{390 - base_height:.2f}" width="52" '
            f'height="{max(2, base_height):.2f}" fill="#64748b"/>',
            f'<rect x="{x + 78}" y="{390 - candidate_height:.2f}" width="52" '
            f'height="{max(2, candidate_height):.2f}" fill="#22c55e"/>',
            f'<text x="{x + 65}" y="425" class="sub" text-anchor="middle">'
            f'{summary["candidate_prefill_speedup_by_batch"][str(batch)]:.3f}x prefill</text>',
        ))
    parts.extend((
        '<text x="80" y="500" class="label">grey: upstream-exact logits Max</text>',
        '<text x="390" y="500" class="label" fill="#22c55e">green: + exact QK/P×V</text>',
        f'<text x="80" y="555" class="label">core bitwise: '
        f'{str(summary["candidate_core_bitwise_equal"]).lower()}</text>',
        f'<text x="420" y="555" class="label">min prefill speed: '
        f'{summary["candidate_minimum_prefill_speedup"]:.3f}x</text>',
        f'<text x="790" y="555" class="label">admitted: '
        f'{str(summary["candidate_admitted"]).lower()}</text>',
        '<rect x="60" y="600" width="1460" height="60" rx="12" '
        'fill="#111827" stroke="#334155"/>',
        '<text x="90" y="637" class="sub">A core fix is kept only if complete logits '
        'improve and every batch keeps at least 0.95x end-to-end prefill.</text>',
        '</svg>',
    ))
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="microllm-prefill-attention-model-") as root:
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
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"cases", "policy_summaries"}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"fp32_prefill_attention_model_gate: {error}", file=sys.stderr)
        raise SystemExit(2) from error
