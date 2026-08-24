#!/usr/bin/env python3
"""Validate optimization records, local links and generated SVG assets."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
EXPECTED_COLUMNS = [
    "experiment", "commit", "date", "status", "score", "qwen_train",
    "qwen_generate", "deepseek_train", "deepseek_generate", "qwen_train_mem",
    "qwen_generate_mem", "deepseek_train_mem", "deepseek_generate_mem", "description",
]
STATUSES = {"baseline", "keep", "discard", "crash", "invalid"}
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
BF16_POLICY_COLUMNS = [
    "model", "fp32_tokens_per_second", "bf16_policy_tokens_per_second",
    "throughput_ratio", "fp32_engine_bytes", "bf16_engine_bytes",
    "extra_engine_bytes", "exact_tokens", "decision",
]


def validate_results(errors: list[str]) -> int:
    path = ROOT / "results.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != EXPECTED_COLUMNS:
            errors.append("results.tsv header does not match SCHEMA.md")
            return 0
        rows = list(reader)
    seen: set[int] = set()
    for row in rows:
        experiment = int(row["experiment"])
        if experiment in seen:
            errors.append(f"duplicate experiment: {experiment:03d}")
        seen.add(experiment)
        if row["status"] not in STATUSES:
            errors.append(f"invalid status in experiment {experiment:03d}")
        values = [float(row[name]) for name in EXPECTED_COLUMNS[4:13]]
        if any(not math.isfinite(value) or value < 0 for value in values):
            errors.append(f"non-finite/negative metric in experiment {experiment:03d}")
        ratios = [float(row[name]) for name in
                  ("qwen_train", "qwen_generate", "deepseek_train", "deepseek_generate")]
        expected_score = math.prod(ratios) ** 0.25 if all(value > 0 for value in ratios) else 0.0
        if abs(float(row["score"]) - expected_score) > 1.0e-6:
            errors.append(f"score mismatch in experiment {experiment:03d}")
        if "\t" in row["description"] or not row["description"].strip():
            errors.append(f"invalid description in experiment {experiment:03d}")
    ordered = [int(row["experiment"]) for row in rows]
    if ordered != sorted(ordered):
        errors.append("experiment numbers are not strictly increasing")
    return len(rows)


def validate_steps(errors: list[str]) -> int:
    expected = [ROOT / "steps" / f"{index:02d}-{name}.md" for index, name in enumerate((
        "baseline", "parallel-cross-entropy", "transpose-aware-gemm", "parallel-rmsnorm",
        "device-kv-cache", "device-sampling", "memory-pool", "autograd-buffers",
        "batched-fmha", "fusion-autotune", "bf16", "fp8", "hip-graph-final",
    ))]
    for path in expected:
        if not path.is_file():
            errors.append(f"missing planned step: {path.name}")
    return len(expected)


def validate_bf16_policy(errors: list[str]) -> int:
    path = ROOT / "bf16-model-policy.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != BF16_POLICY_COLUMNS:
            errors.append("bf16-model-policy.tsv header does not match its contract")
            return 0
        rows = list(reader)
    for row in rows:
        fp32 = float(row["fp32_tokens_per_second"])
        candidate = float(row["bf16_policy_tokens_per_second"])
        ratio = float(row["throughput_ratio"])
        if abs(ratio - candidate / fp32) > 1.0e-6:
            errors.append(f'BF16 policy ratio mismatch: {row["model"]}')
        fp32_bytes = int(row["fp32_engine_bytes"])
        candidate_bytes = int(row["bf16_engine_bytes"])
        if int(row["extra_engine_bytes"]) != candidate_bytes - fp32_bytes:
            errors.append(f'BF16 policy memory mismatch: {row["model"]}')
        if row["exact_tokens"] not in {"true", "false"}:
            errors.append(f'BF16 policy token gate is invalid: {row["model"]}')
        if row["decision"] not in {"keep", "discard"}:
            errors.append(f'BF16 policy decision is invalid: {row["model"]}')
    return len(rows)


def validate_bf16_ffn(errors: list[str]) -> int:
    data = ROOT / "experiments" / "030-data"
    raw_path = data / "raw.jsonl"
    summary_path = data / "summary.json"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 36:
        errors.append(f"BF16 FFN raw record count is {len(records)}, expected 36")
    keys = {(row["model"], row["tokens"], row["path"], row["process_run"])
            for row in records}
    if len(keys) != len(records):
        errors.append("BF16 FFN raw records contain duplicate workload/run keys")
    for row in records:
        if not row["accuracy_passed"]:
            errors.append("BF16 FFN raw record failed accuracy")
        if row["host_to_device_calls_measured"] != 0 or row["device_to_host_calls_measured"] != 0:
            errors.append("BF16 FFN measured region contains a host payload transfer")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("track") != "bf16_ffn" or len(summary.get("rows", [])) != 4:
        errors.append("BF16 FFN summary does not contain the fixed four-row matrix")
    return len(records)


def validate_bf16_models(errors: list[str]) -> int:
    data = ROOT / "experiments" / "031-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 18:
        errors.append(f"BF16 official-model record count is {len(records)}, expected 18")
    keys = {(row["model"], row["policy"], row["process_run"]) for row in records}
    if len(keys) != len(records):
        errors.append("BF16 official-model records contain duplicate keys")
    if any(not row.get("exact_expected_tokens") for row in records):
        errors.append("BF16 official-model record lacks exact expected token evidence")
    micro_bf16 = [row for row in records if row["policy"] == "bf16_ffn"]
    if any(row["max_abs_logit_difference_vs_fp32_run1"] > 0.2 for row in micro_bf16):
        errors.append("BF16 official-model max logit difference exceeds 0.2")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if summary.get("track") != "bf16_ffn_official_models" or len(summary.get("rows", [])) != 2:
        errors.append("BF16 official-model summary does not contain two model rows")
    for row in summary.get("rows", []):
        if row["current_memory_ratio"] >= 1.0 or row["decode_speedup"] <= 1.0:
            errors.append(f'BF16 official-model self-baseline gate failed: {row["model"]}')
    preparation = [json.loads(line) for line in
                   (data / "preparation-smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(preparation) != 2:
        errors.append("BF16 preparation smoke must contain two model rows")
    for row in preparation:
        expected_peak = row["fp32_weight_bytes"] + row["bf16_weight_bytes_retained"]
        if row["preparation_peak_bytes"] != expected_peak:
            errors.append(f'BF16 preparation peak formula changed: {row["model"]}')
        if row["preparation_current_bytes"] != row["resident_weight_bytes"]:
            errors.append(f'BF16 preparation retained-byte mismatch: {row["model"]}')
    return len(records)


def validate_bf16_prefill(errors: list[str]) -> int:
    data = ROOT / "experiments" / "032-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 12:
        errors.append(f"BF16 prefill record count is {len(records)}, expected 12")
    keys = {(row["model"], row["policy"], row["process_run"]) for row in records}
    if len(keys) != len(records) or any(not row.get("exact_expected_tokens") for row in records):
        errors.append("BF16 prefill records lack unique exact-token keys")
    before = json.loads(
        (ROOT / "experiments" / "031-data" / "summary.json").read_text(encoding="utf-8")
    )
    before_by_model = {row["model"]: row for row in before["rows"]}
    after = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if len(after.get("rows", [])) != 2:
        errors.append("BF16 prefill summary must contain two rows")
    for row in after.get("rows", []):
        old = before_by_model[row["model"]]
        if row["bf16_ffn_prefill_tokens_per_second"] <= \
                old["bf16_ffn_prefill_tokens_per_second"] * 1.4:
            errors.append(f'BF16 prefill gain is below retained gate: {row["model"]}')
        if row["bf16_ffn_decode_tokens_per_second"] < \
                old["bf16_ffn_decode_tokens_per_second"] * 0.95:
            errors.append(f'BF16 decode regressed over 5%: {row["model"]}')
    return len(records)


def validate_decode_profile(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "033-data"
    with (data / "kernel-stats.csv").open(encoding="utf-8", newline="") as stream:
        kernel_rows = list(csv.DictReader(stream))
    with (data / "hip-api-stats.csv").open(encoding="utf-8", newline="") as stream:
        api_rows = list(csv.DictReader(stream))
    kernel_calls = sum(int(row["Calls"]) for row in kernel_rows)
    api_calls = sum(int(row["Calls"]) for row in api_rows)
    summary = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    if kernel_calls != summary.get("kernel_dispatches") or kernel_calls != 10038:
        errors.append("DeepSeek decode kernel aggregate does not match summary")
    if api_calls != summary.get("hip_api_calls") or api_calls != 147537:
        errors.append("DeepSeek decode HIP API aggregate does not match summary")
    gemm = summary.get("categories", {}).get("gemm", {})
    if gemm.get("calls") != 3743 or abs(gemm.get("kernel_time_percent", 0) - 67.638) > 0.001:
        errors.append("DeepSeek decode GEMM category changed")
    return kernel_calls, api_calls


def validate_bf16_attention(errors: list[str]) -> int:
    data = ROOT / "experiments" / "034-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 8:
        errors.append(f"BF16 Attention raw count is {len(records)}, expected 8")
    candidates = [row for row in records if row.get("record_type") ==
                  "bf16_attention_candidate"]
    if len(candidates) != 6 or any(not row.get("exact_expected_tokens") for row in candidates):
        errors.append("BF16 Attention candidate rows lack exact-token evidence")
    if len({(row["model"], row["process_run"]) for row in candidates}) != 6:
        errors.append("BF16 Attention candidate process keys are not unique")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if len(summary.get("rows", [])) != 2:
        errors.append("BF16 Attention summary must contain two rows")
    for row in summary.get("rows", []):
        if row["decode_speedup_vs_bf16_ffn"] <= 1.0 or \
                row["prefill_speedup_vs_bf16_ffn"] < 0.95 or \
                not row["all_exact_expected_tokens"]:
            errors.append(f'BF16 Attention keep gate failed: {row["model"]}')
    pilot = (data / "naive-pilot.jsonl").read_text(encoding="utf-8").splitlines()
    if len(pilot) != 2:
        errors.append("BF16 Attention naive pilot must contain two rows")
    return len(records)


def validate_post_attention_profile(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "035-data"
    with (data / "kernel-stats.csv").open(encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "hip-api-stats.csv").open(encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    kernel_calls = sum(int(row["Calls"]) for row in kernels)
    api_calls = sum(int(row["Calls"]) for row in api)
    summary = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    if kernel_calls != 11214 or summary.get("kernel_dispatches") != kernel_calls:
        errors.append("post-Attention kernel aggregate mismatch")
    if api_calls != 152850 or summary.get("hip_api_calls") != api_calls:
        errors.append("post-Attention HIP API aggregate mismatch")
    if summary["categories"]["gemm"]["calls"] != 3743:
        errors.append("post-Attention GEMM count changed")
    return kernel_calls, api_calls


def validate_bf16_plan_cache(errors: list[str]) -> int:
    data = ROOT / "experiments" / "036-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = [row for row in records if row.get("record_type") ==
                  "bf16_attention_candidate"]
    if len(records) != 8 or len(candidates) != 6:
        errors.append("BF16 plan-cache raw matrix must contain 8/6 total/candidate rows")
    if any(not row.get("exact_expected_tokens") for row in candidates):
        errors.append("BF16 plan-cache candidate lacks exact-token evidence")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if len(summary.get("rows", [])) != 2:
        errors.append("BF16 plan-cache summary must contain two rows")
    for row in summary.get("rows", []):
        if min(row["decode_speedup_vs_bf16_attention"],
               row["prefill_speedup_vs_bf16_attention"],
               row["decode_ratio_vs_pytorch_bf16"],
               row["prefill_ratio_vs_pytorch_bf16"]) <= 1.0:
            errors.append(f'BF16 plan-cache keep gate failed: {row["model"]}')
    return len(records)


def validate_bf16_training(errors: list[str]) -> int:
    data = ROOT / "experiments" / "037-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 18:
        errors.append(f"BF16 training raw count is {len(records)}, expected 18")
    if len({(row["model"], row["policy"], row["process_run"]) for row in records}) != 18:
        errors.append("BF16 training process keys are not unique")
    if any(not row.get("parameter_changed") or
           not math.isfinite(float(row["final_loss"])) for row in records):
        errors.append("BF16 training record lacks a finite parameter update")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if len(summary.get("rows", [])) != 2:
        errors.append("BF16 training summary must contain two rows")
    for row in summary.get("rows", []):
        if row["microllm_bf16_ratio_vs_pytorch_bf16_amp"] <= 1.0 or \
                row["bf16_speedup_vs_microllm_fp32"] >= 1.0 or \
                row["bf16_peak_ratio_vs_microllm_fp32"] != 1.0 or \
                row["microllm_bf16_final_loss"] >= row["microllm_bf16_first_loss"]:
            errors.append(f'BF16 training evidence boundary changed: {row["model"]}')
    failure = json.loads((data / "pytorch-native-bf16-failure.json").read_text(
        encoding="utf-8"))
    if failure.get("status") != "failed":
        errors.append("native PyTorch BF16 parameter failure is missing")
    return len(records)


def validate_bf16_training_profile(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "038-data"
    summary = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    calls = []
    for policy in ("fp32", "bf16"):
        with (data / policy / "kernel-stats.csv").open(encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        with (data / policy / "hip-api-stats.csv").open(encoding="utf-8", newline="") as stream:
            api = list(csv.DictReader(stream))
        kernel_calls = sum(int(row["Calls"]) for row in kernels)
        api_calls = sum(int(row["Calls"]) for row in api)
        expected = summary["bf16_fp32_master" if policy == "bf16" else "fp32"]
        if kernel_calls != expected["kernel_dispatches"] or api_calls != expected["hip_api_calls"]:
            errors.append(f"BF16 training {policy} profiler aggregate mismatch")
        calls.extend((kernel_calls, api_calls))
    return calls[0] + calls[2], calls[1] + calls[3]


def validate_bf16_training_qkv(errors: list[str]) -> int:
    data = ROOT / "experiments" / "039-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 6 or len({(row["model"], row["process_run"]) for row in records}) != 6:
        errors.append("BF16 training QKV raw matrix must contain six unique rows")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if len(summary.get("rows", [])) != 2:
        errors.append("BF16 training QKV summary must contain two rows")
    ratios = [row["speedup_vs_bf16_independent_linears"] for row in summary.get("rows", [])]
    if len(ratios) == 2 and math.prod(ratios) ** 0.5 >= 1.0:
        errors.append("BF16 training QKV discard boundary changed")
    if "bf16_qkv_projection" in (REPOSITORY / "include" / "microllm" /
                                  "autograd" / "autograd.h").read_text(encoding="utf-8"):
        errors.append("discarded BF16 training QKV graph API remains in public header")
    return len(records)


def validate_bf16_training_mirrors(errors: list[str]) -> int:
    data = ROOT / "experiments" / "040-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 6 or len({(row["model"], row["process_run"])
                                 for row in records}) != 6:
        errors.append("BF16 training mirror raw matrix must contain six unique rows")
    for row in records:
        if row.get("record_type") != "bf16_training_weight_mirrors_candidate" or \
                not row.get("parameter_changed") or \
                not math.isfinite(float(row["final_loss"])) or \
                row["final_loss"] >= row["first_loss"] or \
                row.get("bf16_training_mirror_tensors", 0) <= 0 or \
                row.get("bf16_training_mirror_bytes", 0) <= 0 or \
                row.get("optimizer_host_to_device_calls") != 0 or \
                row.get("optimizer_device_to_host_calls") != 0:
            errors.append(f'BF16 training mirror evidence failed: {row.get("model")}')
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if summary.get("track") != "bf16_training_weight_mirrors" or \
            len(summary.get("rows", [])) != 2:
        errors.append("BF16 training mirror summary must contain the fixed two-model track")
    for row in summary.get("rows", []):
        if row["speedup_vs_bf16_independent_linears"] <= 1.05 or \
                row["peak_ratio_vs_baseline"] <= 1.0 or \
                row["ratio_vs_pytorch_bf16_amp"] <= 1.0:
            errors.append(f'BF16 training mirror trade-off boundary changed: {row["model"]}')
    return len(records)


def validate_bf16_training_island(errors: list[str]) -> int:
    data = ROOT / "experiments" / "041-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = [row for row in records if row["record_type"] ==
                  "bf16_training_ffn_islands_candidate"]
    controls = [row for row in records if row["record_type"] ==
                "bf16_training_weight_mirror_control"]
    if len(records) != 4 or len(candidates) != 3 or len(controls) != 1 or \
            len({row["process_run"] for row in candidates}) != 3:
        errors.append("BF16 training island evidence must contain 3 candidates and 1 control")
    if any(not row["parameter_changed"] or row["final_loss"] >= row["first_loss"] or
           row["optimizer_host_to_device_calls"] != 0 or
           row["optimizer_device_to_host_calls"] != 0 for row in records):
        errors.append("BF16 training island/control correctness evidence failed")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    qwen = summary.get("qwen", {})
    if summary.get("decision") != "discard" or \
            qwen.get("speedup_vs_same_window_control", 1.0) >= 1.05 or \
            summary.get("deepseek", {}).get("status") != "early_stop":
        errors.append("BF16 training island discard/early-stop boundary changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "kernel-stats.csv").open(encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "hip-api-stats.csv").open(encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    if sum(int(row["Calls"]) for row in kernels) != profile.get("kernel_dispatches") or \
            sum(int(row["Calls"]) for row in api) != profile.get("hip_api_calls"):
        errors.append("BF16 training island profiler aggregates changed")
    public_autograd = (REPOSITORY / "include" / "microllm" / "autograd" /
                       "autograd.h").read_text(encoding="utf-8")
    public_model = (REPOSITORY / "include" / "microllm" / "model" /
                    "model.h").read_text(encoding="utf-8")
    if "Value bf16_ffn" in public_autograd or \
            "enable_bf16_ffn_training_islands" in public_model:
        errors.append("discarded BF16 FFN training graph API remains public")
    return len(records)


def validate_bf16_training_shapes(errors: list[str]) -> int:
    data = ROOT / "experiments" / "042-data"
    records = [json.loads(line) for line in
               (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("framework"), row.get("batch"), row.get("context"),
             row.get("process_run")) for row in records}
    if len(records) != 24 or len(keys) != 24:
        errors.append("official training shape raw matrix must contain 24 unique rows")
    expected_shapes = {(1, 3), (2, 3), (1, 32), (1, 128)}
    if {(row.get("batch"), row.get("context")) for row in records} != expected_shapes:
        errors.append("official training shape matrix changed its fixed shapes")
    for row in records:
        if row.get("status") != "pass" or not row.get("parameter_changed") or \
                row.get("trained_tokens") != row.get("batch") * row.get("context") * 2 or \
                not math.isfinite(float(row.get("final_loss", math.nan))):
            errors.append("official training shape row lacks finite update/token evidence")
        if row.get("framework") == "microllm" and (
                row.get("optimizer_host_to_device_calls") != 0 or
                row.get("optimizer_device_to_host_calls") != 0):
            errors.append("official training shape microLLM row copied optimizer payloads")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    if summary.get("track") != "official_training_shape_matrix" or \
            summary.get("runs_per_framework") != 3 or len(rows) != 4 or \
            any(row.get("status") != "pass" for row in rows):
        errors.append("official training shape summary contract changed")
    by_shape = {(row["batch"], row["context"]): row for row in rows}
    if len(by_shape) == 4 and (
            by_shape[(1, 32)]["throughput_ratio_microllm_over_pytorch"] >= 0.2 or
            by_shape[(1, 128)]["throughput_ratio_microllm_over_pytorch"] <=
            by_shape[(1, 32)]["throughput_ratio_microllm_over_pytorch"] or
            by_shape[(1, 128)]["peak_memory_ratio"] <= 1.0):
        errors.append("official training shape bottleneck boundary changed")
    return len(records)


def validate_weight_gradient_routing(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "043-data"
    candidates = [json.loads(line) for line in
                  (data / "candidate" / "raw.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    keys = {(row.get("framework"), row.get("batch"), row.get("context"),
             row.get("process_run")) for row in candidates}
    if len(candidates) != 24 or len(keys) != 24 or \
            any(row.get("status") != "pass" for row in candidates):
        errors.append("weight-gradient candidate matrix must contain 24 passing unique rows")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    if comparison.get("decision") != "keep" or len(rows) != 4 or \
            any(row["self_speedup"] < 1.0 or
                row["peak_ratio_after_vs_before"] != 1.0 for row in rows):
        errors.append("weight-gradient official keep gate changed")
    by_shape = {(row["batch"], row["context"]): row for row in rows}
    if len(by_shape) == 4 and by_shape[(1, 32)]["self_speedup"] <= 4.0:
        errors.append("weight-gradient context-32 speedup fell below evidence boundary")

    microbench = [json.loads(line) for line in
                  (data / "microbench.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(microbench) != 12 or any(row["maximum_absolute_error"] > 3.0e-7
                                    for row in microbench):
        errors.append("weight-gradient microbenchmark matrix/error contract changed")
    optimized = [row for row in microbench if row["implementation"] == "hipblaslt"]
    if len(optimized) != 6 or any(row["speedup_vs_readable"] <= 1.0 for row in optimized):
        errors.append("weight-gradient microbenchmark speedup gate changed")

    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    for label, key in (("before-context32", "before_context32"),
                       ("before-context128", "before_context128"),
                       ("after-context32", "after_context32")):
        with (data / "profile" / label / "kernel-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        with (data / "profile" / label / "hip-api-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            api = list(csv.DictReader(stream))
        expected = profile[key]
        if sum(int(row["Calls"]) for row in kernels) != expected["kernel_dispatches"] or \
                sum(int(row["TotalDurationNs"]) for row in kernels) != \
                expected["kernel_time_ns"] or \
                sum(int(row["Calls"]) for row in api) != expected["hip_api_calls"]:
            errors.append(f"weight-gradient {label} profiler aggregate changed")
    if profile["before_context32"]["readable_transpose_calls"] != 507 or \
            profile["after_context32"]["readable_transpose_calls"] != 0:
        errors.append("weight-gradient readable transpose hotspot boundary changed")
    return len(candidates), len(microbench)


def validate_fused_causal_gqa(errors: list[str]) -> int:
    data = ROOT / "experiments" / "044-data"
    records = [json.loads(line) for line in
               (data / "candidate" / "raw.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    keys = {(row.get("framework"), row.get("batch"), row.get("context"),
             row.get("process_run")) for row in records}
    if len(records) != 24 or len(keys) != 24 or \
            any(row.get("status") != "pass" for row in records):
        errors.append("fused causal GQA matrix must contain 24 passing unique rows")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    if comparison.get("decision") != "keep" or len(rows) != 4 or any(
            row["self_speedup"] < 1.05 or
            row["peak_ratio_after_vs_before"] >= 1.0 or
            row["peak_bytes_saved"] <= 0 for row in rows):
        errors.append("fused causal GQA official keep gate changed")

    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    before_path = ROOT / "experiments" / "043-data" / "profile" / \
        "before-context128"
    after_path = data / "profile" / "after-context128"
    for path, key in ((before_path, "before_context128"),
                      (after_path, "after_context128")):
        with (path / "kernel-stats.csv").open(encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        with (path / "hip-api-stats.csv").open(encoding="utf-8", newline="") as stream:
            api = list(csv.DictReader(stream))
        expected = profile[key]
        if sum(int(row["Calls"]) for row in kernels) != expected["kernel_dispatches"] or \
                sum(int(row["TotalDurationNs"]) for row in kernels) != \
                expected["kernel_time_ns"] or \
                sum(int(row["Calls"]) for row in api) != expected["hip_api_calls"]:
            errors.append(f"fused causal GQA {key} profiler aggregate changed")
    if profile["dispatch_reduction"] != 1080 or \
            profile["after_context128"]["fused_forward_calls"] != 72 or \
            profile["after_context128"]["fused_backward_calls"] != 72:
        errors.append("fused causal GQA dispatch/kernel boundary changed")
    return len(records)


def validate_deepseek_shapes_and_load(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "045-data"
    pilot = [json.loads(line) for line in
             (data / "pilot-before-load.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [json.loads(line) for line in
               (data / "candidate" / "raw.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    keys = {(row.get("framework"), row.get("batch"), row.get("context"),
             row.get("process_run")) for row in records}
    if len(pilot) != 8 or len(records) != 24 or len(keys) != 24 or \
            any(row.get("status") != "pass" for row in records):
        errors.append("DeepSeek pilot/formal shape record counts changed")
    micro = [row for row in records if row.get("framework") == "microllm"]
    torch = [row for row in records if row.get("framework") == "pytorch"]
    if any(not (60000.0 < row.get("load_ms", 0) < 70000.0) for row in micro) or \
            any(not (0.0 < row.get("load_ms", 0) < 3000.0) for row in torch):
        errors.append("DeepSeek load-time evidence boundary changed")
    summary = json.loads((data / "candidate" / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    if summary.get("runs_per_framework") != 3 or len(rows) != 4 or any(
            row.get("status") != "pass" or
            row["throughput_ratio_microllm_over_pytorch"] >= 0.6 or
            not (0.85 < row["peak_memory_ratio"] < 1.0) for row in rows):
        errors.append("DeepSeek shape baseline boundary changed")
    load = json.loads((data / "load-summary.json").read_text(encoding="utf-8"))
    if load.get("decision") != "keep" or \
            load.get("before_observed_process_setup_seconds", {}).get("minimum") != 360:
        errors.append("DeepSeek device-native load decision evidence changed")
    return len(pilot), len(records)


def validate_deepseek_optimizer_profile(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "046-data"
    summary = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    kernel_calls = sum(int(row["Calls"]) for row in kernels)
    kernel_time = sum(int(row["TotalDurationNs"]) for row in kernels)
    api_calls = sum(int(row["Calls"]) for row in api)
    api_time = sum(int(row["TotalDurationNs"]) for row in api)
    if (kernel_calls, kernel_time, api_calls, api_time) != (
            summary.get("kernel_dispatches"), summary.get("kernel_time_ns"),
            summary.get("hip_api_calls"), summary.get("hip_api_time_ns")):
        errors.append("DeepSeek optimizer profiler aggregate changed")
    categories = summary.get("categories", [])
    if len(categories) != 9 or \
            sum(row.get("calls", 0) for row in categories) != kernel_calls or \
            sum(row.get("total_duration_ns", 0) for row in categories) != kernel_time or \
            abs(sum(row.get("kernel_time_percent", 0.0) for row in categories) - 100.0) > 1e-5:
        errors.append("DeepSeek optimizer profile category partition changed")
    by_name = {row.get("name"): row for row in categories}
    adamw = by_name.get("AdamW master and BF16 mirror update", {})
    copy = by_name.get("strided copy", {})
    counts = summary.get("clean_count_contracts", {})
    if adamw.get("calls") != 1017 or adamw.get("kernel_time_percent", 0.0) < 32.0 or \
            adamw.get("training_only") is not True or \
            copy.get("training_only") is not False or \
            counts.get("parameter_tensors", 0) * counts.get("optimizer_steps_in_trace", 0) != \
            counts.get("adamw_calls") or \
            counts.get("transformer_layers", 0) * counts.get("attention_steps_in_trace", 0) != \
            counts.get("attention_forward_calls") or \
            counts.get("attention_forward_calls") != counts.get("attention_backward_calls"):
        errors.append("DeepSeek optimizer profile attribution contract changed")
    return kernel_calls, api_calls


def validate_stable_gradient_discard(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "047-data"
    matched = [json.loads(line) for line in
               (data / "candidate" / "raw.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    mismatched = [json.loads(line) for line in
                  (data / "protocol-mismatch" / "raw.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    matched_keys = {(row.get("framework"), row.get("process_run")) for row in matched}
    mismatch_keys = {(row.get("framework"), row.get("process_run")) for row in mismatched}
    if len(matched) != 6 or len(matched_keys) != 6 or any(
            row.get("status") != "pass" or row.get("batch") != 1 or
            row.get("context") != 128 or row.get("warmup") != 1 or
            row.get("steps") != 2 for row in matched):
        errors.append("stable-gradient matched protocol evidence changed")
    if len(mismatched) != 6 or len(mismatch_keys) != 6 or any(
            row.get("status") != "pass" or row.get("warmup") != 2 or
            row.get("steps") != 5 for row in mismatched):
        errors.append("stable-gradient rejected protocol evidence changed")
    summary = json.loads((data / "candidate" / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    baseline_records = [json.loads(line) for line in
                        (ROOT / "experiments" / "044-data" / "candidate" /
                         "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    baseline = sorted(float(row["tokens_per_second"]) for row in baseline_records
                      if row.get("framework") == "microllm" and
                      row.get("batch") == 1 and row.get("context") == 128)[1]
    candidate = sorted(float(row["tokens_per_second"]) for row in matched
                       if row.get("framework") == "microllm")[1]
    candidate_peak = sorted(int(row["engine_peak_bytes"]) for row in matched
                            if row.get("framework") == "microllm")[1]
    if len(rows) != 1 or summary.get("runs_per_framework") != 3 or \
            abs(rows[0].get("microllm_tokens_per_second", 0.0) - candidate) > 1e-6 or \
            comparison.get("decision") != "discard" or \
            abs(comparison.get("baseline_tokens_per_second", 0.0) - baseline) > 1e-6 or \
            abs(comparison.get("candidate_tokens_per_second", 0.0) - candidate) > 1e-6 or \
            comparison.get("candidate_peak_bytes") != candidate_peak or \
            comparison.get("speed_ratio", 1.0) >= 0.95 or \
            comparison.get("peak_ratio", 1.0) >= 0.95:
        errors.append("stable-gradient discard boundary changed")
    return len(matched), len(mismatched)


def validate_chunked_adamw_discard(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "048-data"
    pilot = [json.loads(line) for line in
             (data / "all-tensor-early-stop" / "raw.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    context128 = [json.loads(line) for line in
                  (data / "small-tensor" / "context128" / "raw.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    rest = [json.loads(line) for line in
            (data / "small-tensor" / "rest" / "raw.jsonl").read_text(
                encoding="utf-8").splitlines()]
    formal = context128 + rest
    keys = {(row.get("framework"), row.get("batch"), row.get("context"),
             row.get("process_run")) for row in formal}
    if len(pilot) != 2 or any(row.get("status") != "pass" for row in pilot) or \
            len(formal) != 24 or len(keys) != 24 or any(
                row.get("status") != "pass" or row.get("warmup") != 1 or
                row.get("steps") != 2 for row in formal):
        errors.append("chunked AdamW raw protocol changed")
    micro = [row for row in formal if row.get("framework") == "microllm"]
    if len(micro) != 12 or any(
            row.get("optimizer_tensor_updates") != 290 or
            row.get("optimizer_hip_scalar_launches") != 169 or
            row.get("optimizer_hip_group_launches") != 8 or
            row.get("optimizer_maximum_group_size") != 16 or
            row.get("optimizer_host_to_device_calls") != 0 or
            row.get("optimizer_device_to_host_calls") != 0 for row in micro):
        errors.append("small-tensor AdamW dispatch/transfer contract changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("small_tensor_candidate", {}).get("rows", [])
    all_group = comparison.get("all_tensor_pilot", {})
    if comparison.get("decision") != "discard" or len(rows) != 4 or \
            any(row.get("speedup", 0.0) >= 1.05 or row.get("peak_ratio") != 1.0
                for row in rows) or \
            all_group.get("speedup", 1.0) >= 0.6 or \
            all_group.get("group_launches") != 19 or \
            comparison.get("small_tensor_candidate", {}).get("dispatches_after") != 177:
        errors.append("chunked AdamW discard boundary changed")
    baseline = json.loads((ROOT / "experiments" / "044-data" / "candidate" /
                           "summary.json").read_text(encoding="utf-8"))
    baseline_by_shape = {(row["batch"], row["context"]): row for row in baseline["rows"]}
    formal_summaries = []
    for path in (data / "small-tensor" / "context128" / "summary.json",
                 data / "small-tensor" / "rest" / "summary.json"):
        formal_summaries.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    candidate_by_shape = {(row["batch"], row["context"]): row
                          for row in formal_summaries}
    for row in rows:
        key = (row["batch"], row["context"])
        if key not in baseline_by_shape or key not in candidate_by_shape or \
                abs(row["before_tokens_per_second"] -
                    baseline_by_shape[key]["microllm_tokens_per_second"]) > 1e-6 or \
                abs(row["after_tokens_per_second"] -
                    candidate_by_shape[key]["microllm_tokens_per_second"]) > 1e-6:
            errors.append(f"chunked AdamW shape comparison changed: {key}")
    return len(pilot), len(formal)


def validate_vectorized_adamw(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "049-data"
    vector4 = [json.loads(line) for line in
               (data / "vector4-mirror.jsonl").read_text(encoding="utf-8").splitlines()]
    vector8 = [json.loads(line) for line in
               (data / "vector8-mirror.jsonl").read_text(encoding="utf-8").splitlines()]
    rsqrt = [json.loads(line) for line in
             (data / "rsqrt-mirror.jsonl").read_text(encoding="utf-8").splitlines()]
    no_mirror = [json.loads(line) for line in
                 (data / "vector4-no-mirror.jsonl").read_text(encoding="utf-8").splitlines()]
    pilot = [json.loads(line) for line in
             (data / "qwen-pilot" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("elements"), row.get("implementation")) for row in vector4}
    if len(vector4) != 24 or len(keys) != 24 or any(
            row.get("sample_maximum_absolute_error", 1.0) > 3e-8 or
            row.get("warmup") != 3 or row.get("repetitions") != 10 or
            row.get("bf16_mirror") is not True for row in vector4):
        errors.append("vectorized AdamW float4 evidence changed")
    if len(vector8) != 12 or len(rsqrt) != 12 or len(no_mirror) != 4 or \
            any(row.get("sample_maximum_absolute_error", 1.0) > 3e-8
                for row in vector8 + rsqrt + no_mirror):
        errors.append("vectorized AdamW counterexample records changed")
    scalar = {row["elements"]: row for row in vector4
              if row["implementation"] == "scalar"}
    float4 = {row["elements"]: row for row in vector4
              if row["implementation"] == "vectorized"}
    width8 = {row["elements"]: row for row in vector8}
    corrected_rsqrt = {row["elements"]: row for row in rsqrt}
    if set(scalar) != set(float4) or any(
            width8[key]["kernel_ms_mean"] <= float4[key]["kernel_ms_mean"] or
            corrected_rsqrt[key]["kernel_ms_mean"] <= float4[key]["kernel_ms_mean"]
            for key in scalar):
        errors.append("vector-width/rsqrt rejection boundary changed")
    pilot_keys = {(row.get("framework"), row.get("batch"), row.get("context"))
                  for row in pilot}
    if len(pilot) != 8 or len(pilot_keys) != 8 or any(
            row.get("status") != "pass" for row in pilot) or any(
                row.get("adamw_implementation") != "vectorized"
                for row in pilot if row.get("framework") == "microllm"):
        errors.append("vectorized AdamW Qwen pilot changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    operator_rows = comparison.get("operator_rows", [])
    model_rows = comparison.get("qwen_vectorized_pilot", [])
    no_mirror_rows = comparison.get("no_mirror_rows", [])
    if comparison.get("decision") != "keep-explicit-only" or \
            len(operator_rows) != 12 or \
            sum(row.get("speedup", 0.0) >= 1.05 for row in operator_rows) != 3 or \
            any(row.get("speedup", 1.0) >= 1.0 for row in model_rows) or \
            any(row.get("speedup", 1.0) >= 1.0 for row in no_mirror_rows) or \
            comparison.get("maximum_sample_absolute_error", 1.0) > 3e-8 or \
            "Auto remains Scalar" not in comparison.get("policy", ""):
        errors.append("vectorized AdamW explicit-only policy changed")
    return len(vector4) + len(vector8) + len(rsqrt) + len(no_mirror), len(pilot)


def validate_streaming_load(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "050-data"
    smoke = [json.loads(line) for line in
             (data / "load-smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "deepseek-formal" / "raw.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    keys = {(row.get("framework"), row.get("batch"), row.get("context"),
             row.get("process_run")) for row in formal}
    if len(smoke) != 2 or any(row.get("status") != "pass" for row in smoke) or \
            len(formal) != 24 or len(keys) != 24 or any(
                row.get("status") != "pass" for row in formal):
        errors.append("streaming load raw record contract changed")
    for row in smoke:
        if row.get("load_host_to_device_bytes") != row.get("parameter_count", 0) * 2 or \
                row.get("load_current_engine_bytes") != row.get("fp32_weight_bytes") or \
                row.get("load_device_to_host_calls") != 0 or \
                row.get("load_device_to_device_calls") != 0 or \
                not (1.0 < row.get("load_peak_engine_bytes", 0) /
                     row.get("fp32_weight_bytes", 1) < 1.15):
            errors.append(f'streaming load byte/peak contract changed: {row.get("model")}')
    micro = [row for row in formal if row.get("framework") == "microllm"]
    torch = [row for row in formal if row.get("framework") == "pytorch"]
    if any(not (1000.0 < row.get("load_ms", 0.0) < 1500.0) or
           row.get("adamw_implementation") != "auto" for row in micro) or \
            any(not (1900.0 < row.get("load_ms", 0.0) < 2300.0) for row in torch):
        errors.append("streaming load timing boundary changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    loads = comparison.get("load_rows", [])
    training = comparison.get("deepseek_training_rows", [])
    safety = comparison.get("safety", {})
    if comparison.get("decision") != "keep" or len(loads) != 2 or \
            loads[0].get("speedup", 0.0) < 30.0 or loads[1].get("speedup", 0.0) < 45.0 or \
            loads[1].get("speedup_vs_pytorch", 0.0) < 1.4 or \
            len(training) != 4 or any(
                not (0.99 < row.get("self_speedup", 0.0) < 1.01) or
                row.get("peak_ratio") != 1.0 for row in training) or \
            not all((safety.get("strict_metadata_preflight_before_payload_transfer"),
                     safety.get("partial_io_failure_keeps_model_uninitialized"),
                     safety.get("initialized_model_uses_atomic_state_dict_fallback"))):
        errors.append("streaming load keep/safety gate changed")
    return len(smoke), len(formal)


def validate_context512(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "051-data"
    pilot = [json.loads(line) for line in
             (data / "pilot" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in formal}
    if len(pilot) != 4 or len(formal) != 12 or len(keys) != 12 or any(
            row.get("status") != "pass" or row.get("batch") != 1 or
            row.get("context") != 512 or row.get("warmup") != 1 or
            row.get("steps") != 2 for row in pilot + formal):
        errors.append("context-512 raw protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    if comparison.get("status") != "stable-failure" or len(rows) != 2 or any(
            row.get("throughput_ratio", 1.0) >= 0.1 or
            row.get("peak_ratio", 0.0) <= 1.0 or
            row.get("parameter_changed") is not True for row in rows):
        errors.append("context-512 stable-failure boundary changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    kernel_calls = sum(int(row["Calls"]) for row in kernels)
    kernel_time = sum(int(row["TotalDurationNs"]) for row in kernels)
    api_calls = sum(int(row["Calls"]) for row in api)
    api_time = sum(int(row["TotalDurationNs"]) for row in api)
    categories = profile.get("categories", [])
    if (kernel_calls, kernel_time, api_calls, api_time) != (
            profile.get("kernel_dispatches"), profile.get("kernel_time_ns"),
            profile.get("hip_api_calls"), profile.get("hip_api_time_ns")) or \
            len(categories) != 6 or \
            sum(row.get("calls", 0) for row in categories) != kernel_calls or \
            sum(row.get("time_ns", 0) for row in categories) != kernel_time or \
            profile.get("attention_time_percent", 0.0) < 64.0 or \
            categories[0].get("calls") != 72 or categories[1].get("calls") != 72:
        errors.append("context-512 profiler/category contract changed")
    return len(pilot), len(formal)


def validate_split_kv_discard(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "052-data"
    pilot = [json.loads(line) for line in
             (data / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(pilot) != 2 or any(row.get("status") != "pass" or
                              row.get("context") != 512 for row in pilot):
        errors.append("split K/V pilot protocol changed")
    summary = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    kernel_calls = sum(int(row["Calls"]) for row in kernels)
    kernel_time = sum(int(row["TotalDurationNs"]) for row in kernels)
    api_calls = sum(int(row["Calls"]) for row in api)
    api_time = sum(int(row["TotalDurationNs"]) for row in api)
    if (kernel_calls, kernel_time, api_calls, api_time) != (
            summary.get("kernel_dispatches"), summary.get("kernel_time_ns"),
            summary.get("hip_api_calls"), summary.get("hip_api_time_ns")) or \
            summary.get("split_rows_calls") != 72 or \
            summary.get("split_kv_calls") != 72:
        errors.append("split K/V profiler aggregate changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    if comparison.get("decision") != "discard" or \
            comparison.get("throughput_ratio", 1.0) >= 0.9 or \
            comparison.get("backward_speedup", 1.0) >= 0.8 or \
            comparison.get("peak_ratio") != 1.0 or \
            comparison.get("candidate_backward_time_ns") != \
            comparison.get("candidate_row_time_ns", 0) + \
            comparison.get("candidate_kv_time_ns", 0):
        errors.append("split K/V discard boundary changed")
    return len(pilot), kernel_calls


def validate_batched_gemm(errors: list[str]) -> int:
    data = ROOT / "experiments" / "053-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("implementation"), row.get("m"), row.get("k"), row.get("n"),
             row.get("transpose_left"), row.get("transpose_right")) for row in raw}
    if len(raw) != 6 or len(keys) != 6 or any(
            row.get("batch") != 14 or row.get("warmup") != 3 or
            row.get("repetitions") != 10 for row in raw):
        errors.append("strided-batched GEMM raw protocol changed")
    optimized = [row for row in raw if row.get("implementation") == "hipblaslt"]
    invalid_readable = [row for row in raw if row.get("implementation") == "readable" and
                        row.get("transpose_left") is True]
    if len(optimized) != 3 or any(row.get("maximum_absolute_error", 1.0) > 2e-6
                                  for row in optimized) or \
            len(invalid_readable) != 1 or \
            invalid_readable[0].get("maximum_absolute_error", 0.0) < 0.1:
        errors.append("strided-batched GEMM correctness/invalid-control boundary changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    valid = [row for row in rows if row.get("valid_speedup")]
    invalid = [row for row in rows if not row.get("valid_speedup")]
    if comparison.get("decision") != "keep" or len(rows) != 3 or len(valid) != 2 or \
            any(row.get("speedup", 0.0) < 20.0 for row in valid) or \
            len(invalid) != 1 or invalid[0].get("speedup") is not None or \
            comparison.get("correctness", {}).get("transpose_layouts") != 4:
        errors.append("strided-batched GEMM keep gate changed")
    return len(raw)


def validate_batched_attention_backward(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "054-data"
    pilot = [json.loads(line) for line in
             (data / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    fallback = [json.loads(line) for line in
                (data / "fallback128.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in formal}
    if len(pilot) != 2 or len(formal) != 12 or len(keys) != 12 or \
            len(fallback) != 1 or any(row.get("status") != "pass" for row in
                                      pilot + formal + fallback) or \
            fallback[0].get("context") != 128:
        errors.append("batched Attention backward raw protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    fallback_summary = comparison.get("fallback128", {})
    if comparison.get("decision") != "keep" or len(rows) != 2 or any(
            row.get("self_speedup", 0.0) < 1.35 or
            row.get("peak_ratio_after_vs_before") != 1.0 for row in rows) or \
            fallback_summary.get("speedup", 0.0) < 0.99 or \
            fallback_summary.get("peak_ratio") != 1.0:
        errors.append("batched Attention official keep gate changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    after = profile.get("after", {})
    if sum(int(row["Calls"]) for row in kernels) != after.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != after.get("kernel_time_ns") or \
            sum(int(row["Calls"]) for row in api) != after.get("hip_api_calls") or \
            sum(int(row["TotalDurationNs"]) for row in api) != after.get("hip_api_time_ns") or \
            after.get("row_backward_calls") != 72 or after.get("batched_gemm_calls") != 144 or \
            after.get("gqa_reduce_calls") != 144 or \
            profile.get("kernel_time_speedup", 0.0) < 1.34 or \
            profile.get("identified_backward_speedup", 0.0) < 2.0:
        errors.append("batched Attention retained profile changed")
    return len(formal), after.get("kernel_dispatches", 0)


def validate_saved_attention(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "055-data"
    pilot = [json.loads(line) for line in
             (data / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    fallback = [json.loads(line) for line in
                (data / "fallback128.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in formal}
    if len(pilot) != 2 or len(formal) != 12 or len(keys) != 12 or \
            len(fallback) != 1 or any(row.get("status") != "pass" for row in
                                      pilot + formal + fallback) or \
            fallback[0].get("context") != 128:
        errors.append("saved Attention raw protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    fallback_summary = comparison.get("fallback128", {})
    if comparison.get("decision") != "keep" or len(rows) != 2 or any(
            row.get("self_speedup", 0.0) < 1.13 or
            not (1.0 < row.get("peak_ratio", 0.0) < 1.03) or
            row.get("peak_bytes_added") != 352321536 for row in rows) or \
            fallback_summary.get("speedup", 0.0) < 0.99 or \
            fallback_summary.get("peak_ratio") != 1.0 or \
            "T>=256" not in comparison.get("policy", ""):
        errors.append("saved Attention speed/memory policy changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    after = profile.get("after", {})
    if sum(int(row["Calls"]) for row in kernels) != after.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != after.get("kernel_time_ns") or \
            sum(int(row["Calls"]) for row in api) != after.get("hip_api_calls") or \
            sum(int(row["TotalDurationNs"]) for row in api) != after.get("hip_api_time_ns") or \
            after.get("saved_row_backward_calls") != 72 or after.get("forward_calls") != 72 or \
            profile.get("kernel_time_speedup", 0.0) < 1.12 or \
            profile.get("row_backward_speedup", 0.0) < 1.55 or \
            profile.get("dispatch_ratio") != 1.0:
        errors.append("saved Attention retained profile changed")
    return len(formal), after.get("kernel_dispatches", 0)


def validate_batched_attention_forward(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "056-data"
    pilot = [json.loads(line) for line in
             (data / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    fallback = [json.loads(line) for line in
                (data / "fallback128.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in formal}
    if len(pilot) != 2 or len(formal) != 12 or len(keys) != 12 or \
            len(fallback) != 1 or any(row.get("status") != "pass" for row in
                                      pilot + formal + fallback) or \
            any(row.get("context") != 512 for row in formal) or \
            fallback[0].get("context") != 128:
        errors.append("batched Attention forward raw protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    fallback_summary = comparison.get("fallback128", {})
    if comparison.get("decision") != "keep" or len(rows) != 2 or \
            rows[0].get("self_speedup", 0.0) < 1.09 or \
            rows[1].get("self_speedup", 0.0) < 1.16 or \
            any(row.get("peak_ratio") != 1.0 for row in rows) or \
            fallback_summary.get("speedup", 0.0) < 0.99 or \
            fallback_summary.get("peak_ratio") != 1.0 or \
            "T>=256" not in comparison.get("policy", ""):
        errors.append("batched Attention forward keep/fallback gate changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    before = profile.get("before", {})
    after = profile.get("after", {})
    forward_sum = sum(after.get(name, 0) for name in (
        "softmax_time_ns", "repeat_kv_time_ns", "query_scale_time_ns",
        "forward_batched_gemm_time_ns"))
    if sum(int(row["Calls"]) for row in kernels) != after.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != \
            after.get("kernel_time_ns") or \
            sum(int(row["Calls"]) for row in api) != after.get("hip_api_calls") or \
            sum(int(row["TotalDurationNs"]) for row in api) != \
            after.get("hip_api_time_ns") or \
            after.get("forward_batched_gemm_calls") != 144 or \
            after.get("repeat_kv_calls") != 144 or \
            after.get("forward_stage_time_ns") != forward_sum or \
            before.get("fused_forward_calls") != 72 or \
            profile.get("forward_stage_speedup", 0.0) < 1.52 or \
            profile.get("kernel_time_speedup", 0.0) < 1.08 or \
            profile.get("kernel_dispatch_ratio", 0.0) <= 1.0:
        errors.append("batched Attention forward retained profile changed")
    return len(formal), after.get("kernel_dispatches", 0)


def validate_full_batched_attention_backward(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "057-data"
    pilot = [json.loads(line) for line in
             (data / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    fallback = [json.loads(line) for line in
                (data / "fallback128.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in formal}
    if len(pilot) != 1 or len(formal) != 12 or len(keys) != 12 or \
            len(fallback) != 1 or any(row.get("status") != "pass" for row in
                                      pilot + formal + fallback) or \
            any(row.get("context") != 512 for row in formal) or \
            fallback[0].get("context") != 128:
        errors.append("full batched Attention backward raw protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    fallback_summary = comparison.get("fallback128", {})
    if comparison.get("decision") != "keep" or len(rows) != 2 or \
            rows[0].get("self_speedup", 0.0) < 1.20 or \
            rows[1].get("self_speedup", 0.0) < 1.30 or \
            any(row.get("peak_ratio") != 1.0 for row in rows) or \
            fallback_summary.get("speedup", 0.0) < 0.95 or \
            fallback_summary.get("peak_ratio") != 1.0 or \
            "T>=256" not in comparison.get("policy", ""):
        errors.append("full batched Attention backward keep/fallback gate changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    before = profile.get("before", {})
    after = profile.get("after", {})
    replacement = sum(after.get(name, 0) for name in (
        "causal_softmax_backward_time_ns", "additional_repeat_time_ns",
        "additional_score_scale_time_ns", "additional_batched_gemm_time_ns"))
    if sum(int(row["Calls"]) for row in kernels) != after.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != \
            after.get("kernel_time_ns") or \
            sum(int(row["Calls"]) for row in api) != after.get("hip_api_calls") or \
            sum(int(row["TotalDurationNs"]) for row in api) != \
            after.get("hip_api_time_ns") or \
            before.get("saved_row_calls") != 72 or after.get("saved_row_calls") != 0 or \
            after.get("causal_softmax_backward_calls") != 72 or \
            after.get("additional_batched_gemm_calls") != 144 or \
            after.get("replacement_stage_time_ns") != replacement or \
            profile.get("saved_backward_stage_speedup", 0.0) < 2.5 or \
            profile.get("kernel_time_speedup", 0.0) < 1.19:
        errors.append("full batched Attention backward retained profile changed")
    return len(formal), after.get("kernel_dispatches", 0)


def validate_block_row_causal_softmax(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "058-data"
    pilot = [json.loads(line) for line in
             (data / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    fallback = [json.loads(line) for line in
                (data / "fallback128.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in formal}
    if len(pilot) != 1 or len(formal) != 12 or len(keys) != 12 or \
            len(fallback) != 1 or any(row.get("status") != "pass" for row in
                                      pilot + formal + fallback) or \
            any(row.get("context") != 512 for row in formal) or \
            fallback[0].get("context") != 128:
        errors.append("block-row causal-softmax raw protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    fallback_summary = comparison.get("fallback128", {})
    if comparison.get("decision") != "keep" or len(rows) != 2 or \
            rows[0].get("self_speedup", 0.0) < 1.30 or \
            rows[1].get("self_speedup", 0.0) < 1.19 or \
            any(row.get("peak_ratio") != 1.0 for row in rows) or \
            fallback_summary.get("speedup", 0.0) < 0.95 or \
            fallback_summary.get("peak_ratio") != 1.0 or \
            "T>=256" not in comparison.get("policy", ""):
        errors.append("block-row causal-softmax keep/fallback gate changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    before = profile.get("before", {})
    after = profile.get("after", {})
    if sum(int(row["Calls"]) for row in kernels) != after.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != \
            after.get("kernel_time_ns") or \
            sum(int(row["Calls"]) for row in api) != after.get("hip_api_calls") or \
            sum(int(row["TotalDurationNs"]) for row in api) != \
            after.get("hip_api_time_ns") or \
            before.get("forward_calls") != 72 or before.get("backward_calls") != 72 or \
            after.get("forward_row_calls") != 72 or \
            after.get("backward_row_calls") != 72 or \
            profile.get("forward_speedup", 0.0) < 4.25 or \
            profile.get("backward_speedup", 0.0) < 4.80 or \
            profile.get("combined_softmax_speedup", 0.0) < 4.45 or \
            profile.get("kernel_time_speedup", 0.0) < 1.27 or \
            profile.get("dispatch_ratio") != 1.0:
        errors.append("block-row causal-softmax retained profile changed")
    return len(formal), after.get("kernel_dispatches", 0)


def validate_block_column_rmsnorm_weight_gradient(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "059-data"
    pilot = [json.loads(line) for line in
             (data / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    fallback = [json.loads(line) for line in
                (data / "fallback128.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in formal}
    if len(pilot) != 1 or len(formal) != 12 or len(keys) != 12 or \
            len(fallback) != 1 or any(row.get("status") != "pass" for row in
                                      pilot + formal + fallback) or \
            any(row.get("context") != 512 for row in formal) or \
            fallback[0].get("context") != 128:
        errors.append("block-column RMSNorm raw protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    fallback_summary = comparison.get("fallback128", {})
    if comparison.get("decision") != "keep" or len(rows) != 2 or \
            rows[0].get("self_speedup", 0.0) < 1.21 or \
            rows[1].get("self_speedup", 0.0) < 1.12 or \
            any(row.get("peak_ratio") != 1.0 for row in rows) or \
            fallback_summary.get("speedup", 0.0) < 0.95 or \
            fallback_summary.get("peak_ratio") != 1.0 or \
            "rows>=256" not in comparison.get("policy", ""):
        errors.append("block-column RMSNorm keep/fallback gate changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    before = profile.get("before", {})
    after = profile.get("after", {})
    if sum(int(row["Calls"]) for row in kernels) != after.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != \
            after.get("kernel_time_ns") or \
            sum(int(row["Calls"]) for row in api) != after.get("hip_api_calls") or \
            sum(int(row["TotalDurationNs"]) for row in api) != \
            after.get("hip_api_time_ns") or \
            before.get("weight_gradient_calls") != 147 or \
            after.get("weight_gradient_row_calls") != 147 or \
            profile.get("weight_gradient_speedup", 0.0) < 16.3 or \
            profile.get("kernel_time_speedup", 0.0) < 1.19 or \
            profile.get("dispatch_ratio") != 1.0:
        errors.append("block-column RMSNorm retained profile changed")
    return len(formal), after.get("kernel_dispatches", 0)


def validate_inference_shape_matrix(errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "060-data"

    def raw(name: str) -> list[dict]:
        return [json.loads(line) for line in
                (data / name / "raw.jsonl").read_text(encoding="utf-8").splitlines()]

    core = raw("core")
    batch = raw("batch")
    long_warm = raw("long-warm")
    long_no_warm = raw("long-no-warm")
    expected_key = lambda row: (
        row.get("model"), row.get("context"), row.get("batch"),
        row.get("workload"), row.get("cache_mode"), row.get("framework"),
        row.get("process_run"))
    if len(core) != 108 or len({expected_key(row) for row in core}) != 108 or \
            any(row.get("status") != "pass" or row.get("warmup") != 1 or
                row.get("steps") != 2 for row in core):
        errors.append("inference core raw protocol changed")
    unsupported = [row for row in batch if row.get("status") == "unsupported"]
    if len(batch) != 48 or len({expected_key(row) for row in batch}) != 48 or \
            len(unsupported) != 6 or any(
                row.get("framework") != "microllm" or row.get("workload") != "decode" or
                row.get("cache_mode") != "cached" or row.get("batch") not in {2, 4, 8}
                for row in unsupported) or any(
                row.get("status") not in {"pass", "unsupported"} for row in batch):
        errors.append("inference batch pass/unsupported contract changed")
    if len(long_warm) != 24 or any(row.get("status") != "pass" or
                                   row.get("warmup") != 1 for row in long_warm):
        errors.append("warm long-context inference protocol changed")
    if len(long_no_warm) != 36 or any(row.get("status") != "pass" or
                                      row.get("warmup") != 0 for row in long_no_warm):
        errors.append("no-warm long-context feasibility protocol changed")

    for row in core + batch + long_warm:
        if row.get("status") != "pass":
            continue
        expected_precision = ("mixed_bf16_weights_fp32_activations"
                              if row.get("framework") == "microllm"
                              else "full_bf16_model")
        if row.get("precision") != expected_precision:
            errors.append("inference precision residency policy changed")
            break
        if row.get("workload") == "decode" and row.get("cache_mode") == "cached":
            actual = int(row.get("kv_cache_actual_bytes", 0))
            theoretical = int(row.get("kv_cache_theoretical_bytes", -1))
            element_bytes = int(row.get("kv_cache_element_bytes", 0))
            expected_element = 4 if row.get("framework") == "microllm" else 2
            utilization = float(row.get("kv_cache_utilization", 0.0))
            if actual <= 0 or actual != theoretical or element_bytes != expected_element or \
                    not (0.0 < utilization <= 1.0):
                errors.append("inference KV Storage/formula contract changed")
                break

    core_summary = json.loads((data / "core" / "summary.json").read_text(encoding="utf-8"))
    batch_summary = json.loads((data / "batch" / "summary.json").read_text(encoding="utf-8"))
    warm_summary = json.loads(
        (data / "long-warm" / "summary.json").read_text(encoding="utf-8"))
    no_warm_summary = json.loads(
        (data / "long-no-warm" / "summary.json").read_text(encoding="utf-8"))
    if core_summary.get("status") != "pass" or len(core_summary.get("rows", [])) != 18 or \
            any(row.get("cross_framework_tokens_equal") is not True for row in
                core_summary["rows"] if row.get("workload") == "decode"):
        errors.append("inference core summary/token gate changed")
    if batch_summary.get("status") != "complete_with_recorded_limits" or \
            len(batch_summary.get("rows", [])) != 24:
        errors.append("inference batch summary limit gate changed")
    if warm_summary.get("status") != "pass" or len(warm_summary.get("rows", [])) != 12 or \
            min(row.get("throughput_ratio_microllm_over_pytorch", 1.0)
                for row in warm_summary["rows"] if row.get("workload") == "prefill") >= 0.01:
        errors.append("warm long-context inference failure boundary changed")
    if no_warm_summary.get("status") != "pass" or len(no_warm_summary.get("rows", [])) != 18:
        errors.append("no-warm long-context summary changed")

    invalidation = json.loads((data / "invalidation.json").read_text(encoding="utf-8"))
    if invalidation.get("decision") != "invalid" or len(invalidation.get("reasons", [])) != 4:
        errors.append("inference invalid-pilot decision changed")
    smoke = raw("final-schema-smoke")
    smoke_summary = json.loads(
        (data / "final-schema-smoke" / "summary.json").read_text(encoding="utf-8"))
    if len(smoke) != 6 or any(row.get("status") != "pass" for row in smoke) or \
            smoke_summary.get("status") != "pass" or len(smoke_summary.get("rows", [])) != 3:
        errors.append("final inference schema smoke changed")
    else:
        prefill = next(row for row in smoke_summary["rows"] if row["workload"] == "prefill")
        cached = next(row for row in smoke_summary["rows"] if row["cache_mode"] == "cached")
        if prefill.get("prefill_top_token_equal") is not True or \
                prefill.get("prefill_top_logit_abs_difference", 1.0) >= 0.1 or \
                cached.get("microllm_mean_cache_prepare_ms", 0.0) <= 0.0 or \
                cached.get("pytorch_mean_cache_prepare_ms", 0.0) <= 0.0:
            errors.append("final inference top-logit/cache-prepare gate changed")
    return len(core), len(batch), len(long_warm), len(long_no_warm)


def validate_batched_prefill_inference(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "061-data"
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("context"), row.get("framework"),
             row.get("process_run")) for row in formal}
    if len(formal) != 24 or len(keys) != 24 or any(
            row.get("status") != "pass" or row.get("workload") != "prefill" or
            row.get("cache_mode") != "uncached" or row.get("context") not in {512, 1024}
            for row in formal):
        errors.append("batched prefill formal protocol changed")
    summary = json.loads((data / "formal" / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or len(summary.get("rows", [])) != 4 or any(
            row.get("prefill_top_token_equal") is not True or
            row.get("prefill_top_logit_abs_difference", 1.0) >= 0.2
            for row in summary.get("rows", [])):
        errors.append("batched prefill official top-logit gate changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = comparison.get("rows", [])
    if comparison.get("decision") != "keep" or len(rows) != 4 or any(
            row.get("self_speedup", 0.0) < 6.7 or
            row.get("top_token_equal") is not True or
            row.get("top_logit_abs_difference", 1.0) >= 0.2
            for row in rows) or \
            comparison.get("fallback128", {}).get("speedup", 0.0) < 1.7:
        errors.append("batched prefill keep/fallback contract changed")
    pilots = []
    for name in ("route-unused-pilot.jsonl", "integrated-pilot.jsonl",
                 "fallback128.jsonl"):
        pilots.extend(json.loads(line) for line in
                      (data / name).read_text(encoding="utf-8").splitlines())
    if len(pilots) != 3 or any(row.get("status") != "pass" for row in pilots):
        errors.append("batched prefill pilot evidence changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    for phase in ("before", "after"):
        with (data / f"profile-{phase}" / "kernel-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        with (data / f"profile-{phase}" / "hip-api-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            api = list(csv.DictReader(stream))
        recorded = profile[phase]
        if sum(int(row["Calls"]) for row in kernels) != recorded["kernel_dispatches"] or \
                sum(int(row["TotalDurationNs"]) for row in kernels) != \
                recorded["kernel_time_ns"] or \
                sum(int(row["Calls"]) for row in api) != recorded["hip_api_calls"] or \
                sum(int(row["TotalDurationNs"]) for row in api) != \
                recorded["hip_api_time_ns"]:
            errors.append(f"batched prefill {phase} profiler aggregate changed")
    if profile.get("kernel_time_speedup", 0.0) < 5.1 or \
            profile["before"].get("readable_attention_matmul_calls") != 144 or \
            profile["after"].get("readable_attention_matmul_calls") != 0 or \
            profile.get("additional_library_gemm_calls") != 144:
        errors.append("batched prefill profiler routing gate changed")
    return len(formal), profile["after"].get("kernel_dispatches", 0)


def validate_full_prefill_cache(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "062-data"
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    long_rows = [json.loads(line) for line in
                 (data / "long" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("context"), row.get("framework"),
             row.get("process_run")) for row in formal}
    if len(formal) != 36 or len(keys) != 36 or any(
            row.get("status") != "pass" or row.get("workload") != "decode" or
            row.get("cache_mode") != "cached" or row.get("context") not in {8, 512, 1024}
            for row in formal) or any(
                row.get("cache_prefill_mode") != "full" for row in formal
                if row.get("framework") == "microllm"):
        errors.append("full prefill-cache formal protocol changed")
    if len(long_rows) != 4 or any(row.get("status") != "pass" or
                                  row.get("context") != 2048 for row in long_rows):
        errors.append("full prefill-cache T2048 protocol changed")
    for row in formal + long_rows:
        if row.get("status") != "pass" or row.get("cache_mode") != "cached":
            continue
        if int(row.get("kv_cache_actual_bytes", 0)) != \
                int(row.get("kv_cache_theoretical_bytes", -1)) or \
                row.get("generated_tokens") is None:
            errors.append("full prefill-cache KV/token formula changed")
            break
    summary = json.loads((data / "formal" / "summary.json").read_text(encoding="utf-8"))
    long_summary = json.loads((data / "long" / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or len(summary.get("rows", [])) != 6 or any(
            row.get("cross_framework_tokens_equal") is not True or
            row.get("microllm_mean_cache_prepare_ms", 0.0) <= 0.0 or
            row.get("microllm_mean_end_to_end_generation_ms", 0.0) <= 0.0
            for row in summary.get("rows", [])):
        errors.append("full prefill-cache formal summary changed")
    if long_summary.get("status") != "pass" or len(long_summary.get("rows", [])) != 2:
        errors.append("full prefill-cache T2048 summary changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    if comparison.get("decision") != "keep" or len(comparison.get("rows", [])) != 6 or \
            len(comparison.get("long_2048", [])) != 2 or \
            len(comparison.get("falsification_events", [])) != 2 or \
            comparison.get("same_window_profile", {}).get("prepare_speedup", 0.0) < 270.0:
        errors.append("full prefill-cache keep/falsification contract changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    for phase in ("token", "full"):
        with (data / f"profile-{phase}" / "kernel-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        with (data / f"profile-{phase}" / "hip-api-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            api = list(csv.DictReader(stream))
        recorded = profile[phase]
        if sum(int(row["Calls"]) for row in kernels) != recorded["kernel_dispatches"] or \
                sum(int(row["TotalDurationNs"]) for row in kernels) != \
                recorded["kernel_time_ns"] or \
                sum(int(row["Calls"]) for row in api) != recorded["hip_api_calls"] or \
                sum(int(row["TotalDurationNs"]) for row in api) != \
                recorded["hip_api_time_ns"]:
            errors.append(f"full prefill-cache {phase} profile aggregate changed")
    if profile.get("prepare_speedup", 0.0) < 270.0 or \
            profile.get("kernel_time_speedup", 0.0) < 110.0 or \
            profile.get("kernel_dispatch_reduction", 0.0) < 150.0 or \
            profile.get("token_cached_attention_calls") != 24624:
        errors.append("full prefill-cache profiler gate changed")
    return len(formal), len(long_rows), profile["full"].get("kernel_dispatches", 0)


def validate_device_row_argmax(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "063-data"
    device = [json.loads(line) for line in
              (data / "batch" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    host = [json.loads(line) for line in
            (data / "host-batch" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    transfer = [json.loads(line) for line in
                (data / "transfer-control.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(device) != 16 or len(host) != 16 or any(
            row.get("status") != "pass" or row.get("workload") != "decode" or
            row.get("cache_mode") != "uncached" for row in device + host):
        errors.append("row-wise argmax batch raw protocol changed")
    if any(row.get("batch_argmax_mode") != "device" for row in device
           if row.get("framework") == "microllm") or any(
            row.get("batch_argmax_mode") != "host" for row in host
            if row.get("framework") == "microllm"):
        errors.append("row-wise argmax explicit mode contract changed")
    device_summary = json.loads((data / "batch" / "summary.json").read_text(encoding="utf-8"))
    host_summary = json.loads(
        (data / "host-batch" / "summary.json").read_text(encoding="utf-8"))
    if device_summary.get("status") != "pass" or host_summary.get("status") != "pass" or \
            len(device_summary.get("rows", [])) != 8 or \
            len(host_summary.get("rows", [])) != 8 or any(
                row.get("cross_framework_tokens_equal") is not True
                for row in device_summary.get("rows", [])):
        errors.append("row-wise argmax batch summary/token gate changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    if comparison.get("decision") != "keep" or len(comparison.get("rows", [])) != 8 or \
            any(row.get("speedup", 0.0) < 1.12 or row.get("peak_ratio") != 1.0 or
                row.get("tokens_equal") is not True for row in comparison.get("rows", [])) or \
            comparison.get("transfer_control", {}).get("d2h_byte_reduction") != 151936.0:
        errors.append("row-wise argmax keep/transfer contract changed")
    if len(transfer) != 2 or transfer[0].get("batch_argmax_mode") != "host" or \
            transfer[1].get("batch_argmax_mode") != "device" or \
            transfer[0].get("measured_d2h_bytes") != 38895616 or \
            transfer[1].get("measured_d2h_bytes") != 256:
        errors.append("row-wise argmax direct transfer control changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    for phase in ("host", "device"):
        with (data / f"profile-{phase}" / "kernel-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        with (data / f"profile-{phase}" / "hip-api-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            api = list(csv.DictReader(stream))
        recorded = profile[phase]
        if sum(int(row["Calls"]) for row in kernels) != recorded["kernel_dispatches"] or \
                sum(int(row["TotalDurationNs"]) for row in kernels) != \
                recorded["kernel_time_ns"] or \
                sum(int(row["Calls"]) for row in api) != recorded["hip_api_calls"] or \
                sum(int(row["TotalDurationNs"]) for row in api) != \
                recorded["hip_api_time_ns"]:
            errors.append(f"row-wise argmax {phase} profile aggregate changed")
    if profile.get("profile_throughput_speedup", 0.0) < 2.0 or \
            profile["host"].get("rocprof_d2h_calls") != 12 or \
            profile["device"].get("rocprof_d2h_calls") != 0 or \
            profile["device"].get("argmax_row_calls") != 12:
        errors.append("row-wise argmax profiler gate changed")
    return len(device), len(host), profile["device"].get("kernel_dispatches", 0)


def validate_batched_kv_cache(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "064-data"
    formal = [json.loads(line) for line in
              (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    pilot = [json.loads(line) for line in
             (data / "pilot" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("batch"), row.get("framework"),
             row.get("process_run")) for row in formal}
    if len(formal) != 48 or len(keys) != 48 or any(
            row.get("status") != "pass" or row.get("workload") != "decode" or
            row.get("cache_mode") != "cached" or row.get("batch") not in {1, 2, 4, 8}
            for row in formal):
        errors.append("batched KV formal raw protocol changed")
    if len(pilot) != 16 or any(row.get("status") != "pass" for row in pilot):
        errors.append("batched KV pilot protocol changed")
    for row in formal:
        if row.get("status") != "pass":
            continue
        if int(row.get("kv_cache_actual_bytes", 0)) != \
                int(row.get("kv_cache_theoretical_bytes", -1)):
            errors.append("batched KV byte formula changed")
            break
    summary = json.loads((data / "formal" / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or len(summary.get("rows", [])) != 8 or any(
            row.get("cross_framework_tokens_equal") is not True
            for row in summary.get("rows", [])):
        errors.append("batched KV formal summary/token gate changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    if comparison.get("decision") != "keep" or len(comparison.get("rows", [])) != 8 or \
            comparison.get("before", {}).get("unsupported_records") != 6 or \
            comparison.get("batch8_efficiency", {}).get("qwen", 0.0) < 0.98 or \
            comparison.get("batch8_efficiency", {}).get("deepseek", 0.0) < 0.99 or any(
                row.get("tokens_equal") is not True for row in comparison.get("rows", [])):
        errors.append("batched KV keep/scaling contract changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    with (data / "profile" / "memory-copy-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        copies = list(csv.DictReader(stream))
    if sum(int(row["Calls"]) for row in kernels) != profile.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != \
            profile.get("kernel_time_ns") or \
            sum(int(row["Calls"]) for row in api) != profile.get("hip_api_calls") or \
            sum(int(row["TotalDurationNs"]) for row in api) != \
            profile.get("hip_api_time_ns") or \
            sum(int(row["Calls"]) for row in copies
                if row["Name"] == "MEMORY_COPY_DEVICE_TO_HOST") != \
            profile.get("memory_copy_d2h_calls") or \
            profile.get("measured_d2h_bytes") != 256 or \
            profile.get("cached_attention_calls") != 216:
        errors.append("batched KV retained profile contract changed")
    return len(formal), len(pilot), profile.get("kernel_dispatches", 0)


def validate_bf16_kv_cache(errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "065-data"
    baseline = [json.loads(line) for line in
                (data / "baseline-release" / "raw.jsonl").read_text(
                    encoding="utf-8").splitlines()]
    formal = [json.loads(line) for line in
              (data / "formal-release" / "raw.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    for name, rows, element_bytes in (("baseline", baseline, 4),
                                      ("formal", formal, 2)):
        keys = {(row.get("model"), row.get("context"), row.get("batch"),
                 row.get("framework"), row.get("process_run")) for row in rows}
        if len(rows) != 72 or len(keys) != 72 or any(
                row.get("status") != "pass" or row.get("workload") != "decode" or
                row.get("cache_mode") != "cached" or row.get("context") not in
                {32, 512, 2048} or row.get("batch") not in {1, 8}
                for row in rows):
            errors.append(f"BF16 KV {name} Release protocol changed")
        micro_rows = [row for row in rows if row.get("framework") == "microllm"]
        if any(int(row.get("kv_cache_actual_bytes", 0)) !=
               int(row.get("kv_cache_theoretical_bytes", -1)) or
               int(row.get("kv_cache_element_bytes", -1)) != element_bytes
               for row in micro_rows):
            errors.append(f"BF16 KV {name} byte/dtype contract changed")
    comparison = json.loads((data / "release-comparison.json").read_text(
        encoding="utf-8"))
    comparison_rows = comparison.get("rows", [])
    if comparison.get("decision") != "keep_opt_in" or len(comparison_rows) != 12 or any(
            row.get("cache_byte_reduction") != 2.0 or
            row.get("micro_tokens_equal") is not True or
            float(row.get("throughput_ratio_bf16_over_fp32", 0.0)) < 0.99
            for row in comparison_rows) or sum(
                float(row["throughput_ratio_bf16_over_fp32"]) >= 1.0
                for row in comparison_rows) != 11:
        errors.append("BF16 KV Release keep/throughput/token contract changed")
    precision = json.loads((data / "precision" / "summary.json").read_text(
        encoding="utf-8"))
    precision_rows = precision.get("records", [])
    failed = [row for row in precision_rows if row.get("status") == "failed"]
    if precision.get("status") != "failed" or len(precision_rows) != 12 or \
            len(failed) != 1 or failed[0].get("model") != \
            "deepseek-r1-distill-qwen-1.5b" or failed[0].get("context") != 512 or \
            failed[0].get("batch") != 1 or any(
                row.get("all_logits_finite") is not True or
                row.get("top_tokens_equal") is not True or
                row.get("generated_tokens_equal") is not True or
                row.get("cache_byte_reduction") != 2.0 for row in precision_rows):
        errors.append("BF16 KV precision failure contract changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    for phase in ("before", "after"):
        with (data / f"profile-{phase}" / "kernel-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        with (data / f"profile-{phase}" / "hip-api-stats.csv").open(
                encoding="utf-8", newline="") as stream:
            api = list(csv.DictReader(stream))
        recorded = profile.get(phase, {})
        if sum(int(row["Calls"]) for row in kernels) != \
                recorded.get("kernel_dispatches") or sum(
                    int(row["TotalDurationNs"]) for row in kernels) != \
                recorded.get("kernel_time_ns") or sum(
                    int(row["Calls"]) for row in api) != recorded.get("hip_api_calls") or \
                sum(int(row["TotalDurationNs"]) for row in api) != \
                recorded.get("hip_api_time_ns"):
            errors.append(f"BF16 KV {phase} profile aggregate changed")
    if profile.get("ratios", {}).get("cached_attention_speedup", 0.0) < 1.15 or \
            profile.get("ratios", {}).get("additional_cast_calls") != 96:
        errors.append("BF16 KV retained profile explanation changed")
    return len(baseline), len(formal), len(precision_rows), \
        profile.get("after", {}).get("kernel_dispatches", 0)


def validate_fused_prefix_pair_discard(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "066-data"
    formal = [json.loads(line) for line in
              (data / "formal-release" / "raw.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("context"), row.get("batch"),
             row.get("framework"), row.get("process_run")) for row in formal}
    if len(formal) != 72 or len(keys) != 72 or any(
            row.get("status") != "pass" or row.get("workload") != "decode" or
            row.get("cache_mode") != "cached" for row in formal):
        errors.append("fused prefix-pair formal protocol changed")
    micro = [row for row in formal if row.get("framework") == "microllm"]
    if len(micro) != 36 or any(
            int(row.get("measured_d2d_calls", -1)) != 0 or
            int(row.get("measured_d2d_bytes", -1)) != 0 for row in micro):
        errors.append("fused prefix-pair zero-D2D contract changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    comparison_rows = comparison.get("rows", [])
    qwen_long = next((row for row in comparison_rows
                      if row.get("model") == "qwen2.5-0.5b" and
                      row.get("context") == 2048 and row.get("batch") == 8), {})
    if comparison.get("decision") != "discard" or len(comparison_rows) != 12 or \
            float(qwen_long.get("prepare_speedup", 1.0)) >= 0.77 or \
            float(qwen_long.get("end_to_end_speedup", 1.0)) >= 0.83 or any(
                row.get("tokens_equal") is not True for row in comparison_rows):
        errors.append("fused prefix-pair discard/formal failure changed")
    precision = json.loads((data / "precision" / "summary.json").read_text(
        encoding="utf-8"))
    precision_rows = precision.get("records", [])
    failed = [row for row in precision_rows if row.get("status") == "failed"]
    if precision.get("status") != "failed" or len(precision_rows) != 12 or \
            len(failed) != 1 or failed[0].get("context") != 512 or \
            failed[0].get("batch") != 1:
        errors.append("fused prefix-pair precision inheritance changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    with (data / "profile" / "hip-api-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        api = list(csv.DictReader(stream))
    candidate = profile.get("candidate", {})
    if sum(int(row["Calls"]) for row in kernels) != \
            candidate.get("kernel_dispatches") or sum(
                int(row["TotalDurationNs"]) for row in kernels) != \
            candidate.get("kernel_time_ns") or sum(
                int(row["Calls"]) for row in api) != candidate.get("hip_api_calls") or \
            candidate.get("prefix_pair_calls") != 48 or \
            profile.get("ratios", {}).get("full_kernel_speedup", 0.0) < 1.02:
        errors.append("fused prefix-pair retained profile changed")
    for source in (REPOSITORY / "include/microllm/ops/ops.h",
                   REPOSITORY / "src/ops/hip/basic_kernels.hip"):
        if "kv_cache_store_prefix_pair" in source.read_text(encoding="utf-8"):
            errors.append("discarded prefix-pair implementation remains in source")
    return len(formal), len(precision_rows), candidate.get("kernel_dispatches", 0)


def validate_mixed_layer_kv_policy(errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "067-data"
    formal = [json.loads(line) for line in
              (data / "formal-release" / "raw.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("context"), row.get("batch"),
             row.get("framework"), row.get("process_run")) for row in formal}
    if len(formal) != 72 or len(keys) != 72 or any(
            row.get("status") != "pass" for row in formal):
        errors.append("mixed-layer KV formal protocol changed")
    micro = [row for row in formal if row.get("framework") == "microllm"]
    if len(micro) != 36 or any(
            row.get("kv_cache_fp32_layer_policy") != "1" or
            int(row.get("kv_cache_fp32_layers", 0)) != 1 or
            int(row.get("kv_cache_actual_bytes", 0)) !=
            int(row.get("kv_cache_theoretical_bytes", -1)) or
            (int(row.get("kv_cache_bf16_layers", 0)) not in {23, 27})
            for row in micro):
        errors.append("mixed-layer KV dtype/byte policy changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    comparison_rows = comparison.get("rows", [])
    deepseek_long = next((row for row in comparison_rows
                          if row.get("model") == "deepseek-r1-distill-qwen-1.5b" and
                          row.get("context") == 2048 and row.get("batch") == 8), {})
    if comparison.get("decision") != "keep_explicit" or len(comparison_rows) != 12 or any(
            float(row.get("throughput_ratio_hybrid_over_uniform", 0.0)) < 0.975 or
            row.get("tokens_equal") is not True for row in comparison_rows) or \
            float(deepseek_long.get("end_to_end_speedup", 1.0)) >= 0.87:
        errors.append("mixed-layer KV explicit keep/tradeoff changed")
    precision = json.loads((data / "layer1-precision" / "summary.json").read_text(
        encoding="utf-8"))
    precision_rows = precision.get("records", [])
    if precision.get("status") != "pass" or len(precision_rows) != 12 or any(
            row.get("status") != "pass" or row.get("all_logits_finite") is not True or
            row.get("top_tokens_equal") is not True or
            row.get("generated_tokens_equal") is not True or
            row.get("bf16_fp32_layer_policy") != "1"
            for row in precision_rows):
        errors.append("mixed-layer KV 12-shape precision pass changed")
    layer0 = json.loads((data / "layer0-full" / "summary.json").read_text(
        encoding="utf-8"))
    layer0_failed = [row for row in layer0.get("records", [])
                     if row.get("status") == "failed"]
    if layer0.get("status") != "failed" or len(layer0_failed) != 1 or \
            layer0_failed[0].get("context") != 32 or \
            layer0_failed[0].get("batch") != 1:
        errors.append("mixed-layer KV layer0 rebuttal changed")
    search = json.loads((data / "search-summary.json").read_text(encoding="utf-8"))
    if search.get("selected_policy") != "1" or len(search.get("rows", [])) != 16:
        errors.append("mixed-layer KV policy search changed")
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    with (data / "profile" / "kernel-stats.csv").open(
            encoding="utf-8", newline="") as stream:
        kernels = list(csv.DictReader(stream))
    hybrid = profile.get("hybrid", {})
    if sum(int(row["Calls"]) for row in kernels) != hybrid.get("kernel_dispatches") or \
            sum(int(row["TotalDurationNs"]) for row in kernels) != \
            hybrid.get("kernel_time_ns") or hybrid.get("cached_attention_bf16_calls") != 138 or \
            hybrid.get("cached_attention_fp32_calls") != 6:
        errors.append("mixed-layer KV profile/layer decomposition changed")
    if "layer_dtype" not in (REPOSITORY / "include/microllm/inference/kv_cache.h").read_text(
            encoding="utf-8"):
        errors.append("mixed-layer KV public policy API is missing")
    return len(formal), len(precision_rows), len(search.get("rows", [])), \
        hybrid.get("kernel_dispatches", 0)


def validate_targeted_prefix_pair_discard(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "068-data"
    for name in ("reference", "paired"):
        rows = [json.loads(line) for line in
                (data / name / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != 6 or any(row.get("status") != "pass" for row in rows):
            errors.append(f"targeted prefix-pair {name} protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    reference = comparison.get("reference", {})
    paired = comparison.get("paired", {})
    ratios = comparison.get("ratios", {})
    if comparison.get("decision") != "discard" or \
            comparison.get("tokens_equal") is not True or \
            ratios.get("prepare_speedup", 1.0) >= 1.0 or \
            ratios.get("end_to_end_speedup", 1.0) >= 1.0 or \
            any(int(value) != 4480 for value in
                reference.get("measured_d2d_calls", [])) or \
            any(int(value) != 4320 for value in
                paired.get("measured_d2d_calls", [])):
        errors.append("targeted prefix-pair same-binary discard changed")
    precision = json.loads((data / "precision" / "summary.json").read_text(
        encoding="utf-8"))
    records = precision.get("records", [])
    if precision.get("status") != "pass" or len(records) != 1 or \
            records[0].get("status") != "pass":
        errors.append("targeted prefix-pair precision control changed")
    for source in (REPOSITORY / "include/microllm/inference/kv_cache.h",
                   REPOSITORY / "include/microllm/ops/ops.h"):
        if "PrefixStoreImplementation" in source.read_text(encoding="utf-8") or \
                "kv_cache_store_prefix_pair_fp32" in source.read_text(encoding="utf-8"):
            errors.append("discarded targeted prefix-pair route remains in source")
    return 12, len(records)


def validate_same_binary_kv_policy(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "069-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("context"), row.get("batch"),
             row.get("policy"), row.get("process_run")) for row in raw}
    if len(raw) != 72 or len(keys) != 72 or any(
            row.get("status") != "pass" or row.get("policy") not in
            {"uniform", "candidate"} for row in raw):
        errors.append("same-binary KV policy raw protocol changed")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    deepseek = [row for row in rows
                if row.get("model") == "deepseek-r1-distill-qwen-1.5b"]
    deepseek_long = next((row for row in deepseek
                          if row.get("context") == 2048 and row.get("batch") == 8), {})
    if summary.get("status") != "pass" or len(rows) != 12 or any(
            row.get("tokens_equal") is not True for row in rows) or any(
                float(row.get("throughput_ratio_candidate_over_uniform", 0.0)) < 0.99
                for row in deepseek) or \
            float(deepseek_long.get("end_to_end_speedup", 0.0)) < 1.0:
        errors.append("same-binary KV policy conclusion changed")
    old = json.loads((ROOT / "experiments/067-data/comparison.json").read_text(
        encoding="utf-8"))
    old_long = next((row for row in old.get("rows", [])
                     if row.get("model") == "deepseek-r1-distill-qwen-1.5b" and
                     row.get("context") == 2048 and row.get("batch") == 8), {})
    if float(old_long.get("end_to_end_speedup", 1.0)) >= 0.87:
        errors.append("cross-window KV policy rebuttal baseline changed")
    if not (REPOSITORY / "benchmarks/single_gpu/compare_kv_cache_policies.py").is_file():
        errors.append("same-binary KV policy runner is missing")
    return len(raw), len(rows)


def validate_kv_policy_prompt_robustness(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "070-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    layer1 = summary.get("layer1", {})
    first4 = summary.get("first4", {})
    performance = summary.get("performance", {})
    if summary.get("decision") != "keep_first4_robust_strict" or \
            layer1.get("records") != 14 or layer1.get("failed") != 5 or \
            float(layer1.get("worst_rmse", 0.0)) < 2.9 or \
            first4.get("records") != 14 or first4.get("pass") != 14 or \
            float(first4.get("worst_rmse", 1.0)) >= 0.05 or \
            first4.get("cache_byte_reduction") != 1.75 or \
            performance.get("rows") != 6 or \
            performance.get("all_tokens_equal") is not True or \
            float(performance.get("minimum_throughput_ratio", 0.0)) < 0.969 or \
            float(performance.get("minimum_end_to_end_speedup", 0.0)) < 0.971:
        errors.append("KV policy prompt-robustness decision changed")
    layer1_records = []
    for path in sorted((data / "layer1-patterns").glob("*/summary.json")):
        layer1_records.extend(json.loads(path.read_text(encoding="utf-8")).get(
            "records", []))
    first4_records = []
    for path in sorted((data / "first4-patterns").glob("*/summary.json")):
        first4_records.extend(json.loads(path.read_text(encoding="utf-8")).get(
            "records", []))
    if len(layer1_records) != 14 or sum(
            row.get("status") == "failed" for row in layer1_records) != 5 or \
            not any(row.get("generated_tokens_equal") is False
                    for row in layer1_records) or len(first4_records) != 14 or any(
                        row.get("status") != "pass" for row in first4_records):
        errors.append("KV policy prompt raw summaries changed")
    performance_raw = [json.loads(line) for line in
                       (data / "first4-performance" / "raw.jsonl").read_text(
                           encoding="utf-8").splitlines()]
    performance_summary = json.loads((data / "first4-performance" / "summary.json").read_text(
        encoding="utf-8"))
    if len(performance_raw) != 36 or performance_summary.get("status") != "pass" or \
            len(performance_summary.get("rows", [])) != 6:
        errors.append("first-four KV performance pairing changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/compare_kv_cache_precision.py").read_text(
        encoding="utf-8")
    if "--token-pattern" not in runner or "rotated" not in runner or "ramp" not in runner:
        errors.append("KV precision token-pattern runner is missing")
    return len(layer1_records), len(first4_records), len(performance_raw)


def validate_qwen_kv_prompt_failure(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "071-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    uniform = summary.get("uniform", {})
    first2 = summary.get("first2", {})
    constant_long = summary.get("constant_t2048", {})
    if summary.get("decision") != "require_fp32_fallback_for_constant_t2048" or \
            uniform.get("records") != 14 or uniform.get("failed") != 3 or \
            first2.get("records") != 14 or first2.get("failed") != 1 or \
            constant_long.get("only_all_fp32_passes") is not True or \
            constant_long.get("token_divergence") is not True or \
            float(constant_long.get("worst_rmse", 0.0)) < 3.1:
        errors.append("Qwen KV prompt failure/fallback decision changed")
    uniform_records = []
    for path in sorted((data / "uniform-patterns").glob("*/summary.json")):
        uniform_records.extend(json.loads(path.read_text(encoding="utf-8")).get(
            "records", []))
    first2_records = []
    for path in sorted((data / "first2-patterns").glob("*/summary.json")):
        first2_records.extend(json.loads(path.read_text(encoding="utf-8")).get(
            "records", []))
    search_records = []
    for path in sorted((data / "constant-search").glob("*/summary.json")):
        search_records.extend(json.loads(path.read_text(encoding="utf-8")).get(
            "records", []))
    long_search = [row for row in search_records if row.get("context") == 2048]
    if len(uniform_records) != 14 or sum(
            row.get("status") == "failed" for row in uniform_records) != 3 or \
            len(first2_records) != 14 or sum(
                row.get("status") == "failed" for row in first2_records) != 1 or \
            len(search_records) != 9 or len(long_search) != 4 or sum(
                row.get("status") == "pass" for row in long_search) != 1 or \
            not any(row.get("generated_tokens_equal") is False for row in long_search):
        errors.append("Qwen KV prompt raw summaries changed")
    return len(uniform_records), len(first2_records), len(search_records)


def validate_reference_serving_scheduler(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "072-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("device"), row.get("requests"), row.get("process_run"))
            for row in raw}
    if len(raw) != 24 or len(keys) != 24 or any(
            row.get("status") != "pass" or row.get("scheduler") != "serial_reference" or
            row.get("outputs_equal") is not True or row.get("requests") not in {1, 2, 4, 8}
            for row in raw):
        errors.append("reference serving scheduler raw protocol changed")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    hip_rows = [row for row in rows if row.get("device") == "hip"]
    if summary.get("status") != "pass" or summary.get("parameter_count") != 106816 or \
            len(rows) != 8 or any(row.get("outputs_equal") is not True for row in rows) or \
            len(hip_rows) != 4 or any(
                float(row.get("scheduler_over_sequential", 0.0)) < 0.98 or
                float(row.get("scheduler_over_sequential", 2.0)) > 1.02
                for row in hip_rows):
        errors.append("reference serving scheduler baseline changed")
    for path in (REPOSITORY / "include/microllm/inference/scheduler.h",
                 REPOSITORY / "src/inference/scheduler.cpp",
                 REPOSITORY / "benchmarks/end_to_end/benchmark_scheduler.cpp"):
        if not path.is_file():
            errors.append(f"missing serving scheduler artifact: {path.name}")
    return len(raw), len(rows)


def validate_static_batch_generation(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "073-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("device"), row.get("requests"), row.get("process_run"))
            for row in raw}
    if len(raw) != 24 or len(keys) != 24 or any(
            row.get("status") != "pass" or row.get("static_batch_enabled") is not True or
            row.get("static_outputs_equal") is not True for row in raw):
        errors.append("static batch generation raw protocol changed")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    hip_b8 = next((row for row in rows
                   if row.get("device") == "hip" and row.get("requests") == 8), {})
    if summary.get("status") != "pass" or len(rows) != 8 or any(
            row.get("outputs_equal") is not True for row in rows) or \
            float(hip_b8.get("speedup", 0.0)) < 7.3 or \
            float(hip_b8.get("scaling_efficiency", 0.0)) < 0.90:
        errors.append("static batch generation scaling contract changed")
    generator = (REPOSITORY / "include/microllm/inference/generator.h").read_text(
        encoding="utf-8")
    if "generate_batch" not in generator:
        errors.append("public static batch generation API is missing")
    return len(raw), len(rows)


def validate_admission_batch_scheduler(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "074-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("device"), row.get("requests"), row.get("process_run"))
            for row in raw}
    if len(raw) != 30 or len(keys) != 30 or any(
            row.get("status") != "pass" or row.get("admission_enabled") is not True or
            row.get("admission_outputs_equal") is not True for row in raw):
        errors.append("admission batch scheduler raw protocol changed")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    hip = {row.get("requests"): row for row in rows if row.get("device") == "hip"}
    if summary.get("status") != "pass" or len(rows) != 10 or any(
            row.get("outputs_equal") is not True for row in rows) or \
            float(hip.get(4, {}).get("speedup", 0.0)) < 3.7 or \
            int(hip.get(8, {}).get("batch_groups", 0)) != 2 or \
            int(hip.get(16, {}).get("batch_groups", 0)) != 4 or \
            int(hip.get(16, {}).get("maximum_batch_size", 0)) != 4 or \
            float(hip.get(16, {}).get("admission_tokens_per_second", 0.0)) < 1200:
        errors.append("admission batch scheduler grouping/scaling changed")
    scheduler = (REPOSITORY / "include/microllm/inference/scheduler.h").read_text(
        encoding="utf-8")
    if "AdmissionBatchScheduler" not in scheduler:
        errors.append("public admission batch scheduler API is missing")
    return len(raw), len(rows)


def validate_request_cancellation(errors: list[str]) -> tuple[int, int, int]:
    summary = json.loads((ROOT / "experiments" / "075-data" / "summary.json").read_text(
        encoding="utf-8"))
    cpu = summary.get("scheduler_cpu_tests", {})
    hip = summary.get("scheduler_hip_tests", {})
    sanitizer = summary.get("sanitizer_tests", {})
    contracts = summary.get("contracts", {})
    if summary.get("status") != "pass" or cpu.get("passed") != cpu.get("total") or \
            hip.get("passed") != hip.get("total") or \
            sanitizer.get("passed") != sanitizer.get("total") or \
            contracts.get("terminal_idempotent") is not True or \
            contracts.get("cache_bytes_after_cancel") != 0 or \
            contracts.get("cancelled_row_excluded_from_batch") is not True:
        errors.append("request cancellation lifecycle evidence changed")
    scheduler = (REPOSITORY / "include/microllm/inference/scheduler.h").read_text(
        encoding="utf-8")
    if "Cancelled" not in scheduler or "bool cancel(RequestId id)" not in scheduler:
        errors.append("public request cancellation API is missing")
    return int(cpu.get("total", 0)), int(hip.get("total", 0)), \
        int(sanitizer.get("total", 0))


def validate_expanded_inference_service_matrix(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "076-data"
    collections = []
    expected = {"prefill": 48, "cached-fp32": 48, "cached-bf16": 24}
    for name, count in expected.items():
        raw = [json.loads(line) for line in
               (data / name / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(raw) != count or any(row.get("status") != "pass" for row in raw):
            errors.append(f"expanded inference {name} raw protocol changed")
        keys = {(row.get("model"), row.get("context"), row.get("batch"),
                 row.get("framework")) for row in raw}
        if len(keys) != count:
            errors.append(f"expanded inference {name} shape keys changed")
        collections.append(raw)
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    reductions = comparison.get("bf16_comparison", [])
    mismatches = comparison.get("token_mismatches", [])
    if comparison.get("status") != "pass" or \
            comparison.get("record_counts", {}).get("total") != 120 or \
            len(reductions) != 12 or any(
                abs(float(row.get("cache_reduction", 0.0)) - 2.0) > 1.0e-9
                for row in reductions) or len(mismatches) != 8:
        errors.append("expanded inference comparison contract changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/hf_inference_shape_matrix.py").read_text(
        encoding="utf-8")
    for token in ("MATRIX_SUITES", "batch_efficiency", "peak_bytes_per_request",
                  "first_sequence_difference"):
        if token not in runner:
            errors.append(f"expanded inference runner is missing {token}")
    for path in (REPOSITORY / "tests/inference/shape_matrix_test.cpp",
                 REPOSITORY / "tests/inference/hip_shape_matrix_test.cpp"):
        if not path.is_file():
            errors.append(f"missing executable inference matrix gate: {path.name}")
    return expected["prefill"], expected["cached-fp32"], expected["cached-bf16"]


def validate_serving_last_logit_prefill(errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "077-data"
    counts = {}
    for name, expected_count, mode in (("full-logits", 12, "full"),
                                       ("last-logit", 12, "last"),
                                       ("shape-survey-last", 48, "last")):
        raw = [json.loads(line) for line in
               (data / name / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
        keys = {(row.get("model"), row.get("context"), row.get("batch"),
                 row.get("framework"), row.get("process_run")) for row in raw}
        if len(raw) != expected_count or len(keys) != expected_count or any(
                row.get("status") != "pass" or
                row.get("prefill_logits_mode") != mode for row in raw):
            errors.append(f"serving prefill {name} raw protocol changed")
        summary = json.loads((data / name / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "pass" or summary.get("prefill_logits_mode") != mode:
            errors.append(f"serving prefill {name} summary changed")
        counts[name] = len(raw)
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = {row.get("model"): row for row in comparison.get("rows", [])}
    qwen = rows.get("qwen2.5-0.5b", {})
    deepseek = rows.get("deepseek-r1-distill-qwen-1.5b", {})
    if comparison.get("status") != "pass" or len(rows) != 2 or \
            float(qwen.get("micro_speedup", 0.0)) < 2.9 or \
            float(deepseek.get("micro_speedup", 0.0)) < 1.3 or \
            float(qwen.get("peak_reduction", 0.0)) < 0.73 or \
            float(deepseek.get("peak_reduction", 0.0)) < 0.64 or \
            float(qwen.get("d2h_reduction", 0.0)) != 2048.0 or \
            float(deepseek.get("d2h_reduction", 0.0)) != 2048.0:
        errors.append("serving last-logit formal comparison changed")
    precision = json.loads((data / "precision" / "summary.json").read_text(
        encoding="utf-8"))
    precision_rows = precision.get("rows", [])
    if precision.get("status") != "pass" or len(precision_rows) != 2 or any(
            row.get("top_equal") is not True or
            float(row.get("max_abs", 1.0)) > 1.0e-4 or
            float(row.get("rmse", 1.0)) > 1.0e-5 for row in precision_rows):
        errors.append("serving last-logit precision gate changed")
    profile = json.loads((data / "profile" / "summary.json").read_text(
        encoding="utf-8"))
    profile_rows = profile.get("rows", [])
    if profile.get("status") != "pass" or len(profile_rows) != 2 or any(
            float(row.get("after_output_head_ms", 1.0)) /
            float(row.get("before_output_head_ms", 1.0)) > 0.01 or
            abs(float(row.get("after_softmax_ms", 0.0)) -
                float(row.get("before_softmax_ms", 0.0))) > 2.0
            for row in profile_rows):
        errors.append("serving last-logit profile contract changed")
    model_header = (REPOSITORY / "include/microllm/model/model.h").read_text(
        encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/hf_inference_shape_matrix.py").read_text(
        encoding="utf-8")
    if "forward_inference_last_logits" not in model_header or \
            "prefill-logits-mode" not in runner or "logits_to_keep" not in runner:
        errors.append("serving last-logit public/benchmark API is missing")
    return counts["full-logits"], counts["last-logit"], \
        counts["shape-survey-last"], len(precision_rows)


def validate_folded_gqa_discard(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "078-data"
    raw = [json.loads(line) for line in
           (data / "formal" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {(row.get("model"), row.get("framework"), row.get("process_run"))
            for row in raw}
    if len(raw) != 12 or len(keys) != 12 or any(
            row.get("status") != "pass" or row.get("context") != 2048 or
            row.get("batch") != 8 or row.get("prefill_logits_mode") != "last"
            for row in raw):
        errors.append("folded GQA formal protocol changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    rows = {row.get("model"): row for row in comparison.get("rows", [])}
    if comparison.get("status") != "discard" or len(rows) != 2 or \
            float(rows.get("qwen2.5-0.5b", {}).get("speedup", 0.0)) < 1.04 or \
            float(rows.get("deepseek-r1-distill-qwen-1.5b", {}).get(
                "speedup", 0.0)) < 1.07:
        errors.append("folded GQA performance evidence changed")
    precision = json.loads((data / "precision" / "summary.json").read_text(
        encoding="utf-8"))
    precision_rows = precision.get("rows", [])
    if len(precision_rows) != 2 or any(
            row.get("top_equal") is not True or
            float(row.get("max_abs", 0.0)) < 0.05 or
            float(row.get("rmse", 0.0)) < 0.01 for row in precision_rows):
        errors.append("folded GQA retained precision failure changed")
    profile = json.loads((data / "profile" / "summary.json").read_text(
        encoding="utf-8"))
    profile_rows = profile.get("rows", [])
    if len(profile_rows) != 2 or any(
            int(row.get("removed_repeat_calls", 0)) not in {192, 224} or
            float(row.get("folded_repeat_ms", 1.0)) != 0.0 or
            float(row.get("folded_kernel_ms", 1.0)) >=
            float(row.get("reference_kernel_ms", 0.0)) for row in profile_rows):
        errors.append("folded GQA mechanism profile changed")
    source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    if "grouped_rows" in source or "instead of being physically repeated" in source:
        errors.append("discarded folded GQA candidate remains in source")
    return len(raw), len(precision_rows), len(profile_rows)


def validate_register_softmax(errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "079-data"
    precision = json.loads((data / "precision" / "summary.json").read_text(
        encoding="utf-8"))
    precision_rows = precision.get("rows", [])
    if len(precision_rows) != 2 or any(
            row.get("top_equal") is not True or
            float(row.get("max_abs", 1.0)) != 0.0 or
            float(row.get("rmse", 1.0)) != 0.0 for row in precision_rows):
        errors.append("register softmax bit-exact gate changed")
    paired_raw = [json.loads(line) for line in
                  (data / "paired" / "raw.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    paired = json.loads((data / "paired" / "summary.json").read_text(
        encoding="utf-8"))
    paired_rows = {row.get("model"): row for row in paired.get("rows", [])}
    if len(paired_raw) != 12 or paired.get("status") != "pass" or \
            paired.get("pairs") != 3 or len(paired_rows) != 2 or \
            float(paired_rows.get("qwen2.5-0.5b", {}).get(
                "median_pair_ratio", 0.0)) < 1.04 or \
            float(paired_rows.get("deepseek-r1-distill-qwen-1.5b", {}).get(
                "median_pair_ratio", 0.0)) < 1.02:
        errors.append("register softmax paired performance changed")
    survey_raw = [json.loads(line) for line in
                  (data / "shape-survey-paired" / "raw.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    survey = json.loads((data / "shape-survey-paired" / "summary.json").read_text(
        encoding="utf-8"))
    survey_rows = survey.get("rows", [])
    if len(survey_raw) != 32 or len(survey_rows) != 16 or any(
            row.get("top_token_equal") is not True for row in survey_rows) or \
            sum(float(row.get("ratio", 1.0)) < 0.90 for row in survey_rows) != 1:
        errors.append("register softmax shape survey changed")
    recheck_raw = [json.loads(line) for line in
                   (data / "deepseek-t512-b1-recheck" / "raw.jsonl").read_text(
                       encoding="utf-8").splitlines()]
    recheck = json.loads((data / "deepseek-t512-b1-recheck" / "summary.json").read_text(
        encoding="utf-8"))
    if len(recheck_raw) != 6 or float(recheck.get("median_pair_ratio", 0.0)) < 0.99:
        errors.append("register softmax targeted recheck changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    profile = comparison.get("profile", {})
    if comparison.get("status") != "keep" or \
            float(comparison.get("shape_survey", {}).get(
                "accepted_min_ratio", 0.0)) < 0.98 or \
            float(profile.get("softmax_speedup", 0.0)) < 1.17 or \
            profile.get("private_segment_bytes") != 0 or \
            profile.get("sgpr_spills") != 0 or profile.get("vgpr_spills") != 0:
        errors.append("register softmax keep/profile contract changed")
    invalidation = json.loads((data / "unpaired" / "invalidation.json").read_text(
        encoding="utf-8"))
    if invalidation.get("status") != "invalid":
        errors.append("register softmax cross-window invalidation is missing")
    source = (REPOSITORY / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    if "causal_softmax_rows_register_kernel" not in source or \
            "kValuesPerThread = 8" not in source:
        errors.append("register-cached causal softmax source is missing")
    if "RegisterBoundaryT2048MatchesCpuAndZerosMask" not in tests:
        errors.append("register softmax T2048 executable boundary gate is missing")
    return len(paired_raw), len(survey_raw), len(recheck_raw), len(precision_rows)


def validate_readable_fused_attention_discard(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "080-data"
    raw = [json.loads(line) for line in
           (data / "paired" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "paired" / "summary.json").read_text(
        encoding="utf-8"))
    if len(raw) != 4 or summary.get("status") != "discard" or \
            summary.get("pairs") != 2 or summary.get("top_tokens_equal") is not True or \
            float(summary.get("median_ratio", 1.0)) > 0.37 or any(
                float(ratio) > 0.37 for ratio in summary.get("pair_ratios", [])):
        errors.append("readable fused Attention discard evidence changed")
    inventory = json.loads((data / "backend-inventory.json").read_text(encoding="utf-8"))
    if inventory.get("rocwmma_headers") is not True or \
            inventory.get("composable_kernel_headers") is not False or \
            inventory.get("fmha_runtime_library") is not False:
        errors.append("Attention backend inventory changed")
    source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    if "use_long_library_attention" in source or \
            "sequence >= 256 && hipblaslt_available()" not in source:
        errors.append("discarded readable fused Attention route remains in source")
    return len(raw), len(summary.get("pair_ratios", []))


def validate_inplace_causal_softmax(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "081-data"
    precision = json.loads((data / "precision" / "summary.json").read_text(
        encoding="utf-8"))
    precision_rows = precision.get("rows", [])
    if len(precision_rows) != 2 or any(
            row.get("top_equal") is not True or
            float(row.get("max_abs", 1.0)) != 0.0 or
            float(row.get("rmse", 1.0)) != 0.0 for row in precision_rows):
        errors.append("in-place causal softmax bit-exact gate changed")
    paired_raw = [json.loads(line) for line in
                  (data / "paired" / "raw.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    paired = json.loads((data / "paired" / "summary.json").read_text(
        encoding="utf-8"))
    paired_rows = {row.get("model"): row for row in paired.get("rows", [])}
    if len(paired_raw) != 12 or paired.get("pairs") != 3 or len(paired_rows) != 2 or any(
            float(row.get("median_pair_ratio", 0.0)) < 1.0
            for row in paired_rows.values()) or \
            float(paired_rows.get("qwen2.5-0.5b", {}).get(
                "peak_reduction", 0.0)) < 0.33 or \
            float(paired_rows.get("deepseek-r1-distill-qwen-1.5b", {}).get(
                "peak_reduction", 0.0)) < 0.18:
        errors.append("in-place causal softmax paired memory track changed")
    survey_raw = [json.loads(line) for line in
                  (data / "shape-survey" / "raw.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    survey = json.loads((data / "shape-survey" / "summary.json").read_text(
        encoding="utf-8"))
    survey_rows = survey.get("rows", [])
    if len(survey_raw) != 32 or len(survey_rows) != 16 or any(
            row.get("top_token_equal") is not True or
            float(row.get("tps_ratio", 0.0)) < 0.99 for row in survey_rows):
        errors.append("in-place causal softmax shape survey changed")
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    track_rows = comparison.get("paired_t2048_b8", [])
    profile = comparison.get("profile", {})
    if comparison.get("status") != "keep" or len(track_rows) != 2 or any(
            row.get("removed_matches_one_score_tensor") is not True
            for row in track_rows) or \
            profile.get("qwen", {}).get("removed_allocation_calls") != 72 or \
            profile.get("deepseek", {}).get("removed_allocation_calls") != 84 or \
            profile.get("qwen", {}).get("extra_copy_calls") != 0 or \
            profile.get("deepseek", {}).get("extra_copy_calls") != 0:
        errors.append("in-place causal softmax mechanism evidence changed")
    source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    if "input/output aliasing" not in source or source.count(
            "static_cast<float*>(probabilities.data())") < 1:
        errors.append("in-place causal softmax source alias is missing")
    return len(paired_raw), len(survey_raw), len(precision_rows)


def validate_stop_token_early_completion(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "082-data"
    cpu = json.loads((data / "cpu-tests.json").read_text(encoding="utf-8"))
    hip = json.loads((data / "hip-tests.json").read_text(encoding="utf-8"))
    if cpu.get("tests") != 4 or cpu.get("failures") != 0 or cpu.get("errors") != 0:
        errors.append("stop-token CPU executable evidence changed")
    if hip.get("tests") != 2 or hip.get("failures") != 0 or hip.get("errors") != 0:
        errors.append("stop-token HIP executable evidence changed")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    contracts = summary.get("contracts", {})
    if summary.get("status") != "pass" or any(
            contracts.get(name) is not True for name in (
                "single_request_stops_after_appending_token",
                "batch_rows_match_independent_generation",
                "batch_rows_may_have_different_lengths",
                "reference_scheduler_releases_cache_on_stop",
                "completion_reason_is_explicit",
                "stop_token_order_canonicalized_for_batching")) or \
            contracts.get("static_batch_slot_reclaimed_before_group_end") is not False:
        errors.append("stop-token lifecycle contract changed")
    generator = (REPOSITORY / "include/microllm/inference/generator.h").read_text(
        encoding="utf-8")
    scheduler = (REPOSITORY / "include/microllm/inference/scheduler.h").read_text(
        encoding="utf-8")
    tests = (REPOSITORY / "tests/inference/generator_test.cpp").read_text(
        encoding="utf-8")
    if "stop_tokens" not in generator or "CompletionReason" not in scheduler or \
            "StaticBatchStopRowsMatchIndependentVariableLengths" not in tests:
        errors.append("stop-token public API or variable-row gate is missing")
    return int(cpu.get("tests", 0)), int(hip.get("tests", 0))


def validate_kv_cache_clear_row(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "083-data"
    cpu = json.loads((data / "cpu-tests.json").read_text(encoding="utf-8"))
    hip = json.loads((data / "hip-tests.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    contracts = summary.get("contracts", {})
    if cpu.get("tests") != 1 or cpu.get("failures") != 0 or \
            hip.get("tests") != 1 or hip.get("failures") != 0 or \
            summary.get("status") != "pass" or \
            summary.get("test_shape", {}).get("cleared_bytes") != 192 or any(
                contracts.get(name) is not True for name in (
                    "full_capacity_row_zeroed", "other_row_prefix_unchanged",
                    "shared_position_unchanged_by_clear",
                    "next_shared_decode_writes_new_position",
                    "old_cleared_prefix_remains_zero",
                    "hip_clear_has_zero_payload_transfers")) or \
            contracts.get("per_slot_position_supported") is not False:
        errors.append("KV Cache clear-row evidence changed")
    header = (REPOSITORY / "include/microllm/inference/kv_cache.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/inference/kv_cache.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    if "clear_row" not in header or "full_row" not in source or \
            "ClearCacheRowIsDeviceNativeAndMatchesCpuStorage" not in hip_tests:
        errors.append("KV Cache clear-row API or executable gate is missing")
    return int(cpu.get("tests", 0)), int(hip.get("tests", 0)), \
        int(summary.get("test_shape", {}).get("cleared_bytes", 0))


def validate_kv_cache_per_row_positions(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "084-data"
    cpu = json.loads((data / "cpu-tests.json").read_text(encoding="utf-8"))
    hip = json.loads((data / "hip-tests.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    transitions = summary.get("state_transitions", [])
    contracts = summary.get("contracts", {})
    expected = [[0, 0, 0], [2, 0, 0], [0, 0, 0],
                [3, 3, 3], [3, 0, 3], [3, 3, 3]]
    if cpu.get("tests") != 2 or cpu.get("failures") != 0 or \
            hip.get("tests") != 1 or hip.get("failures") != 0 or \
            summary.get("status") != "pass" or \
            [row.get("positions") for row in transitions] != expected or any(
                contracts.get(name) is not True for name in (
                    "uniform_position_backward_compatible",
                    "ambiguous_uniform_position_throws",
                    "row_advance_capacity_checked",
                    "reset_row_clears_storage_and_position")) or \
            contracts.get("model_consumes_divergent_positions") is not False:
        errors.append("KV Cache per-row position evidence changed")
    header = (REPOSITORY / "include/microllm/inference/kv_cache.h").read_text(
        encoding="utf-8")
    tests = (REPOSITORY / "tests/model/model_test.cpp").read_text(encoding="utf-8")
    for token in ("row_position", "row_positions", "positions_uniform", "advance_row"):
        if token not in header:
            errors.append(f"KV Cache per-row API is missing {token}")
    if "PerRowPositionsRejectAmbiguousUniformReads" not in tests:
        errors.append("KV Cache divergent-position executable gate is missing")
    return int(cpu.get("tests", 0)), int(hip.get("tests", 0)), len(transitions)


def validate_inference_shape_memory_matrix(errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "085-data"

    def records(name: str) -> list[dict]:
        return [json.loads(line) for line in
                (data / name).read_text(encoding="utf-8").splitlines()]

    qwen = records("qwen-raw.jsonl")
    deepseek = records("deepseek-raw.jsonl")
    formal = qwen + deepseek
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    qwen_release = records("qwen-release-raw.jsonl")
    deepseek_release = records("deepseek-release-raw.jsonl")
    release = qwen_release + deepseek_release
    release_summary = json.loads(
        (data / "release-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    environment = (data / "environment.txt").read_text(encoding="utf-8")
    invalid = records("invalid-free-first-token-pilot.jsonl")
    mixed = records("mixed-qwen-runner-invalid.jsonl") + \
        records("mixed-deepseek-runner-invalid.jsonl")

    if len(qwen) != 36 or len(deepseek) != 36 or \
            summary.get("status") != "pass" or len(summary.get("rows", [])) != 36:
        errors.append("frozen inference shape matrix row counts changed")
        return len(formal), len(summary.get("rows", [])), len(invalid), len(release)

    expected_shapes = {(context, batch, decode)
                       for context in (8, 512, 2048)
                       for batch in (1, 8) for decode in (1, 8, 32)}
    by_pair: dict[tuple, dict[str, dict]] = {}
    for record in formal:
        shape = (int(record.get("context", -1)), int(record.get("batch", -1)),
                 int(record.get("decode_tokens", -1)))
        if record.get("status") != "pass" or shape not in expected_shapes or \
                record.get("decode_step_semantics") != \
                "one_model_forward_per_measured_token" or \
                int(record.get("measured_tokens", -1)) != \
                int(record.get("measured_forward_steps", -2)):
            errors.append("formal steady-decode execution contract changed")
            continue
        context, _, decode = shape
        expected_active = context + decode
        active = int(record.get("kv_cache_active_tokens", -1))
        capacity = int(record.get("kv_cache_capacity_tokens", -1))
        actual = int(record.get("kv_cache_actual_bytes", 0))
        active_bytes = int(record.get("kv_cache_active_bytes", 0))
        utilization = float(record.get("kv_cache_utilization", -1.0))
        if record.get("framework") == "microllm":
            expected_capacity = context + 32
        else:
            expected_capacity = expected_active
        if active != expected_active or capacity != expected_capacity or actual <= 0 or \
                active_bytes <= 0 or not math.isclose(
                    utilization, active_bytes / actual, rel_tol=1.0e-6,
                    abs_tol=1.0e-9):
            errors.append("formal KV active/capacity/utilization contract changed")
        key = (record.get("model"),) + shape
        by_pair.setdefault(key, {})[str(record.get("framework"))] = record

    matched = 0
    for pair in by_pair.values():
        if set(pair) == {"microllm", "pytorch"} and \
                pair["microllm"].get("generated_tokens") == \
                pair["pytorch"].get("generated_tokens"):
            matched += 1
    if len(by_pair) != 36 or matched != 36:
        errors.append("formal inference token pairing changed")

    long_rows = [row for row in summary["rows"]
                 if row.get("context") == 2048 and row.get("batch") == 8]
    if len(long_rows) != 6:
        errors.append("semantic long-context survey rows changed")
    memory_rows = [row for row in long_rows if row.get("decode_tokens") == 32]
    if len(memory_rows) != 2 or any(
            float(row.get("microllm_peak_bytes", math.inf)) >=
            float(row.get("pytorch_peak_bytes", 0.0)) for row in memory_rows):
        errors.append("recorded long-context peak-memory advantage disappeared")

    if len(qwen_release) != 12 or len(deepseek_release) != 12 or \
            release_summary.get("status") != "pass" or \
            release_summary.get("build_type") != "Release" or \
            len(release_summary.get("rows", [])) != 12:
        errors.append("Release steady-decode matrix row counts changed")
    release_pairs: dict[tuple, dict[str, dict]] = {}
    for record in release:
        context = int(record.get("context", -1))
        batch = int(record.get("batch", -1))
        if record.get("status") != "pass" or \
                int(record.get("decode_tokens", -1)) != 8 or \
                record.get("decode_step_semantics") != \
                "one_model_forward_per_measured_token" or \
                int(record.get("measured_tokens", -1)) != \
                int(record.get("measured_forward_steps", -2)) or \
                int(record.get("kv_cache_active_tokens", -1)) != context + 8 or \
                int(record.get("kv_cache_capacity_tokens", -1)) != context + 8:
            errors.append("Release steady-decode execution contract changed")
        key = (record.get("model"), context, batch)
        release_pairs.setdefault(key, {})[str(record.get("framework"))] = record
    release_matched = sum(
        set(pair) == {"microllm", "pytorch"} and
        pair["microllm"].get("generated_tokens") ==
        pair["pytorch"].get("generated_tokens")
        for pair in release_pairs.values())
    if len(release_pairs) != 12 or release_matched != 10:
        errors.append("Release token-match boundary changed")
    for row in release_summary.get("rows", []):
        ratio = float(row.get("throughput_ratio_microllm_over_pytorch", 0.0))
        model_name = str(row.get("model"))
        context = int(row.get("context", -1))
        expected_pass = model_name == "qwen2.5-0.5b" or context < 2048
        if (ratio >= 1.0) != expected_pass:
            errors.append("Release throughput parity boundary changed")
    release_long_b8 = [row for row in release_summary.get("rows", [])
                       if row.get("context") == 2048 and row.get("batch") == 8]
    if len(release_long_b8) != 2 or any(
            float(row.get("microllm_peak_bytes", math.inf)) >=
            float(row.get("pytorch_peak_bytes", 0.0)) for row in release_long_b8):
        errors.append("Release long-batch peak-memory boundary changed")

    invalid_one = [record for record in invalid
                   if record.get("decode_tokens") == 1]
    missing_forward = sum("measured_forward_steps" not in record for record in mixed)
    if len(invalid_one) != 4 or any(
            record.get("decode_step_semantics") ==
            "one_model_forward_per_measured_token" for record in invalid_one) or \
            not 0 < missing_forward < len(mixed):
        errors.append("invalid free-token or mixed-source counterexample changed")

    if gates.get("status") != "pass" or \
            gates.get("cpu", {}).get("passed") != 207 or \
            gates.get("hip", {}).get("passed") != 88 or \
            gates.get("sanitizer", {}).get("passed") != 200 or \
            gates.get("python_matrix_contract", {}).get("passed") != 15 or \
            gates.get("release_matrix", {}).get("records") != 24 or \
            gates.get("release_matrix", {}).get("token_pairs_matched") != 10:
        errors.append("inference matrix test gates changed")
    if "semantic_build_type=unspecified" not in environment or \
            "release_build_type=Release" not in environment:
        errors.append("inference matrix build-type audit changed")

    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_inference_shape_matrix.py").read_text(encoding="utf-8")
    cpu_test = (REPOSITORY / "tests" / "inference" /
                "shape_matrix_test.cpp").read_text(encoding="utf-8")
    for token in ("--decode-mode", "--cache-capacity", "measured_forward_steps"):
        if token not in app:
            errors.append(f"steady-decode CLI evidence is missing {token}")
    for token in ("boundary", "decode_lengths", "kv_cache_share_of_incremental_peak"):
        if token not in runner:
            errors.append(f"inference matrix runner evidence is missing {token}")
    if "active_cache_bytes" not in cpu_test or "storage_before" not in cpu_test:
        errors.append("tiny active/allocation KV evidence is missing")
    return len(formal), matched, len(invalid), len(release)


def validate_deepseek_steady_profile_d2h_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "086-data"
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    pair = json.loads((data / "d2h-pair-summary.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in
           (data / "d2h-pair-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    if profile.get("status") != "pass" or profile.get("build_type") != "Release" or \
            profile.get("batch_1", {}).get("cached_attention_calls") != 448 or \
            profile.get("batch_8", {}).get("cached_attention_calls") != 448 or \
            profile.get("inference", {}).get(
                "cached_attention_is_primary_decode_hotspot") is not True or \
            profile.get("inference", {}).get(
                "batch_8_allocator_thrash_is_secondary_hotspot") is not True or \
            profile.get("inference", {}).get("argmax_kernel_is_primary_hotspot") is not False:
        errors.append("DeepSeek steady profile evidence changed")
    rows = pair.get("rows", [])
    by_batch = {int(row.get("batch", -1)): row for row in rows}
    if pair.get("status") != "pass" or len(raw) != 12 or set(by_batch) != {1, 8} or \
            not 0.98 < float(by_batch[1].get("candidate_speedup", 0.0)) < 1.02 or \
            float(by_batch[8].get("candidate_speedup", 1.0)) >= 0.9 or \
            any(row.get("tokens_equal") is not True for row in rows) or \
            by_batch[8].get("baseline_measured_d2h_calls") != 24.0 or \
            by_batch[8].get("candidate_measured_d2h_calls") != 3.0 or \
            by_batch[8].get("candidate_engine_backend_allocation_calls", 0) < 10000 or \
            by_batch[8].get("baseline_engine_backend_allocation_calls", 10000) >= 2000:
        errors.append("D2H discard paired-process evidence changed")
    for name in ("b1-kernel-stats.csv", "b8-kernel-stats.csv"):
        if "cached_attention_fused_kernel" not in \
                (data / name).read_text(encoding="utf-8"):
            errors.append(f"cached Attention profile row is missing from {name}")
    header = (REPOSITORY / "include" / "microllm" / "ops" / "ops.h").read_text(
        encoding="utf-8")
    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    retry = ROOT / "experiments" / "090-data" / "summary.json"
    retry_kept = retry.is_file() and json.loads(
        retry.read_text(encoding="utf-8")).get("decision") == "keep"
    if ("argmax_out_" in header or "argmax_last_dim_out_" in header or
            "Tensor history" in app) and not retry_kept:
        errors.append("rejected D2H candidate returned without a retained retry gate")
    return len(raw), int(profile["batch_1"]["cached_attention_calls"]), \
        int(profile["batch_8"]["cached_attention_calls"])


def validate_immediate_default_stream_pool(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "087-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    qwen_pair = json.loads(
        (data / "qwen-t512-b8-pair-summary.json").read_text(encoding="utf-8"))
    deepseek_pair = json.loads(
        (data / "deepseek-t512-b8-pair-summary.json").read_text(encoding="utf-8"))

    def records(name: str) -> list[dict]:
        return [json.loads(line) for line in
                (data / name).read_text(encoding="utf-8").splitlines()]

    qwen = records("qwen-matrix-raw.jsonl")
    deepseek = records("deepseek-matrix-raw.jsonl")
    if summary.get("status") != "pass" or summary.get("decision") != "keep" or \
            not 1.0 < float(summary["t2048_pairs"]["batch_1"]["speedup"]) < 1.05 or \
            not 1.0 < float(summary["t2048_pairs"]["batch_8"]["speedup"]) < 1.10 or \
            summary["t2048_pairs"]["batch_8"]["backend_allocations"] != [903, 94] or \
            summary["t2048_pairs"]["batch_8"]["backend_deallocations"] != [352, 0] or \
            summary.get("safety_contract", {}).get(
                "non_default_stream_permanently_disables_pool") is not True:
        errors.append("immediate allocator keep evidence changed")
    if qwen_pair.get("status") != "pass" or deepseek_pair.get("status") != "pass" or \
            not 1.0 < float(qwen_pair.get("candidate_speedup", 0.0)) < 1.05 or \
            float(deepseek_pair.get("candidate_speedup", 0.0)) < 1.05 or \
            qwen_pair.get("tokens_equal") is not True or \
            deepseek_pair.get("tokens_equal") is not True:
        errors.append("T512 B8 allocator recheck evidence changed")
    if len(qwen) != 12 or len(deepseek) != 12 or any(
            record.get("status") != "pass" for record in qwen + deepseek):
        errors.append("allocator official matrix rows changed")
    micro = [record for record in qwen + deepseek
             if record.get("framework") == "microllm"]
    if len(micro) != 12 or any(
            not 82 <= int(record.get("engine_backend_allocation_calls", -1)) <= 94 or
            int(record.get("engine_backend_deallocation_calls", -1)) != 0
            for record in micro):
        errors.append("allocator candidate reuse counters changed")
    if gates.get("status") != "pass" or gates.get("cpu", {}).get("passed") != 207 or \
            gates.get("hip", {}).get("passed") != 88 or \
            gates.get("sanitizer", {}).get("passed") != 200 or any(
                gates.get("allocator_safety", {}).get(name) is not True for name in (
                    "immediate_exact_size_reuse",
                    "no_sync_256_iteration_kernel_order",
                    "non_default_stream_disables_pool")):
        errors.append("immediate allocator safety gates changed")
    runtime_source = (REPOSITORY / "src" / "runtime" / "runtime.cpp").read_text(
        encoding="utf-8")
    runtime_tests = (REPOSITORY / "tests" / "runtime" / "runtime_test.cpp").read_text(
        encoding="utf-8")
    stress_tests = (REPOSITORY / "tests" / "ops" / "hip_ops_test.cpp").read_text(
        encoding="utf-8")
    if "kRetirementBatchSize" in runtime_source or \
            "std::vector<void*>" not in runtime_source or \
            "notify_non_default_stream permanently disables" not in runtime_source or \
            "DefaultStreamPoolReusesEveryExactSizeWithoutBatchPhase" not in runtime_tests or \
            "if (iteration % 16" in stress_tests:
        errors.append("immediate allocator source or safety gate changed")
    return len(qwen) + len(deepseek), len(micro), \
        int(summary["t2048_pairs"]["batch_8"]["backend_allocations"][1])


def validate_bf16x2_key_load_discard(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "088-data"
    precision = json.loads(
        (data / "precision-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    rows = {int(row.get("batch", -1)): row for row in precision.get("rows", [])}
    if precision.get("status") != "pass" or set(rows) != {1, 8} or \
            int(rows[1].get("values", 0)) != 151936 or \
            float(rows[1].get("max_abs_error", 0.0)) < 0.05 or \
            rows[1].get("tokens_equal") is not True or \
            int(rows[8].get("values", 0)) != 1215488 or \
            float(rows[8].get("max_abs_error", 0.0)) < 10.0 or \
            rows[8].get("tokens_equal") is not False:
        errors.append("BF16x2 official precision failure evidence changed")
    if gates.get("status") != "discard" or \
            gates.get("focused_hip_tests", {}).get("passed") != 4 or \
            gates.get("official_precision", {}).get("failed") != 2 or \
            gates.get("performance_runs") != 0 or \
            gates.get("candidate_reverted") is not True:
        errors.append("BF16x2 discard gates changed")
    source = (REPOSITORY / "src" / "ops" / "hip" /
              "basic_kernels.hip").read_text(encoding="utf-8")
    if "cached_attention_fused_bf16x2_key_kernel" in source:
        errors.append("rejected BF16x2 Key Kernel remains in retained source")
    return len(rows), int(rows[1]["values"]), int(rows[8]["values"])


def validate_raw_packed_key_load_discard(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "089-data"
    previous = json.loads((ROOT / "experiments" / "088-data" /
                           "precision-summary.json").read_text(encoding="utf-8"))
    precision = json.loads(
        (data / "precision-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    rows = {int(row.get("batch", -1)): row for row in precision.get("rows", [])}
    previous_rows = {int(row.get("batch", -1)): row
                     for row in previous.get("rows", [])}
    if precision.get("status") != "pass" or set(rows) != {1, 8} or \
            rows != previous_rows or float(rows[8].get("max_abs_error", 0.0)) < 10.0:
        errors.append("raw-packed identical precision failure evidence changed")
    if gates.get("status") != "discard" or \
            gates.get("focused_hip_tests", {}).get("passed") != 4 or \
            gates.get("official_precision", {}).get("failed") != 2 or \
            gates.get("matches_experiment_088_errors") is not True or \
            gates.get("performance_runs") != 0 or \
            gates.get("candidate_reverted") is not True:
        errors.append("raw-packed discard gates changed")
    source = (REPOSITORY / "src" / "ops" / "hip" /
              "basic_kernels.hip").read_text(encoding="utf-8")
    if "cached_attention_fused_bf16_packed_key_kernel" in source:
        errors.append("rejected raw-packed Key Kernel remains in retained source")
    return len(rows), int(rows[1]["values"]), int(rows[8]["values"])


def validate_device_token_history(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "090-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    qwen_pair = json.loads(
        (data / "qwen-t512-b8-pair-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))

    def records(name: str) -> list[dict]:
        return [json.loads(line) for line in
                (data / name).read_text(encoding="utf-8").splitlines()]

    qwen = records("qwen-matrix-raw.jsonl")
    deepseek = records("deepseek-matrix-raw.jsonl")
    if summary.get("status") != "pass" or summary.get("decision") != "keep" or \
            not 0.98 < float(summary["t2048_pairs"]["batch_1"]["speedup"]) < 1.02 or \
            not 0.98 < float(summary["t2048_pairs"]["batch_8"]["speedup"]) < 1.02 or \
            summary["t2048_pairs"]["batch_8"]["d2h_calls"] != [24, 3] or \
            summary.get("scope", {}).get("sampling_and_stop_paths_unchanged") is not True:
        errors.append("device token-history keep evidence changed")
    if qwen_pair.get("status") != "pass" or \
            not 0.98 < float(qwen_pair.get("candidate_speedup", 0.0)) < 1.02 or \
            qwen_pair.get("baseline_d2h_calls") != 24 or \
            qwen_pair.get("candidate_d2h_calls") != 3 or \
            qwen_pair.get("tokens_equal") is not True:
        errors.append("token-history T512 B8 recheck changed")
    if len(qwen) != 12 or len(deepseek) != 12 or any(
            record.get("status") != "pass" for record in qwen + deepseek):
        errors.append("token-history official matrix rows changed")
    micro = [record for record in qwen + deepseek
             if record.get("framework") == "microllm"]
    if len(micro) != 12 or any(
            int(record.get("measured_d2h_calls", -1)) != 3 or
            not 81 <= int(record.get("engine_backend_allocation_calls", -1)) <= 94 or
            int(record.get("engine_backend_deallocation_calls", -1)) != 0
            for record in micro):
        errors.append("token-history transfer or allocator counters changed")
    if gates.get("status") != "pass" or gates.get("cpu", {}).get("passed") != 208 or \
            gates.get("hip", {}).get("passed") != 89 or \
            gates.get("sanitizer", {}).get("passed") != 201:
        errors.append("token-history final test gates changed")
    header = (REPOSITORY / "include" / "microllm" / "ops" / "ops.h").read_text(
        encoding="utf-8")
    generator = (REPOSITORY / "src" / "inference" / "generator.cpp").read_text(
        encoding="utf-8")
    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    for token in ("argmax_out_", "argmax_last_dim_out_"):
        if token not in header:
            errors.append(f"token-history public operator is missing {token}")
    if "Tensor history" not in generator or "Tensor history" not in app:
        errors.append("token-history generation or benchmark path is missing")
    return len(qwen) + len(deepseek), len(micro), \
        int(summary["t2048_pairs"]["batch_8"]["d2h_calls"][1])


def validate_normalize_cached_probabilities_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "091-data"
    precision = json.loads(
        (data / "precision-summary.json").read_text(encoding="utf-8"))
    pair = json.loads((data / "pair-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    precision_rows = {int(row.get("batch", -1)): row
                      for row in precision.get("rows", [])}
    performance_rows = {int(row.get("batch", -1)): row
                        for row in pair.get("rows", [])}
    if precision.get("status") != "pass" or set(precision_rows) != {1, 8} or any(
            row.get("bit_exact") is not True or
            float(row.get("max_abs_error", -1.0)) != 0.0 or
            float(row.get("rmse", -1.0)) != 0.0 or
            row.get("tokens_equal") is not True
            for row in precision_rows.values()):
        errors.append("cached probability normalization exact gate changed")
    if pair.get("status") != "pass" or set(performance_rows) != {1, 8} or any(
            not 0.98 < float(row.get("candidate_speedup", 0.0)) < 1.0 or
            row.get("tokens_equal") is not True
            for row in performance_rows.values()):
        errors.append("cached probability normalization performance rejection changed")
    if gates.get("status") != "discard" or \
            gates.get("official_precision", {}).get("bit_exact") is not True or \
            gates.get("paired_performance", {}).get("failed") != 2 or \
            gates.get("candidate_reverted") is not True:
        errors.append("cached probability normalization discard gates changed")
    source = (REPOSITORY / "src" / "ops" / "hip" /
              "basic_kernels.hip").read_text(encoding="utf-8")
    if "shared_scores[position] /= denominator" in source:
        errors.append("rejected cached probability normalization remains in source")
    return len(precision_rows), int(precision_rows[1]["values"]), \
        int(precision_rows[8]["values"])


def validate_bf16_paired_value_load_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "092-data"
    precision = json.loads(
        (data / "precision-summary.json").read_text(encoding="utf-8"))
    pair = json.loads((data / "pair-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    precision_rows = {int(row.get("batch", -1)): row
                      for row in precision.get("rows", [])}
    performance_rows = {int(row.get("batch", -1)): row
                        for row in pair.get("rows", [])}
    if precision.get("status") != "pass" or set(precision_rows) != {1, 8} or any(
            row.get("bit_exact") is not True or
            float(row.get("max_abs_error", -1.0)) != 0.0 or
            row.get("tokens_equal") is not True
            for row in precision_rows.values()):
        errors.append("paired Value complete-logit gate changed")
    if pair.get("status") != "pass" or set(performance_rows) != {1, 8} or any(
            not 0.97 < float(row.get("candidate_speedup", 0.0)) < 1.0 or
            row.get("tokens_equal") is not True
            for row in performance_rows.values()):
        errors.append("paired Value performance rejection changed")
    if gates.get("status") != "discard" or \
            gates.get("official_precision", {}).get("bit_exact") is not True or \
            gates.get("paired_performance", {}).get("failed") != 2 or \
            gates.get("candidate_reverted") is not True:
        errors.append("paired Value discard gates changed")
    source = (REPOSITORY / "src" / "ops" / "hip" /
              "basic_kernels.hip").read_text(encoding="utf-8")
    if "second_total" in source:
        errors.append("rejected paired Value accumulation remains in source")
    return len(precision_rows), int(precision_rows[1]["values"]), \
        int(precision_rows[8]["values"])


def validate_divergent_row_cache_reference(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "093-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    transitions = summary.get("state_transitions", [])
    contracts = summary.get("contracts", {})
    expected = [[3, 3], [0, 3], [1, 4], [2, 5], [2, 0]]
    if summary.get("status") != "pass" or \
            [row.get("positions") for row in transitions] != expected or any(
                contracts.get(name) is not True for name in (
                    "fp32_rows_match_independent_b1",
                    "bf16_rows_match_independent_b1",
                    "second_step_rows_match_independent_b1",
                    "uniform_positions_use_existing_batch_path",
                    "shared_storage_address_stable",
                    "reset_maximum_row_shrinks_logical_prefix",
                    "missing_storage_for_nonzero_position_rejected",
                    "hip_matches_cpu",
                    "serial_b1_view_oracle")) or \
            contracts.get("hip_payload_d2h_during_execution") != 0 or \
            contracts.get("parallel_positions_aware_kernel") is not False:
        errors.append("divergent cached-row reference evidence changed")
    if gates.get("status") != "pass" or gates.get("cpu", {}).get("passed") != 210 or \
            gates.get("hip", {}).get("passed") != 90 or \
            gates.get("sanitizer", {}).get("passed") != 203 or \
            gates.get("focused", {}).get("cache_dtypes") != 2 or \
            gates.get("focused", {}).get("decode_steps") != 2:
        errors.append("divergent cached-row final gates changed")
    header = (REPOSITORY / "include" / "microllm" / "model" / "model.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src" / "model" / "model.cpp").read_text(
        encoding="utf-8")
    kv_source = (REPOSITORY / "src" / "inference" / "kv_cache.cpp").read_text(
        encoding="utf-8")
    cpu_tests = (REPOSITORY / "tests" / "model" / "model_test.cpp").read_text(
        encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "inference" /
                 "hip_shape_matrix_test.cpp").read_text(encoding="utf-8")
    if "forward_cached_rows" not in header or "cache_row_view" not in source or \
            "resize_shared_views" not in source or "resize_tensor_prefix" not in kv_source or \
            "DivergentCachedRowsMatchIndependentB1References" not in cpu_tests or \
            "DivergentRowsMatchCpuWithoutPayloadD2H" not in hip_tests:
        errors.append("divergent cached-row source or executable gate is missing")
    return len(transitions), int(gates["focused"]["cache_dtypes"]), \
        int(gates["focused"]["decode_steps"])


def validate_slot_row_prefill(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "094-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    transitions = summary.get("state_transitions", [])
    contracts = summary.get("contracts", {})
    expected = [[3, 3], [0, 3], [2, 3], [3, 4]]
    if summary.get("status") != "pass" or \
            [row.get("positions") for row in transitions] != expected or any(
                contracts.get(name) is not True for name in (
                    "fp32_prefill_matches_independent_b1",
                    "bf16_prefill_matches_independent_b1",
                    "continued_decode_matches_independent_b1",
                    "other_row_key_value_preserved",
                    "shared_storage_address_stable",
                    "empty_nonzero_row_can_be_first_admission",
                    "nonempty_row_rejected",
                    "out_of_range_row_rejected",
                    "hip_matches_cpu")) or \
            contracts.get("hip_payload_d2h_during_execution") != 0 or \
            contracts.get("continuous_scheduler_complete") is not False or \
            contracts.get("performance_claim") is not False:
        errors.append("single-row prefill reference evidence changed")
    if gates.get("status") != "pass" or \
            gates.get("full", {}).get("passed") != 302 or \
            gates.get("cpu", {}).get("passed") != 211 or \
            gates.get("hip", {}).get("passed") != 91 or \
            gates.get("sanitizer", {}).get("passed") != 204 or \
            gates.get("focused", {}).get("cache_dtypes") != 2 or \
            gates.get("focused", {}).get("continued_decode_steps") != 1:
        errors.append("single-row prefill final gates changed")
    header = (REPOSITORY / "include" / "microllm" / "model" / "model.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src" / "model" / "model.cpp").read_text(
        encoding="utf-8")
    cpu_tests = (REPOSITORY / "tests" / "model" / "model_test.cpp").read_text(
        encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "inference" /
                 "hip_shape_matrix_test.cpp").read_text(encoding="utf-8")
    if "forward_prefill_cached_row" not in header or \
            "copy_cache_prefix_to_row" not in source or \
            "RowPrefillReplacesOnlyAnEmptySharedCacheSlot" not in cpu_tests or \
            "RowPrefillPreservesOtherSlotAndMatchesCpu" not in hip_tests:
        errors.append("single-row prefill source or executable gate is missing")
    return len(transitions), int(gates["focused"]["cache_dtypes"]), \
        int(gates["focused"]["continued_decode_steps"])


def validate_serving_inference_efficiency(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "095-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    pilot_raw = [json.loads(line) for line in
                 (data / "pilot-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    pilot = json.loads((data / "pilot-summary.json").read_text(encoding="utf-8"))
    long_raw = [json.loads(line) for line in
                (data / "long-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    long_summary = json.loads((data / "long-summary.json").read_text(encoding="utf-8"))
    rechecks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
        (data / "qwen-t128-b4-n64-recheck").glob("run*.stdout"))]
    suite = summary.get("serving_suite", {})
    pilot_contract = summary.get("incremental_pilot", {})
    failure = summary.get("observed_failure", {})
    long_contract = summary.get("long_context", {})
    if suite.get("contexts") != [1, 8, 32, 128, 512, 2048] or \
            suite.get("batches") != [1, 2, 4, 8] or \
            suite.get("decode_lengths") != [1, 8, 32, 64] or \
            suite.get("paired_cached_cases_per_model") != 96:
        errors.append("serving inference suite axes changed")
    pilot_rows = pilot.get("rows", [])
    paired_pass = [row for row in pilot_rows if row.get("status") == "pass"]
    token_equal = [row for row in paired_pass
                   if row.get("cross_framework_tokens_equal") is True]
    failures = [row for row in pilot_raw if row.get("status") != "pass"]
    if len(pilot_raw) != 24 or sum(row.get("status") == "pass" for row in pilot_raw) != 23 or \
            len(failures) != 1 or len(pilot_rows) != 12 or len(paired_pass) != 11 or \
            len(token_equal) != 8 or pilot_contract.get("raw_failed") != 1:
        errors.append("N64 incremental inference pilot counts changed")
    if not failures or failures[0].get("model") != "qwen2.5-0.5b" or \
            failures[0].get("context") != 128 or failures[0].get("batch") != 4 or \
            failures[0].get("framework") != "microllm":
        errors.append("preserved Qwen batch-row failure changed")
    if len(rechecks) != 3 or any(
            row.get("status") != "pass" or row.get("token_count") != 128 or
            row.get("batch") != 4 or row.get("decode_tokens_per_second", 0.0) < 1100.0 or
            row.get("kv_cache_utilization") != 1.0 or
            len(row.get("generated_tokens", [])) != 64 for row in rechecks) or \
            len({tuple(row["generated_tokens"]) for row in rechecks}) != 1 or \
            failure.get("classification") != "observed_once_not_stable" or \
            failure.get("stable_indexing_bug_supported") is not False:
        errors.append("Qwen non-stable failure recheck changed")
    long_rows = long_summary.get("rows", [])
    by_model = {row.get("model"): row for row in long_rows}
    expected_ratios = {
        "qwen2.5-0.5b": 1.2498952370244671,
        "deepseek-r1-distill-qwen-1.5b": 0.8678597980876258,
    }
    if len(long_raw) != 4 or any(row.get("status") != "pass" for row in long_raw) or \
            len(by_model) != 2 or any(
                row.get("cross_framework_tokens_equal") is not True or
                row.get("microllm_kv_cache_utilization") != 1.0 or
                row.get("microllm_measured_d2h_calls") != 3.0 or
                abs(float(row.get("throughput_ratio_microllm_over_pytorch", 0.0)) -
                    expected_ratios[model]) > 1.0e-9
                for model, row in by_model.items()) or \
            long_contract.get("logical_forward_steps") != 384:
        errors.append("T2048 B2 N64 paired evidence changed")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_inference_shape_matrix.py").read_text(encoding="utf-8")
    tests = (REPOSITORY / "python" / "tests" /
             "test_hf_inference_shape_matrix.py").read_text(encoding="utf-8")
    required = ("\"serving\"", "kv_cache_waste_bytes",
                "kv_cache_active_share_of_incremental_peak",
                "non_kv_incremental_bytes")
    if any(name not in runner for name in required) or \
            "decode_lengths\"], [1, 8, 32, 64]" not in tests:
        errors.append("serving inference runner or contract tests are missing")
    if gates.get("status") != "pass_with_observed_limit" or \
            gates.get("full", {}).get("passed") != 302 or \
            gates.get("sanitizer", {}).get("passed") != 204 or \
            gates.get("runner_contract_tests", {}).get("passed") != 15:
        errors.append("serving inference final gates changed")
    return len(pilot_raw), len(long_raw), len(rechecks)


def validate_continuous_slot_scheduler(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "096-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
        (data / "hip-benchmark").glob("*.json"))]
    release_records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
        (data / "hip-release").glob("*.json"))]
    divergent = [row for row in records
                 if int(row.get("continuous_divergent_batch_decode_calls", 0)) > 0]
    uniform = [row for row in records
               if int(row.get("continuous_uniform_batch_decode_calls", 0)) > 0]
    release_divergent = [row for row in release_records
                         if int(row.get("continuous_divergent_batch_decode_calls", 0)) > 0]
    release_uniform = [row for row in release_records
                       if int(row.get("continuous_uniform_batch_decode_calls", 0)) > 0]
    contracts = summary.get("contracts", {})
    transitions = summary.get("state_transitions", [])
    if summary.get("status") != "pass_with_negative_performance" or \
            [row.get("slots") for row in transitions] != [
                ["A", "B"], [None, "B"], ["C", "B"], [None, None]] or any(
                contracts.get(name) is not True for name in (
                    "length_refill_matches_independent_b1",
                    "delayed_sampling_matches_independent_b1",
                    "stop_releases_row", "cancel_releases_row",
                    "lowest_free_slot_reused", "other_row_survives_refill",
                    "fp32_cache", "bf16_cache", "hip_matches_cpu",
                    "greedy_d2h_calls_equal_scheduler_steps",
                    "allocated_cache_persists_after_active_bytes_zero",
                    "policy_mismatch_rejected")) or \
            contracts.get("positions_aware_parallel_kernel") is not False or \
            contracts.get("performance_speedup_claim") is not False:
        errors.append("continuous slot scheduler semantic evidence changed")
    if len(records) != 8 or len(divergent) != 5 or len(uniform) != 3 or any(
            row.get("status") != "pass" or
            row.get("continuous_outputs_equal") is not True
            for row in records):
        errors.append("continuous slot scheduler benchmark rows changed")
    if any(not 0.7 < float(row.get("continuous_over_reference", 0.0)) < 0.9 or
           int(row.get("continuous_uniform_batch_decode_calls", -1)) != 0 or
           int(row.get("continuous_dummy_decode_rows", 0)) <= 0
           for row in divergent):
        errors.append("divergent continuous negative performance gate changed")
    if any(float(row.get("continuous_over_reference", 0.0)) <= 1.0 or
           not 0.3 < float(row.get("continuous_tokens_per_second", 0.0)) /
                     float(row.get("static_batch_tokens_per_second", 1.0)) < 0.8 or
           int(row.get("continuous_divergent_batch_decode_calls", -1)) != 0 or
           int(row.get("continuous_dummy_decode_rows", -1)) != 0 or
           float(row.get("continuous_slot_utilization", 0.0)) != 1.0 or
           row.get("static_outputs_equal") is not True
           for row in uniform):
        errors.append("uniform continuous control gate changed")
    if len(release_records) != 8 or len(release_divergent) != 5 or \
            len(release_uniform) != 3 or any(
                row.get("status") != "pass" or
                row.get("continuous_outputs_equal") is not True or
                int(row.get("warmup", -1)) != 2 or
                int(row.get("repetitions", -1)) != 10
                for row in release_records):
        errors.append("continuous Release benchmark rows changed")
    if any(not 0.7 < float(row.get("continuous_over_reference", 0.0)) < 0.9 or
           int(row.get("continuous_uniform_batch_decode_calls", -1)) != 0 or
           int(row.get("continuous_dummy_decode_rows", 0)) <= 0
           for row in release_divergent):
        errors.append("divergent continuous Release gate changed")
    if any(not 1.4 < float(row.get("continuous_over_reference", 0.0)) < 2.4 or
           not 0.3 < float(row.get("continuous_tokens_per_second", 0.0)) /
                     float(row.get("static_batch_tokens_per_second", 1.0)) < 0.7 or
           int(row.get("continuous_divergent_batch_decode_calls", -1)) != 0 or
           int(row.get("continuous_dummy_decode_rows", -1)) != 0 or
           float(row.get("continuous_slot_utilization", 0.0)) != 1.0 or
           row.get("static_outputs_equal") is not True
           for row in release_uniform):
        errors.append("uniform continuous Release control changed")
    header = (REPOSITORY / "include" / "microllm" / "inference" /
              "scheduler.h").read_text(encoding="utf-8")
    source = (REPOSITORY / "src" / "inference" /
              "scheduler.cpp").read_text(encoding="utf-8")
    cpu_tests = (REPOSITORY / "tests" / "inference" /
                 "scheduler_test.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "ops" /
                 "hip_ops_test.cpp").read_text(encoding="utf-8")
    benchmark = (REPOSITORY / "benchmarks" / "end_to_end" /
                 "benchmark_scheduler.cpp").read_text(encoding="utf-8")
    if "class ContinuousBatchScheduler" not in header or \
            "struct ContinuousBatchMetrics" not in header or \
            "ContinuousBatchScheduler::step" not in source or \
            "RefillsFreedSlotAndMatchesIndependentRows" not in cpu_tests or \
            "ContinuousSlotsRefillAndMatchCpuWithOneSelectionCopyPerStep" not in hip_tests or \
            "--continuous-slots" not in benchmark:
        errors.append("continuous slot scheduler implementation gate is missing")
    if gates.get("status") != "pass_with_negative_performance" or \
            gates.get("full", {}).get("passed") != 306 or \
            gates.get("cpu", {}).get("passed") != 214 or \
            gates.get("hip", {}).get("passed") != 92 or \
            gates.get("sanitizer", {}).get("passed") != 207 or \
            gates.get("focused", {}).get("hip_benchmark_rows") != 16 or \
            gates.get("focused", {}).get("release_rows") != 8:
        errors.append("continuous slot scheduler final gates changed")
    return len(transitions), len(release_divergent), len(release_uniform)


def validate_active_row_compaction(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "097-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    candidate_paths = sorted((data / "release-matrix").glob("*.json"))
    candidates = [json.loads(path.read_text(encoding="utf-8"))
                  for path in candidate_paths]
    divergent = [(path, row) for path, row in zip(candidate_paths, candidates)
                 if path.name.startswith("divergent-")]
    uniform = [(path, row) for path, row in zip(candidate_paths, candidates)
               if path.name.startswith("uniform-")]
    baseline_directory = ROOT / "experiments" / "096-data" / "hip-release"
    speedups = []
    for path, candidate in divergent:
        baseline = json.loads(
            (baseline_directory / path.name).read_text(encoding="utf-8"))
        speedups.append(
            float(candidate["continuous_tokens_per_second"]) /
            float(baseline["continuous_tokens_per_second"]))
    contracts = summary.get("contracts", {})
    if summary.get("status") != "keep" or any(
            contracts.get(name) is not True for name in (
                "fp32_active_rows_match_independent_b1",
                "bf16_active_rows_match_independent_b1",
                "inactive_full_capacity_unchanged",
                "inactive_positions_unchanged",
                "shared_storage_address_stable",
                "full_uniform_fast_path_preserved", "hip_matches_cpu")) or \
            contracts.get("hip_payload_d2h_during_active_forward") != 0 or \
            contracts.get("dummy_rows_executed") != 0 or \
            contracts.get("cache_allocation_changed") is not False or \
            contracts.get("slot_lifecycle_changed") is not False:
        errors.append("active-row compaction semantic evidence changed")
    if len(candidates) != 8 or len(divergent) != 5 or len(uniform) != 3 or any(
            row.get("status") != "pass" or
            row.get("continuous_outputs_equal") is not True
            for row in candidates):
        errors.append("active-row compaction Release matrix changed")
    if any(not 1.1 < speedup < 1.4 for speedup in speedups) or any(
            int(row.get("continuous_dummy_decode_rows", -1)) != 0 or
            int(row.get("continuous_inactive_rows_skipped", 0)) <= 0 or
            int(row.get("continuous_compacted_batch_decode_calls", 0)) <= 0 or
            not 0.9 < float(row.get("continuous_over_reference", 0.0)) < 1.0
            for _, row in divergent):
        errors.append("active-row compaction divergent performance gate changed")
    if any(int(row.get("continuous_compacted_batch_decode_calls", -1)) != 0 or
           int(row.get("continuous_inactive_rows_skipped", -1)) != 0 or
           row.get("static_outputs_equal") is not True
           for _, row in uniform):
        errors.append("active-row compaction uniform no-regression gate changed")
    pair_records = 0
    for shape in ("r4s4", "r8s2"):
        paths = sorted((data / "paired" / shape).glob("*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        baseline = [row for path, row in zip(paths, rows)
                    if "baseline" in path.name]
        candidate = [row for path, row in zip(paths, rows)
                     if "candidate" in path.name]
        pair_records += len(rows)
        baseline_tps = sorted(float(row["continuous_tokens_per_second"])
                              for row in baseline)[1]
        candidate_tps = sorted(float(row["continuous_tokens_per_second"])
                               for row in candidate)[1]
        baseline_reference = sorted(float(row["scheduler_tokens_per_second"])
                                    for row in baseline)[1]
        candidate_reference = sorted(float(row["scheduler_tokens_per_second"])
                                     for row in candidate)[1]
        if len(rows) != 6 or len(baseline) != 3 or len(candidate) != 3 or \
                not 1.2 < candidate_tps / baseline_tps < 1.35 or \
                not 0.98 < candidate_reference / baseline_reference < 1.02 or \
                len({int(row["token_checksum"]) for row in rows}) != 1 or any(
                    row.get("continuous_outputs_equal") is not True for row in rows):
            errors.append(f"active-row alternating pair changed: {shape}")
    header = (REPOSITORY / "include" / "microllm" / "model" /
              "model.h").read_text(encoding="utf-8")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    scheduler_source = (REPOSITORY / "src" / "inference" /
                        "scheduler.cpp").read_text(encoding="utf-8")
    cpu_tests = (REPOSITORY / "tests" / "model" /
                 "model_test.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "inference" /
                 "hip_shape_matrix_test.cpp").read_text(encoding="utf-8")
    if "forward_cached_active_rows" not in header or \
            "TransformerModel::forward_cached_active_rows" not in model_source or \
            "inactive_rows_skipped" not in scheduler_source or \
            "ActiveCachedRowsSkipInactiveStorageAndMatchB1" not in cpu_tests or \
            "ActiveRowsSkipInactiveSlotAndMatchCpu" not in hip_tests:
        errors.append("active-row compaction source or executable gate is missing")
    if gates.get("status") != "keep" or \
            gates.get("full", {}).get("passed") != 308 or \
            gates.get("cpu", {}).get("passed") != 215 or \
            gates.get("hip", {}).get("passed") != 93 or \
            gates.get("sanitizer", {}).get("passed") != 208 or \
            gates.get("focused", {}).get("alternating_processes") != 12:
        errors.append("active-row compaction final gates changed")
    return len(candidates), pair_records, len(speedups)


def validate_positions_aware_decode(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "098-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    paths = sorted((data / "release-matrix").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    divergent = [(path, row) for path, row in zip(paths, records)
                 if path.name.startswith("divergent-")]
    uniform = [(path, row) for path, row in zip(paths, records)
               if path.name.startswith("uniform-")]
    baseline_directory = ROOT / "experiments" / "097-data" / "release-matrix"
    speedups = []
    for path, candidate in divergent:
        baseline = json.loads(
            (baseline_directory / path.name).read_text(encoding="utf-8"))
        speedups.append(
            float(candidate["continuous_tokens_per_second"]) /
            float(baseline["continuous_tokens_per_second"]))
    contracts = summary.get("contracts", {})
    if summary.get("status") != "keep" or any(
            contracts.get(name) is not True for name in (
                "interleaved_rope_matches_scalar_rows",
                "split_half_rope_matches_scalar_rows",
                "split_half_bias_rope_matches_scalar_rows",
                "mapped_kv_store_fp32", "mapped_kv_store_bf16",
                "per_row_visible_attention_prefix",
                "long_prefix_fallback_above_4096",
                "full_model_rows_match_independent_b1",
                "inactive_full_capacity_unchanged", "hip_matches_cpu")) or \
            contracts.get("hip_payload_d2h") != 0 or \
            contracts.get("cache_bytes_changed") is not False or \
            contracts.get("request_semantics_changed") is not False:
        errors.append("positions-aware decode semantic evidence changed")
    if len(records) != 8 or len(divergent) != 5 or len(uniform) != 3 or any(
            row.get("status") != "pass" or
            row.get("continuous_outputs_equal") is not True
            for row in records):
        errors.append("positions-aware Release matrix changed")
    if sum(speedup > 1.1 for speedup in speedups) != 4 or \
            sum(speedup < 0.9 for speedup in speedups) != 1 or any(
                int(row.get("continuous_positions_aware_batch_decode_calls", -1)) !=
                int(row.get("continuous_compacted_batch_decode_calls", -2)) or
                int(row.get("continuous_dummy_decode_rows", -1)) != 0
                for _, row in divergent):
        errors.append("positions-aware matrix mechanism or retained outlier changed")
    if any(int(row.get("continuous_positions_aware_batch_decode_calls", -1)) != 0 or
           row.get("static_outputs_equal") is not True for _, row in uniform):
        errors.append("positions-aware uniform control changed")
    pair_records = 0
    paired_speedups = []
    for shape in ("r8s2", "r8s4", "r4s4"):
        pair_paths = sorted((data / "paired" / shape).glob("*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in pair_paths]
        baseline = [row for path, row in zip(pair_paths, rows)
                    if "baseline" in path.name]
        candidate = [row for path, row in zip(pair_paths, rows)
                     if "candidate" in path.name]
        pair_records += len(rows)
        baseline_tps = sorted(float(row["continuous_tokens_per_second"])
                              for row in baseline)[1]
        candidate_tps = sorted(float(row["continuous_tokens_per_second"])
                               for row in candidate)[1]
        ratio = candidate_tps / baseline_tps
        paired_speedups.append(ratio)
        if len(rows) != 6 or len(baseline) != 3 or len(candidate) != 3 or \
                not 1.25 < ratio < 1.75 or \
                len({int(row["token_checksum"]) for row in rows}) != 1 or any(
                    row.get("continuous_outputs_equal") is not True
                    for row in rows) or any(
                    float(candidate[index]["continuous_tokens_per_second"]) <=
                    float(baseline[index]["continuous_tokens_per_second"])
                    for index in range(3)):
            errors.append(f"positions-aware alternating pair changed: {shape}")
    ops_header = (REPOSITORY / "include" / "microllm" / "ops" /
                  "ops.h").read_text(encoding="utf-8")
    ops_source = (REPOSITORY / "src" / "ops" /
                  "ops.cpp").read_text(encoding="utf-8")
    kernels = (REPOSITORY / "src" / "ops" / "hip" /
               "basic_kernels.hip").read_text(encoding="utf-8")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    cpu_tests = (REPOSITORY / "tests" / "ops" /
                 "ops_test.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "ops" /
                 "hip_ops_test.cpp").read_text(encoding="utf-8")
    required = ("rope_positions", "rope_split_half_positions",
                "kv_cache_store_pair_positions_",
                "cached_gqa_attention_positions")
    if any(name not in ops_header or name not in ops_source
           for name in required) or \
            "cached_attention_fused_positions_kernel" not in kernels or \
            "forward_cached_positions" not in model_source or \
            "PositionedRopeStoreAndAttentionMatchRowReferences" not in cpu_tests or \
            "LongFallbackMasksEachActivePrefix" not in hip_tests:
        errors.append("positions-aware source or executable gate is missing")
    if gates.get("status") != "keep" or \
            gates.get("full", {}).get("passed") != 311 or \
            gates.get("cpu", {}).get("passed") != 216 or \
            gates.get("hip", {}).get("passed") != 95 or \
            gates.get("sanitizer", {}).get("passed") != 209 or \
            gates.get("focused", {}).get("long_fallback_prefix") != 4097 or \
            gates.get("focused", {}).get("alternating_processes") != 18:
        errors.append("positions-aware final gates changed")
    return len(records), pair_records, len(paired_speedups)


def validate_continuous_profile_scatter_discard(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "099-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    profile_rows = []
    pftrace_count = 0
    for shape in ("r8s4", "r8s2"):
        directory = data / "profile" / shape
        stdout = json.loads((directory / "stdout.json").read_text(encoding="utf-8"))
        kernel_path = next(directory.glob("*_kernel_stats.csv"))
        with kernel_path.open(encoding="utf-8", newline="") as stream:
            kernels = list(csv.DictReader(stream))
        matmul = next(row for row in kernels if "matmul_typed_kernel" in row["Name"])
        copy_buffer = next(row for row in kernels if row["Name"] == "__amd_rocclr_copyBuffer")
        positioned_share = sum(
            float(row["Percentage"]) for row in kernels
            if any(name in row["Name"] for name in (
                "rope_positions_kernel",
                "kv_cache_store_pair_positions_kernel",
                "cached_attention_fused_positions_kernel"))) / 100.0
        profile_rows.append((stdout, float(matmul["Percentage"]) / 100.0,
                             float(copy_buffer["Percentage"]) / 100.0,
                             positioned_share))
        pftrace_count += len(list(directory.glob("*_results.pftrace")))
    if summary.get("status") != "discard" or pftrace_count != 2 or any(
            row.get("status") != "pass" or
            row.get("scheduler") != "continuous_profile" or
            row.get("correctness_gate") != "external_full_suite" or
            int(row.get("measured_d2d_calls", -1)) != 159 or
            int(row.get("measured_d2d_bytes", -1)) != 113664 or
            not 0.60 < matmul_share < 0.65 or
            not 0.08 < copy_share < 0.11 or
            not 0.05 < positioned_share < 0.09
            for row, matmul_share, copy_share, positioned_share in profile_rows):
        errors.append("continuous-only clean profile evidence changed")
    interpretation = summary.get("interpretation", {})
    if any(interpretation.get(name) is not False for name in (
            "hip_memcpy_api_duration_is_copy_bandwidth",
            "all_copy_buffer_time_is_logit_scatter",
            "positioned_attention_is_current_primary_hotspot",
            "tiny_typed_gemm_generalizes_to_official_models")):
        errors.append("continuous profile interpretation boundary changed")
    pair_records = 0
    ratios = []
    for shape in ("r8s4", "r8s2"):
        paths = sorted((data / "paired" / shape).glob("*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        baseline = [row for path, row in zip(paths, rows)
                    if "baseline" in path.name]
        candidate = [row for path, row in zip(paths, rows)
                     if "candidate" in path.name]
        pair_records += len(rows)
        baseline_tps = sorted(float(row["continuous_tokens_per_second"])
                              for row in baseline)[1]
        candidate_tps = sorted(float(row["continuous_tokens_per_second"])
                               for row in candidate)[1]
        ratio = candidate_tps / baseline_tps
        ratios.append(ratio)
        if len(rows) != 6 or not 0.95 < ratio < 1.01 or \
                len({int(row["token_checksum"]) for row in rows}) != 1 or any(
                    row.get("continuous_outputs_equal") is not True for row in rows):
            errors.append(f"scatter discard alternating evidence changed: {shape}")
    benchmark = (REPOSITORY / "benchmarks" / "end_to_end" /
                 "benchmark_scheduler.cpp").read_text(encoding="utf-8")
    cmake = (REPOSITORY / "benchmarks" / "CMakeLists.txt").read_text(
        encoding="utf-8")
    ops_header = (REPOSITORY / "include" / "microllm" / "ops" /
                  "ops.h").read_text(encoding="utf-8")
    kernels = (REPOSITORY / "src" / "ops" / "hip" /
               "basic_kernels.hip").read_text(encoding="utf-8")
    scheduler = (REPOSITORY / "src" / "inference" /
                 "scheduler.cpp").read_text(encoding="utf-8")
    if "--continuous-only" not in benchmark or \
            "correctness_gate" not in benchmark or \
            "SchedulerContinuousProfileSmoke" not in cmake:
        errors.append("continuous-only profile mode or schema smoke is missing")
    if "scatter_rows" in ops_header or "scatter_rows" in kernels or \
            "scatter_rows" in scheduler or "logit_scatter_calls" in scheduler:
        errors.append("rejected logits scatter remains in source")
    if gates.get("status") != "discard" or \
            gates.get("full", {}).get("passed") != 312 or \
            gates.get("cpu", {}).get("passed") != 217 or \
            gates.get("hip", {}).get("passed") != 95 or \
            gates.get("sanitizer", {}).get("passed") != 210 or \
            gates.get("focused", {}).get("scatter_source_reverted") is not True:
        errors.append("continuous profile/scatter final gates changed")
    return len(profile_rows), pair_records, len(ratios)


def validate_packed_decode_metadata(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "100-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    profiles = []
    pair_records = 0
    ratios = []
    expected_calls = {"r8s4": 16, "r8s2": 24}
    baseline_calls = {"r8s4": 32, "r8s2": 56}
    for shape in ("r8s4", "r8s2"):
        directory = data / "paired" / shape
        profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        profiles.append(profile)
        paths = sorted(path for path in directory.glob("pair*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        baseline = [row for path, row in zip(paths, rows)
                    if "baseline" in path.name]
        candidate = [row for path, row in zip(paths, rows)
                     if "candidate" in path.name]
        pair_records += len(rows)
        baseline_tps = sorted(float(row["continuous_tokens_per_second"])
                              for row in baseline)[1]
        candidate_tps = sorted(float(row["continuous_tokens_per_second"])
                               for row in candidate)[1]
        ratio = candidate_tps / baseline_tps
        ratios.append(ratio)
        if profile.get("scheduler") != "continuous_profile" or \
                int(profile.get("measured_h2d_calls", -1)) != expected_calls[shape] or \
                int(profile.get("measured_h2d_bytes", -1)) != 596 or \
                len(rows) != 6 or not 1.02 < ratio < 1.08 or any(
                    float(candidate[index]["continuous_tokens_per_second"]) <=
                    float(baseline[index]["continuous_tokens_per_second"])
                    for index in range(3)) or \
                len({int(row["token_checksum"]) for row in rows}) != 1 or any(
                    row.get("continuous_outputs_equal") is not True for row in rows):
            errors.append(f"packed metadata evidence changed: {shape}")
    mechanism = summary.get("mechanism", [])
    contracts = summary.get("contracts", {})
    if summary.get("status") != "keep" or len(mechanism) != 2 or any(
            int(row.get("baseline_h2d_calls", -1)) != baseline_calls[
                "r8s4" if row.get("shape") == "R8S4" else "r8s2"] or
            int(row.get("h2d_bytes_before", -1)) !=
            int(row.get("h2d_bytes_after", -2))
            for row in mechanism) or any(
                contracts.get(name) is not True for name in (
                    "single_packed_h2d_per_positions_call",
                    "device_token_fallback_preserved",
                    "h2d_bytes_unchanged", "d2h_unchanged",
                    "d2d_unchanged", "cache_bytes_unchanged",
                    "outputs_equal")):
        errors.append("packed metadata mechanism or contracts changed")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "ops" /
                 "hip_ops_test.cpp").read_text(encoding="utf-8")
    if "packed_values, {3, active}" not in model_source or \
            "transfers.host_to_device_bytes, 76U" not in hip_tests:
        errors.append("packed metadata source or exact transfer gate is missing")
    if gates.get("status") != "keep" or \
            gates.get("full", {}).get("passed") != 312 or \
            gates.get("cpu", {}).get("passed") != 217 or \
            gates.get("hip", {}).get("passed") != 95 or \
            gates.get("sanitizer", {}).get("passed") != 210 or \
            gates.get("focused", {}).get("alternating_processes") != 12:
        errors.append("packed metadata final gates changed")
    return len(profiles), pair_records, len(ratios)


def validate_batched_slot_prefill(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "101-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    expected = {
        "uniform-r8s8": (1, 1, 8, 2.8, 3.1),
        "r8s4": (5, 3, 6, 1.2, 1.4),
        "r8s2": (7, 1, 2, 1.02, 1.12),
    }
    profile_count = 0
    pair_records = 0
    ratios = []
    for shape, (batch_calls, batched_calls, batched_rows,
                lower, upper) in expected.items():
        directory = data / "paired" / shape
        profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        profile_count += 1
        paths = sorted(directory.glob("pair*.json"))
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        baseline = [row for path, row in zip(paths, rows)
                    if "baseline" in path.name]
        candidate = [row for path, row in zip(paths, rows)
                     if "candidate" in path.name]
        pair_records += len(rows)
        baseline_tps = sorted(float(row["continuous_tokens_per_second"])
                              for row in baseline)[1]
        candidate_tps = sorted(float(row["continuous_tokens_per_second"])
                               for row in candidate)[1]
        ratio = candidate_tps / baseline_tps
        ratios.append(ratio)
        if int(profile.get("continuous_row_prefill_calls", -1)) != 8 or \
                int(profile.get("continuous_prefill_batch_calls", -1)) != batch_calls or \
                int(profile.get("continuous_batched_prefill_calls", -1)) != batched_calls or \
                int(profile.get("continuous_batched_prefill_rows", -1)) != batched_rows or \
                len(rows) != 6 or not lower < ratio < upper or any(
                    float(candidate[index]["continuous_tokens_per_second"]) <=
                    float(baseline[index]["continuous_tokens_per_second"])
                    for index in range(3)) or \
                len({int(row["token_checksum"]) for row in rows}) != 1 or any(
                    row.get("continuous_outputs_equal") is not True for row in rows):
            errors.append(f"batched slot prefill evidence changed: {shape}")
    contracts = summary.get("contracts", {})
    memory_note = summary.get("memory_note", {})
    if summary.get("status") != "keep" or any(
            contracts.get(name) is not True for name in (
                "equal_length_prompts_share_one_model_prefill",
                "different_lengths_use_separate_stable_groups",
                "logical_prefill_rows_unchanged",
                "partial_shared_rows_supported", "existing_rows_preserved",
                "fp32_cache", "bf16_cache", "hip_matches_cpu",
                "outputs_equal", "allocated_cache_unchanged")) or \
            contracts.get("hip_payload_d2h") != 0 or \
            memory_note.get("classified_as_leak") is not False:
        errors.append("batched slot prefill contracts changed")
    model_header = (REPOSITORY / "include" / "microllm" / "model" /
                    "model.h").read_text(encoding="utf-8")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    scheduler_header = (REPOSITORY / "include" / "microllm" / "inference" /
                        "scheduler.h").read_text(encoding="utf-8")
    scheduler_source = (REPOSITORY / "src" / "inference" /
                        "scheduler.cpp").read_text(encoding="utf-8")
    cpu_tests = (REPOSITORY / "tests" / "model" /
                 "model_test.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "inference" /
                 "hip_shape_matrix_test.cpp").read_text(encoding="utf-8")
    if "forward_prefill_cached_rows" not in model_header or \
            "TransformerModel::forward_prefill_cached_rows" not in model_source or \
            "batched_prefill_rows" not in scheduler_header or \
            "prompt_length" not in scheduler_source or \
            "BatchedRowPrefillMapsEqualPromptsIntoEmptySlots" not in cpu_tests or \
            "BatchedRowPrefillMapsEqualPromptsAndMatchesCpu" not in hip_tests or \
            "transfers.host_to_device_calls, 5U" not in (
                REPOSITORY / "tests" / "ops" / "hip_ops_test.cpp").read_text(
                    encoding="utf-8"):
        errors.append("batched slot prefill source or tests are missing")
    if gates.get("status") != "keep" or \
            gates.get("full", {}).get("passed") != 314 or \
            gates.get("cpu", {}).get("passed") != 218 or \
            gates.get("hip", {}).get("passed") != 96 or \
            gates.get("sanitizer", {}).get("passed") != 211 or \
            gates.get("focused", {}).get("alternating_processes") != 18:
        errors.append("batched slot prefill final gates changed")
    return profile_count, pair_records, len(ratios)


def validate_official_continuous_serving(
        errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "102-data"
    raw = [json.loads(line) for line in
           (data / "micro-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "micro-summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((data / "comparison.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    pytorch = [json.loads(path.read_text(encoding="utf-8")) for path in
               sorted((data / "pytorch").glob("*/*.json"))]
    keys = {(row.get("model"), row.get("case"), row.get("process_run"))
            for row in raw}
    if len(raw) != 24 or len(keys) != 24 or any(
            row.get("status") != "pass" or
            row.get("deterministic_across_steps") is not True or
            row.get("allocated_cache_bytes") != row.get("expected_cache_bytes") or
            not 0 < row.get("peak_active_cache_bytes", 0) <=
            row.get("allocated_cache_bytes", 0) or
            not 0 < row.get("kv_cache_byte_utilization", 0) <= 1 or
            len(row.get("generated_tokens", [])) != row.get("request_count") or
            any(len(tokens) != expected for tokens, expected in zip(
                row.get("generated_tokens", []), row.get("new_token_lengths", [])))
            for row in raw):
        errors.append("official continuous microLLM raw evidence changed")
    if summary.get("track") != "official_continuous_serving_matrix" or \
            summary.get("status") != "pass" or summary.get("runs") != 3 or \
            len(summary.get("aggregates", [])) != 8 or any(
                row.get("successful_runs") != 3 or row.get("status") != "pass"
                for row in summary.get("aggregates", [])):
        errors.append("official continuous microLLM aggregate changed")
    if len(pytorch) != 8 or any(
            row.get("record_type") != "pytorch_sequential_request_reference" or
            row.get("serving_mode") != "sequential_requests" or
            row.get("status") != "pass" or
            row.get("deterministic_across_steps") is not True
            for row in pytorch):
        errors.append("official continuous PyTorch reference changed")
    rows = comparison.get("rows", [])
    qwen_exact = sum(row.get("exact_generated_tokens") is True for row in rows
                     if row.get("model") == "qwen2.5-0.5b")
    deepseek_exact = sum(row.get("exact_generated_tokens") is True for row in rows
                         if row.get("model") ==
                         "deepseek-r1-distill-qwen-1.5b")
    if comparison.get("status") != "complete_with_recorded_accuracy_failures" or \
            "sequential requests" not in comparison.get("comparison_boundary", "") or \
            len(rows) != 8 or qwen_exact != 4 or deepseek_exact != 1 or any(
                not math.isfinite(row.get("observed_service_throughput_ratio", 0)) or
                row.get("observed_service_throughput_ratio", 0) <= 0 or
                not 0 < row.get("micro_kv_cache_byte_utilization", 0) <= 1 or
                not 0 < row.get("micro_slot_utilization", 0) <= 1
                for row in rows):
        errors.append("official continuous comparison boundary or accuracy gate changed")
    scheduler_header = (REPOSITORY / "include" / "microllm" / "inference" /
                        "scheduler.h").read_text(encoding="utf-8")
    scheduler_source = (REPOSITORY / "src" / "inference" /
                        "scheduler.cpp").read_text(encoding="utf-8")
    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    contracts = (REPOSITORY / "python" / "tests" /
                 "test_hf_continuous_matrix.py").read_text(encoding="utf-8")
    if "max_sequence_length" not in scheduler_header or \
            "configured_capacity" not in scheduler_source or \
            'workload == "continuous"' not in app or \
            "generated_tokens" not in app or \
            "test_cache_formula_uses_request_bound_not_model_maximum" not in contracts or \
            "test_comparison_marks_token_mismatch_and_names_boundary" not in contracts:
        errors.append("official continuous source or contract tests are missing")
    if gates.get("status") != "complete_with_recorded_accuracy_failures" or \
            gates.get("full", {}).get("passed") != 315 or \
            gates.get("cpu", {}).get("passed") != 219 or \
            gates.get("hip", {}).get("passed") != 96 or \
            gates.get("sanitizer", {}).get("passed") != 212 or \
            gates.get("focused", {}).get("micro_processes") != 24 or \
            gates.get("focused", {}).get("deepseek_mismatched_cases") != 3:
        errors.append("official continuous final gates changed")
    return len(raw), len(pytorch), qwen_exact, deepseek_exact


def validate_fixed_request_slot_sweep(
        errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "103-data"
    before = [json.loads(line) for line in
              (data / "before-fix-raw.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    after = [json.loads(line) for line in
             (data / "after-raw.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    summary = json.loads((data / "after-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    failed = [row for row in before if row.get("status") == "failed"]
    expected_failed_cases = {"short_s1", "long_s1", "long_s2"}
    if len(before) != 48 or len(failed) != 18 or \
            {row.get("case") for row in failed} != expected_failed_cases or \
            any("KV prefix requires an empty cache" not in row.get("error", "")
                for row in failed) or \
            len([row for row in before if row.get("status") == "pass"]) != 30:
        errors.append("fixed-request pre-fix failure evidence changed")
    after_keys = {(row.get("model"), row.get("case"), row.get("process_run"))
                  for row in after}
    if len(after) != 48 or len(after_keys) != 48 or any(
            row.get("status") != "pass" or
            row.get("allocated_cache_bytes") != row.get("expected_cache_bytes") or
            row.get("deterministic_across_steps") is not True
            for row in after):
        errors.append("fixed-request post-fix execution evidence changed")
    sweeps = summary.get("slot_sweeps", [])
    by_group = {(row.get("model"), row.get("group")): row for row in sweeps}
    expected_groups = {
        ("qwen2.5-0.5b", "short"): True,
        ("qwen2.5-0.5b", "long"): True,
        ("deepseek-r1-distill-qwen-1.5b", "short"): False,
        ("deepseek-r1-distill-qwen-1.5b", "long"): True,
    }
    if summary.get("execution_status") != "pass" or \
            summary.get("status") != "complete_with_recorded_accuracy_failures" or \
            len(sweeps) != 4 or set(by_group) != set(expected_groups):
        errors.append("fixed-request slot-sweep summary changed")
    for key, expected_exact in expected_groups.items():
        sweep = by_group.get(key, {})
        slots = sweep.get("slots", [])
        if sweep.get("generated_tokens_equal_across_slots") is not expected_exact or \
                [row.get("slots") for row in slots] != [1, 2, 4, 8] or \
                len(slots) != 4:
            errors.append(f"fixed-request slot contract changed: {key}")
            continue
        baseline_cache = slots[0]["allocated_cache_bytes"]
        for row in slots:
            slot = row["slots"]
            if row["allocated_cache_bytes"] != baseline_cache * slot or \
                    not 0 < row["parallel_efficiency_vs_s1"] <= 1.01 or \
                    not 0 < row["kv_cache_byte_utilization"] <= 1 or \
                    not 0 < row["slot_utilization"] <= 1:
                errors.append(f"fixed-request efficiency/cache gate changed: {key} S{slot}")
        if key == ("deepseek-r1-distill-qwen-1.5b", "short"):
            for row in slots[2:]:
                difference = row.get("token_difference_vs_s1", {})
                if difference.get("differing_requests") != [5] or \
                        difference.get("first_difference") != {
                            "request": 5, "token": 4}:
                    errors.append("DeepSeek short slot divergence changed")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    cpu_tests = (REPOSITORY / "tests" / "inference" /
                 "scheduler_test.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests" / "ops" /
                 "hip_ops_test.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_continuous_matrix.py").read_text(encoding="utf-8")
    if "storage_is_empty" not in model_source or \
            "&& storage_is_empty" not in model_source or \
            "recycled.metrics().slot_refills" not in cpu_tests or \
            "recycled_hip.metrics().slot_refills" not in hip_tests or \
            '"slot-sweep"' not in runner or "token_difference" not in runner:
        errors.append("fixed-request recycle source or tests are missing")
    if gates.get("status") != "complete_with_recorded_accuracy_failures" or \
            gates.get("full", {}).get("passed") != 315 or \
            gates.get("cpu", {}).get("passed") != 219 or \
            gates.get("hip", {}).get("passed") != 96 or \
            gates.get("sanitizer", {}).get("passed") != 212 or \
            gates.get("focused", {}).get("before_stable_failures") != 18 or \
            gates.get("focused", {}).get("after_passed") != 48:
        errors.append("fixed-request final gates changed")
    return len(before), len(after), sum(expected_groups.values()), \
        len(expected_groups) - sum(expected_groups.values())


def validate_deepseek_prefill_divergence(
        errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "104-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    expected_cases = {
        "short_s1", "short_s2", "short_s4", "short_s8",
        "short_s4_serial_prefill", "short_s8_serial_prefill",
    }
    keys = {(row.get("case"), row.get("process_run")) for row in raw}
    if len(raw) != 18 or len(keys) != 18 or \
            {row.get("case") for row in raw} != expected_cases or any(
                row.get("status") != "pass" or
                row.get("selection_diagnostic_count") != 96 or
                len(row.get("selection_diagnostics", [])) != 96 or
                any(not diagnostic.get("device_argmax_matches_top1")
                    for diagnostic in row.get("selection_diagnostics", []))
                for row in raw):
        errors.append("DeepSeek prefill divergence raw evidence changed")
    if summary.get("track") != "official_continuous_slot_divergence" or \
            summary.get("status") != "complete_with_recorded_accuracy_failure" or \
            summary.get("runs") != 3 or \
            summary.get("first_difference") != {"request": 5, "token": 4} or \
            len(summary.get("diagnostic_evidence", [])) != 6 or \
            "excluded" not in summary.get("measurement_boundary", ""):
        errors.append("DeepSeek prefill divergence summary changed")
    evidence = {row["case"]: row for row in summary.get("diagnostic_evidence", [])}
    expected = {
        "short_s1": ([23606], [1], 0.015623093),
        "short_s2": ([23606], [2], 0.011352539),
        "short_s4": ([1196], [4], 0.000669479),
        "short_s8": ([1196], [8], 0.000669479),
        "short_s4_serial_prefill": ([23606], [4], 0.011352539),
        "short_s8_serial_prefill": ([23606], [8], 0.011352539),
    }
    for case, (tokens, batches, margin) in expected.items():
        row = evidence.get(case, {})
        if row.get("selected_tokens") != tokens or \
                row.get("logit_batch_sizes") != batches or \
                row.get("cache_positions") != [36] or \
                row.get("stable_across_runs") is not True or \
                abs(float(row.get("margin_p50", -1)) - margin) > 1.0e-7:
            errors.append(f"DeepSeek divergence diagnostic changed: {case}")
    comparisons = {row["case"]: row for row in summary.get("comparisons", [])}
    if comparisons.get("short_s4", {}).get("difference_vs_s1", {}).get("exact") \
            is not False or \
            comparisons.get("short_s4_serial_prefill", {}).get(
                "difference_vs_s1", {}).get("exact") is not True:
        errors.append("DeepSeek prefill counterfactual no longer refutes decode")
    pytorch = summary.get("pytorch_comparison", {})
    pytorch_rows = {row["case"]: row for row in pytorch.get("comparisons", [])}
    if pytorch.get("default_s4_matches_reference_at_original_divergence") is not True or \
            pytorch.get("serial_s4_matches_reference_at_original_divergence") is not False or \
            pytorch_rows.get("short_s4", {}).get(
                "difference_vs_pytorch", {}).get("differing_requests") != [7] or \
            pytorch_rows.get("short_s4_serial_prefill", {}).get(
                "difference_vs_pytorch", {}).get("differing_requests") != [5, 7]:
        errors.append("DeepSeek PyTorch no-rollback gate changed")
    scheduler_header = (REPOSITORY / "include" / "microllm" / "inference" /
                        "scheduler.h").read_text(encoding="utf-8")
    scheduler_source = (REPOSITORY / "src" / "inference" /
                        "scheduler.cpp").read_text(encoding="utf-8")
    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "python" / "tests" /
             "test_hf_continuous_matrix.py").read_text(encoding="utf-8")
    if "SelectionDiagnostic" not in scheduler_header or \
            "capture_selection_diagnostics" not in scheduler_source or \
            "batch_equal_length_prefill" not in scheduler_source or \
            "--continuous-diagnostics" not in app or \
            "--continuous-prefill-batch" not in app or \
            "test_divergence_summary_keeps_top2_source_and_margin" not in tests:
        errors.append("DeepSeek divergence source or tests are missing")
    if gates.get("status") != "keep_diagnostics_reject_serial_default" or \
            gates.get("full", {}).get("passed") != 315 or \
            gates.get("cpu", {}).get("passed") != 219 or \
            gates.get("hip", {}).get("passed") != 96 or \
            gates.get("sanitizer", {}).get("passed") != 212 or \
            gates.get("focused", {}).get("diagnostic_processes") != 18 or \
            gates.get("focused", {}).get("device_argmax_top1_mismatches") != 0:
        errors.append("DeepSeek divergence final gates changed")
    return len(raw), len(evidence), \
        gates.get("focused", {}).get("default_processes", 0), \
        gates.get("focused", {}).get("counterfactual_processes", 0)


def validate_b2_prefill_row_audit(
        errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "105-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    expected_offsets = {
        "single_5": [5], "pair_4_5": [4, 5],
        "pair_5_4": [5, 4], "duplicate_5": [5, 5],
    }
    keys = {(row.get("case"), row.get("process_run")) for row in raw}
    if len(raw) != 12 or len(keys) != 12 or any(
            row.get("status") != "pass" or
            row.get("prompt_offsets") != expected_offsets.get(row.get("case")) or
            any(not diagnostic.get("device_argmax_matches_top1")
                for diagnostic in row.get("selection_diagnostics", []))
            for row in raw):
        errors.append("B2 prefill row-audit raw evidence changed")
    required_true = (
        "b2_target_outputs_equal_across_row_order_and_duplicates",
        "duplicate_b2_prefill_numeric_signatures_equal",
        "swapped_b2_target_prefill_signatures_equal",
        "single_and_b2_outputs_differ",
    )
    if summary.get("track") != "official_b2_prefill_row_audit" or \
            summary.get("status") != "pass" or summary.get("runs") != 3 or \
            any(summary.get(name) is not True for name in required_true) or \
            "does not follow local row" not in summary.get("conclusion", "") or \
            len(summary.get("target_rows", [])) != 5:
        errors.append("B2 prefill row-audit summary changed")
    rows = {(row["case"], row["request"]): row
            for row in summary.get("target_rows", [])}
    single = rows.get(("single_5", 0), {})
    b2_keys = (("pair_4_5", 1), ("pair_5_4", 0),
               ("duplicate_5", 0), ("duplicate_5", 1))
    if single.get("generated_equal_to_single") is not True or \
            single.get("decision_diagnostic", {}).get("top1_token") != 23606 or \
            abs(float(single.get("prefill_diagnostic", {}).get(
                "top1_logit", -1)) - 12.352085114) > 1.0e-7:
        errors.append("B1 row-audit reference changed")
    for key in b2_keys:
        row = rows.get(key, {})
        prefill = row.get("prefill_diagnostic", {})
        decision = row.get("decision_diagnostic", {})
        if row.get("generated_equal_to_single") is not False or \
                row.get("first_difference_vs_single") != {
                    "request": 0, "token": 4} or \
                prefill.get("logit_batch_size") != 2 or \
                abs(float(prefill.get("top1_logit", -1)) -
                    12.29726696) > 1.0e-7 or \
                decision.get("top1_token") != 1196:
            errors.append(f"B2 row-audit target changed: {key}")
    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_prefill_row_audit.py").read_text(encoding="utf-8")
    tests = (REPOSITORY / "python" / "tests" /
             "test_hf_continuous_matrix.py").read_text(encoding="utf-8")
    if "--continuous-prompt-offsets" not in app or \
            "prompt_offsets" not in app or \
            '"duplicate_5"' not in runner or \
            "test_prefill_row_audit_command_preserves_explicit_offsets" not in tests:
        errors.append("B2 row-audit source or tests are missing")
    if gates.get("status") != "row_copy_refuted" or \
            gates.get("full", {}).get("passed") != 315 or \
            gates.get("cpu", {}).get("passed") != 219 or \
            gates.get("hip", {}).get("passed") != 96 or \
            gates.get("sanitizer", {}).get("passed") != 212 or \
            gates.get("focused", {}).get("diagnostic_processes") != 12:
        errors.append("B2 row-audit final gates changed")
    return len(raw), len(rows), len(b2_keys), \
        gates.get("focused", {}).get("device_argmax_top1_mismatches", -1)


def validate_prefill_layer_drift(
        errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "106-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if len(raw) != 3 or any(
            row.get("status") != "pass" or len(row.get("stages", [])) != 31 or
            row.get("trace_record_count_b1") != 33 or
            row.get("trace_record_count_b2") != 33
            for row in raw) or any(
                row.get("stages") != raw[0].get("stages") for row in raw[1:]):
        errors.append("prefill layer-drift fresh-pair evidence changed")
    stages = {row["name"]: row for row in summary.get("stages", [])}
    if summary.get("track") != "official_prefill_layer_drift" or \
            summary.get("status") != "pass" or summary.get("runs") != 3 or \
            summary.get("stage_count") != 31 or \
            summary.get("first_nonzero_stage") != "inference.blocks.0" or \
            summary.get("duplicate_b2_rows_exact_at_every_stage") is not True or \
            "no performance claim" not in summary.get("measurement_boundary", ""):
        errors.append("prefill layer-drift summary changed")
    expected = {
        "inference.embedding": (0.0, 0.0),
        "inference.blocks.0": (0.00135040283203125,
                               0.00005165932698272103),
        "inference.blocks.27": (1.9002685546875,
                                0.0062605414066010815),
        "inference.final_norm": (0.3941173553466797,
                                 0.00841151503415215),
        "inference.logits": (0.1530160903930664,
                             0.013777231522314946),
    }
    for name, (maximum, relative) in expected.items():
        row = stages.get(name, {})
        drift = row.get("b1_vs_b2_row0", {})
        duplicate = row.get("b2_row0_vs_row1", {})
        if abs(float(drift.get("max_abs", -1)) - maximum) > 1.0e-10 or \
                abs(float(drift.get("relative_l2", -1)) - relative) > 1.0e-12 or \
                duplicate.get("exact") is not True:
            errors.append(f"prefill layer-drift stage changed: {name}")
    logits = stages.get("inference.logits", {}).get("b1_vs_b2_row0", {})
    if abs(float(logits.get("mean_abs", -1)) - 0.02892810391008479) > 1.0e-12 or \
            int(logits.get("elements", 0)) != 151936:
        errors.append("complete-logit drift evidence changed")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_prefill_layer_drift.py").read_text(encoding="utf-8")
    model_tests = (REPOSITORY / "tests" / "model" /
                   "model_test.cpp").read_text(encoding="utf-8")
    contracts = (REPOSITORY / "python" / "tests" /
                 "test_hf_continuous_matrix.py").read_text(encoding="utf-8")
    if '"inference.blocks."' not in model_source or \
            "--trace-max-elements" not in app or \
            "trace_scope.reset" not in app or \
            "compare_traces" not in runner or \
            '"inference.embedding"' not in model_tests or \
            "test_layer_drift_compares_complete_rows_and_rejects_truncation" \
            not in contracts:
        errors.append("prefill layer-drift source or tests are missing")
    if gates.get("status") != "first_drift_is_block0" or \
            gates.get("full", {}).get("passed") != 315 or \
            gates.get("cpu", {}).get("passed") != 219 or \
            gates.get("hip", {}).get("passed") != 96 or \
            gates.get("sanitizer", {}).get("passed") != 212 or \
            gates.get("focused", {}).get("fresh_pairs") != 3:
        errors.append("prefill layer-drift final gates changed")
    exact_duplicate_stages = sum(
        row.get("b2_row0_vs_row1", {}).get("exact") is True
        for row in summary.get("stages", []))
    return len(raw), len(stages), exact_duplicate_stages, \
        int(logits.get("elements", 0))


def validate_block0_drift(
        errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "107-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if len(raw) != 3 or any(
            row.get("status") != "pass" or len(row.get("stages", [])) != 43 or
            row.get("trace_record_count_b1") != 45 or
            row.get("trace_record_count_b2") != 45
            for row in raw) or any(
                row.get("stages") != raw[0].get("stages") for row in raw[1:]):
        errors.append("block0 drift fresh-pair evidence changed")
    stages = {row["name"]: row for row in summary.get("stages", [])}
    if summary.get("track") != "official_prefill_layer_drift" or \
            summary.get("status") != "pass" or summary.get("stage_count") != 43 or \
            summary.get("first_nonzero_stage") != \
            "inference.blocks.0.ffn.output" or \
            summary.get("duplicate_b2_rows_exact_at_every_stage") is not True:
        errors.append("block0 drift summary changed")
    exact_names = (
        "inference.blocks.0.attention_norm",
        "inference.blocks.0.attention.q_projection",
        "inference.blocks.0.attention.k_projection",
        "inference.blocks.0.attention.v_projection",
        "inference.blocks.0.attention.q_rope",
        "inference.blocks.0.attention.k_rope",
        "inference.blocks.0.attention.value",
        "inference.blocks.0.attention.context",
        "inference.blocks.0.attention.output",
        "inference.blocks.0.attention_residual",
        "inference.blocks.0.ffn_norm",
    )
    if any(stages.get(name, {}).get("b1_vs_b2_row0", {}).get("exact") is not True
           for name in exact_names):
        errors.append("block0 pre-FFN exact boundary changed")
    ffn = stages.get("inference.blocks.0.ffn.output", {}).get(
        "b1_vs_b2_row0", {})
    block = stages.get("inference.blocks.0", {}).get("b1_vs_b2_row0", {})
    if abs(float(ffn.get("max_abs", -1)) - 0.00135040283203125) > 1.0e-12 or \
            abs(float(ffn.get("relative_l2", -1)) -
                0.00007269202489080616) > 1.0e-14 or \
            abs(float(block.get("max_abs", -1)) -
                float(ffn.get("max_abs", -2))) > 1.0e-15:
        errors.append("block0 FFN first-drift value changed")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    model_tests = (REPOSITORY / "tests" / "model" /
                   "model_test.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_prefill_layer_drift.py").read_text(encoding="utf-8")
    if 'trace_detail(trace_prefix, "q_projection"' not in model_source or \
            'trace_detail(trace_prefix, "attention_norm"' not in model_source or \
            'trace_detail(trace_prefix, "output", reshaped)' not in model_source or \
            '"inference.blocks.0.attention.q_projection"' not in model_tests or \
            "right_shape[0] != 2 * left_shape[0]" not in runner:
        errors.append("block0 drift source or tests are missing")
    if gates.get("status") != "first_drift_is_block0_ffn_output" or \
            gates.get("full", {}).get("passed") != 315 or \
            gates.get("cpu", {}).get("passed") != 219 or \
            gates.get("hip", {}).get("passed") != 96 or \
            gates.get("sanitizer", {}).get("passed") != 212 or \
            gates.get("focused", {}).get(
                "block0_exact_substages_before_ffn_output") != 11:
        errors.append("block0 drift final gates changed")
    exact_duplicate = sum(
        row.get("b2_row0_vs_row1", {}).get("exact") is True
        for row in summary.get("stages", []))
    return len(raw), len(stages), len(exact_names), exact_duplicate


def validate_bf16_ffn_drift(errors: list[str]) -> tuple[int, int, int, int]:
    data = ROOT / "experiments" / "108-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if len(raw) != 3 or any(
            row.get("status") != "pass" or len(row.get("stages", [])) != 48 or
            row.get("trace_record_count_b1") != 50 or
            row.get("trace_record_count_b2") != 50
            for row in raw) or any(
                row.get("stages") != raw[0].get("stages") for row in raw[1:]):
        errors.append("BF16 FFN drift raw evidence changed")
    stages = {row["name"]: row for row in summary.get("stages", [])}
    if summary.get("stage_count") != 48 or summary.get("first_nonzero_stage") != \
            "inference.blocks.0.ffn.gate" or \
            summary.get("duplicate_b2_rows_exact_at_every_stage") is not True:
        errors.append("BF16 FFN drift summary changed")
    expected = {
        "inference.blocks.0.ffn.input_bf16": (0.0, 0.0),
        "inference.blocks.0.ffn.gate": (0.015625,
            0.00006106343369635627),
        "inference.blocks.0.ffn.up": (0.001953125,
            0.000019422787601982993),
        "inference.blocks.0.ffn.activated": (0.0078125,
            0.00011019402428538509),
        "inference.blocks.0.ffn.down": (0.00135040283203125,
            0.00007269202489080616),
    }
    for name, (maximum, relative) in expected.items():
        row = stages.get(name, {})
        drift = row.get("b1_vs_b2_row0", {})
        if abs(float(drift.get("max_abs", -1)) - maximum) > 1.0e-12 or \
                abs(float(drift.get("relative_l2", -1)) - relative) > 1.0e-14 or \
                row.get("b2_row0_vs_row1", {}).get("exact") is not True:
            errors.append(f"BF16 FFN internal stage changed: {name}")
    trace_source = (REPOSITORY / "src" / "profiling" /
                    "trace.cpp").read_text(encoding="utf-8")
    optimized = (REPOSITORY / "src" / "ops" /
                 "optimized.cpp").read_text(encoding="utf-8")
    model_source = (REPOSITORY / "src" / "model" /
                    "model.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests" / "profiling" /
             "trace_test.cpp").read_text(encoding="utf-8")
    if "is_floating_point(tensor.dtype())" not in trace_source or \
            "bf16_ffn_diagnostics" not in optimized or \
            'trace_detail(trace_prefix, "gate"' not in model_source or \
            "low_precision.statistics.finite_count" not in tests:
        errors.append("BF16 FFN drift source or low-precision trace tests are missing")
    if gates.get("status") != "first_drift_is_gate_gemm" or \
            gates.get("full", {}).get("passed") != 315 or \
            gates.get("sanitizer", {}).get("passed") != 212 or \
            gates.get("focused", {}).get("initial_low_precision_trace_failures") != 1:
        errors.append("BF16 FFN drift final gates changed")
    exact_duplicate = sum(row.get("b2_row0_vs_row1", {}).get("exact") is True
                          for row in summary.get("stages", []))
    return len(raw), len(stages), len(expected), exact_duplicate


def validate_bf16_algorithm_inventory(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "109-data"
    inventory = json.loads((data / "inventory.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    shapes = inventory.get("shapes", [])
    if inventory.get("record_type") != "bf16_algorithm_inventory" or \
            inventory.get("status") != "pass" or len(shapes) != 2 or \
            [row.get("rows") for row in shapes] != [32, 64] or \
            any(row.get("candidate_count") != 64 for row in shapes) or \
            inventory.get("common_candidate_count") != 53 or \
            len(inventory.get("common_indices", [])) != 53:
        errors.append("BF16 algorithm inventory changed")
    source = (REPOSITORY / "benchmarks" / "micro" /
              "benchmark_bf16_algorithms.cpp").read_text(encoding="utf-8")
    cmake = (REPOSITORY / "benchmarks" / "CMakeLists.txt").read_text(encoding="utf-8")
    if "hipblasLtMatmulAlgoGetHeuristic" not in source or \
            "getIndexFromAlgo" not in source or \
            "microllm_bench_bf16_algorithms" not in cmake:
        errors.append("BF16 algorithm inventory CLI is missing")
    if gates.get("status") != "common_algorithms_available" or \
            gates.get("focused", {}).get("common_candidates") != 53 or \
            gates.get("focused", {}).get("invalid_cli_failures") != 1:
        errors.append("BF16 algorithm inventory gates changed")
    return shapes[0].get("candidate_count", 0), \
        shapes[1].get("candidate_count", 0), \
        inventory.get("common_candidate_count", 0)


def validate_bf16_same_algorithm(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "110-data"
    precision = json.loads((data / "precision-summary.json").read_text(encoding="utf-8"))
    performance = json.loads((data / "performance-summary.json").read_text(encoding="utf-8"))
    precision_raw = [json.loads(line) for line in
                     (data / "precision-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    performance_raw = [json.loads(line) for line in
                       (data / "performance-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if len(precision_raw) != 3 or precision.get("algorithm_index") != 75892 or \
            precision.get("first_nonzero_stage") is not None or \
            precision.get("b1_b2_exact_at_every_stage") is not True or \
            precision.get("duplicate_b2_rows_exact_at_every_stage") is not True:
        errors.append("same-algorithm precision evidence changed")
    ratios = {row["batch"]: row["common_over_default"]
              for row in performance.get("rows", [])}
    if len(performance_raw) != 12 or performance.get("algorithm_index") != 75892 or \
            not 0.95 < ratios.get(1, 0) < 0.98 or \
            not 0.97 < ratios.get(2, 0) < 1.0:
        errors.append("same-algorithm performance evidence changed")
    optimized = (REPOSITORY / "src" / "ops" / "optimized.cpp").read_text(encoding="utf-8")
    app = (REPOSITORY / "apps" / "hf_infer.cpp").read_text(encoding="utf-8")
    if "register_bf16_algorithm" not in optimized or \
            "matmulIsAlgoSupported" not in optimized or \
            "--bf16-algorithm-index" not in app:
        errors.append("same-algorithm registry source is missing")
    if gates.get("status") != "keep_optional_strict_algorithm" or \
            gates.get("focused", {}).get("exact_stages") != 48:
        errors.append("same-algorithm gates changed")
    return len(precision_raw), len(performance_raw), 48


def validate_qwen_common_discard(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "111-data"
    inventory = json.loads((data / "inventory.json").read_text(encoding="utf-8"))
    precision = json.loads((data / "precision-summary.json").read_text(encoding="utf-8"))
    performance = json.loads((data / "performance-summary.json").read_text(encoding="utf-8"))
    if inventory.get("common_candidate_count") != 56 or \
            precision.get("algorithm_index") != 75789 or \
            precision.get("first_nonzero_stage") != "inference.blocks.0.ffn.gate" or \
            precision.get("b1_b2_exact_at_every_stage") is not False:
        errors.append("Qwen common-algorithm discard precision changed")
    ratios = {row["batch"]: row["common_over_default"] for row in performance.get("rows", [])}
    if not 0.98 < ratios.get(1, 0) < 1.01 or not 0.98 < ratios.get(2, 0) < 1.02:
        errors.append("Qwen common-algorithm discard performance changed")
    return inventory.get("common_candidate_count", 0), 3, 12


def validate_qwen_algorithm_search(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "112-data"
    first = json.loads((data / "first16-summary.json").read_text(encoding="utf-8"))
    rest = json.loads((data / "rest40-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if first.get("tested_candidates") != 16 or rest.get("tested_candidates") != 40 or \
            first.get("exact_candidate") is not None or rest.get("exact_candidate") is not None or \
            gates.get("best_candidate") != 75886:
        errors.append("Qwen full algorithm search changed")
    return 56, 0


def validate_request_latency(errors: list[str]) -> tuple[int, int]:
    data = ROOT / "experiments" / "113-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    if len(raw) != 48 or any(
            len(row.get("request_ttft_ms", [])) != row.get("request_count") or
            len(row.get("request_completion_ms", [])) != row.get("request_count") or
            any(done < first for first, done in zip(
                row.get("request_ttft_ms", []), row.get("request_completion_ms", [])))
            for row in raw):
        errors.append("request latency raw evidence changed")
    aggregates = summary.get("aggregates", [])
    if len(aggregates) != 16 or any(
            row.get("request_ttft_p95_ms_p50", -1) <
            row.get("request_ttft_p50_ms_p50", 0) for row in aggregates):
        errors.append("request latency aggregates changed")
    return len(raw), len(aggregates)


def validate_length_bucket_tradeoff(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "114-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    telemetry = (data / "gpu3-telemetry.log").read_text(
        encoding="utf-8").splitlines()
    utilization = [line for line in telemetry if
                   "GPU[3]" in line and "GPU use (%)" in line]
    if len(raw) != 12 or any(row.get("status") != "pass" for row in raw) or \
            sum(row.get("bucketed_cache") is True for row in raw) != 6 or \
            sum(row.get("bucketed_cache") is False for row in raw) != 6:
        errors.append("length-bucket raw process evidence changed")
    models = {row.get("model") for row in raw}
    for model in models:
        uniform = [row for row in raw if row.get("model") == model and
                   row.get("case") == "long_uniform_s8"]
        bucketed = [row for row in raw if row.get("model") == model and
                    row.get("case") == "long_bucketed_s8"]
        if len(uniform) != 3 or len(bucketed) != 3 or \
                len({json.dumps(row.get("generated_tokens"))
                     for row in uniform + bucketed}) != 1 or \
                any(row.get("request_bucket_indices") !=
                    [0, 0, 1, 1, 2, 2, 3, 3] for row in bucketed):
            errors.append(f"length-bucket token or routing evidence changed: {model}")
    aggregates = summary.get("aggregates", [])
    comparisons = summary.get("bucket_comparisons", [])
    if summary.get("status") != "pass" or len(aggregates) != 4 or \
            any(row.get("successful_runs") != 3 for row in aggregates) or \
            len(comparisons) != 2 or any(
                row.get("token_difference", {}).get("exact") is not True or
                abs(float(row.get("allocated_cache_ratio", 0.0)) -
                    0.47093023255813954) > 1.0e-12 or
                not 0.55 < float(row.get("tokens_per_second_ratio", 0.0)) < 0.60 or
                not 0.40 < float(row.get("request_ttft_p50_ratio", 0.0)) < 0.46 or
                not 1.70 < float(row.get("request_completion_p50_ratio", 0.0)) < 1.80
                for row in comparisons):
        errors.append("length-bucket aggregate tradeoff changed")
    if gates.get("status") != \
            "keep_as_opt_in_memory_policy_with_measured_throughput_limit" or \
            gates.get("full_release", {}).get("passed") != 318 or \
            gates.get("sanitizer", {}).get("passed") != 214 or \
            gates.get("official_matrix", {}).get("passed") != 12:
        errors.append("length-bucket test gates changed")
    if len(utilization) < 80 or utilization[0].split(":")[-1].strip() != "0" or \
            utilization[-1].split(":")[-1].strip() != "0":
        errors.append("length-bucket GPU telemetry boundary changed")
    return len(raw), len(comparisons), len(utilization)


def validate_bucket_pareto(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "115-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected = [json.loads(line) for line in
                (data / "rejected-partial-raw.jsonl").read_text(
                    encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    telemetry = (data / "rejected-gpu3-telemetry.log").read_text(
        encoding="utf-8").splitlines()
    vram = [int(line.rsplit(":", 1)[-1].strip()) for line in telemetry
            if "GPU[3]" in line and "VRAM%" in line]
    if len(raw) != 18 or any(row.get("status") != "pass" for row in raw) or \
            any(row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                not 0 <= row.get("post_run_gpu_state", {}).get(
                    "vram_percent", 99) <= 2 for row in raw):
        errors.append("idle-gated bucket Pareto raw evidence changed")
    sweeps = summary.get("bucket_sweeps", [])
    if summary.get("status") != "pass" or len(sweeps) != 2 or any(
            row.get("generated_tokens_equal_across_bucket_counts") is not True or
            [point.get("bucket_count") for point in row.get("rows", [])] !=
            [1, 2, 4] or
            abs(float(row["rows"][1].get("allocated_cache_ratio", 0.0)) -
                0.625968992248062) > 1.0e-12 or
            abs(float(row["rows"][2].get("allocated_cache_ratio", 0.0)) -
                0.47093023255813954) > 1.0e-12 or
            not 0.84 < float(row["rows"][1].get(
                "tokens_per_second_ratio", 0.0)) < 0.89 or
            not 0.50 < float(row["rows"][2].get(
                "tokens_per_second_ratio", 0.0)) < 0.60
            for row in sweeps):
        errors.append("bucket Pareto aggregate evidence changed")
    if len(rejected) != 18 or len(vram) < 130 or max(vram, default=0) < 96 or \
            not any(value >= 60 for value in vram):
        errors.append("retained contaminated bucket window changed")
    if gates.get("status") != "pass_with_retained_contaminated_window" or \
            gates.get("official_matrix", {}).get("passed") != 18 or \
            gates.get("official_matrix", {}).get("post_vram_percent_max") != 2 or \
            gates.get("rejected_window", {}).get("performance_rows_accepted") != 0 or \
            gates.get("python_contract", {}).get("passed") != 13:
        errors.append("bucket Pareto gates changed")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_continuous_matrix.py").read_text(encoding="utf-8")
    if "--physical-gpu-index" not in runner or \
            "--max-idle-vram-percent" not in runner or \
            "pre_run_gpu_state" not in runner:
        errors.append("physical GPU idle gate source is missing")
    return len(raw), len(rejected), len(vram)


def validate_traffic_skew(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "116-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    rejected_monitor = [json.loads(line) for line in
                        (data / "rejected-monitor-timeout.jsonl").read_text(
                            encoding="utf-8").splitlines()]
    rejected_post = [json.loads(line) for line in
                     (data / "rejected-post-gate-selection.jsonl").read_text(
                         encoding="utf-8").splitlines()]
    if len(raw) != 36 or any(row.get("status") != "pass" for row in raw) or \
            any(not 0 <= row.get("pre_run_gpu_state", {}).get(
                    "vram_percent", 99) <= 1 or
                not 0 <= row.get("pre_run_gpu_state", {}).get(
                    "gpu_use_percent", 99) <= 2 or
                not 0 <= row.get("post_run_gpu_state", {}).get(
                    "vram_percent", 99) <= 2 or
                not 0 <= row.get("post_run_gpu_state", {}).get(
                    "gpu_use_percent", 99) <= 5 for row in raw):
        errors.append("traffic-skew raw or idle-gate evidence changed")
    comparisons = summary.get("traffic_comparisons", [])
    by_group = {}
    for row in comparisons:
        by_group.setdefault(row.get("group"), []).append(row)
    if summary.get("status") != "pass" or len(comparisons) != 6 or \
            any(row.get("token_difference", {}).get("exact") is not True
                for row in comparisons) or \
            set(by_group) != {"short_heavy", "long_heavy", "delayed"} or \
            any(len(rows) != 2 for rows in by_group.values()):
        errors.append("traffic-skew comparison contracts changed")
    if any(not 0.71 < row.get("bucketed_over_uniform_tps", 0.0) < 0.75 or
           row.get("bucketed_over_uniform_focus_ttft", 9.0) > 0.32 or
           row.get("bucketed_over_uniform_focus_ttft_p95", 0.0) < 3.1 or
           row.get("bucketed_over_uniform_focus_completion_p95", 0.0) < 2.2
           for row in by_group.get("short_heavy", [])):
        errors.append("short-heavy median/tail failure changed")
    if any(not 0.55 < row.get("bucketed_over_uniform_tps", 0.0) < 0.59 or
           row.get("bucketed_over_uniform_focus_ttft_p95", 0.0) < 2.9 or
           not 1.70 < row.get("bucketed_over_uniform_focus_completion_p95", 0.0) < 1.80
           for row in by_group.get("long_heavy", [])):
        errors.append("long-heavy tail failure changed")
    if any(not 0.93 < row.get("bucketed_over_uniform_tps", 0.0) < 0.95 or
           not 1.06 < row.get("bucketed_over_uniform_focus_ttft", 0.0) < 1.10 or
           not 1.06 < row.get("bucketed_over_uniform_focus_completion", 0.0) < 1.09
           for row in by_group.get("delayed", [])):
        errors.append("delayed-arrival regression changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight) or len(rejected_monitor) != 180 or \
            len(rejected_post) != 1:
        errors.append("traffic-skew preflight or rejected windows changed")
    if gates.get("status") != "pass_with_no_work_stealing_tail_failure" or \
            gates.get("official_matrix", {}).get("passed") != 36 or \
            gates.get("official_matrix", {}).get("token_exact_comparisons") != 6 or \
            gates.get("decision", {}).get("median_only_decision_rejected") is not True:
        errors.append("traffic-skew gates changed")
    return len(raw), len(comparisons), len(rejected_monitor)


def validate_compatible_overflow(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "117-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    rejected = [json.loads(line) for line in
                (data / "rejected-routing-raw.jsonl").read_text(
                    encoding="utf-8").splitlines()]
    rejected_preflight = [json.loads(line) for line in
                          (data / "rejected-routing-preflight.jsonl").read_text(
                              encoding="utf-8").splitlines()]
    if len(raw) != 54 or any(row.get("status") != "pass" for row in raw) or \
            any(not 0 <= row.get("pre_run_gpu_state", {}).get(
                    "vram_percent", 99) <= 1 or
                not 0 <= row.get("pre_run_gpu_state", {}).get(
                    "gpu_use_percent", 99) <= 2 or
                not 0 <= row.get("post_run_gpu_state", {}).get(
                    "vram_percent", 99) <= 3 or
                not 0 <= row.get("post_run_gpu_state", {}).get(
                    "gpu_use_percent", 99) <= 5 for row in raw):
        errors.append("compatible-overflow raw or idle-gate evidence changed")
    comparisons = summary.get("overflow_comparisons", [])
    by_group = {}
    for row in comparisons:
        by_group.setdefault(row.get("group"), []).append(row)
    if summary.get("status") != "pass" or len(comparisons) != 6 or \
            any(row.get("token_difference_vs_fixed", {}).get("exact") is not True or
                row.get("token_difference_vs_uniform", {}).get("exact") is not True
                for row in comparisons) or \
            set(by_group) != {"short_heavy", "long_heavy", "delayed"} or \
            any(len(rows) != 2 for rows in by_group.values()):
        errors.append("compatible-overflow comparison contracts changed")
    expected_short_route = [0, 0, 0, 0, 1, 1, 1, 1]
    if any(row.get("overflow_routes") != expected_short_route or
           row.get("overflow_routed_requests") != 2 or
           not 1.12 < row.get("overflow_over_fixed_tps", 0.0) < 1.14 or
           not 0.37 < row.get("overflow_over_fixed_focus_ttft_p95", 9.0) < 0.40 or
           not 0.59 < row.get("overflow_over_fixed_focus_completion_p95", 9.0) < 0.62 or
           not 0.81 < row.get("overflow_over_uniform_tps", 0.0) < 0.84 or
           not 1.20 < row.get("overflow_over_uniform_focus_ttft_p95", 0.0) < 1.25
           for row in by_group.get("short_heavy", [])):
        errors.append("short-heavy compatible-overflow recovery changed")
    for group in ("long_heavy", "delayed"):
        if any(row.get("overflow_routed_requests") != 0 or
               not 0.98 < row.get("overflow_over_fixed_tps", 0.0) < 1.02 or
               not 0.98 < row.get("overflow_over_fixed_focus_ttft_p95", 0.0) < 1.02 or
               not 0.98 < row.get("overflow_over_fixed_focus_completion_p95", 0.0) < 1.02
               for row in by_group.get(group, [])):
            errors.append(f"{group} overflow no-op boundary changed")
    if len(preflight) != 3 or len(rejected) != 6 or \
            len(rejected_preflight) != 3:
        errors.append("compatible-overflow preflight or rejected route evidence changed")
    if gates.get("status") != "keep_optional_compatible_overflow" or \
            gates.get("full_release", {}).get("passed") != 319 or \
            gates.get("sanitizer", {}).get("passed") != 215 or \
            gates.get("official_matrix", {}).get("passed") != 54 or \
            gates.get("decision", {}).get("default_enabled") is not False:
        errors.append("compatible-overflow final gates changed")
    source = (REPOSITORY / "src" / "inference" / "scheduler.cpp").read_text(
        encoding="utf-8")
    tests = (REPOSITORY / "tests" / "inference" /
             "scheduler_test.cpp").read_text(encoding="utf-8")
    if "const auto load = schedulers[index].active_request_count();" not in source or \
            "CompatibleOverflowBorrowsLargerCapacityWithoutMovingRequests" not in tests:
        errors.append("compatible-overflow pending-count fix or threshold test missing")
    return len(raw), len(comparisons), len(rejected)


def validate_slot_ratio_sweep(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "118-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    if len(raw) != 48 or any(row.get("status") != "pass" for row in raw) or \
            any(not 0 <= row.get("pre_run_gpu_state", {}).get(
                    "vram_percent", 99) <= 1 or
                not 0 <= row.get("pre_run_gpu_state", {}).get(
                    "gpu_use_percent", 99) <= 2 or
                not 0 <= row.get("post_run_gpu_state", {}).get(
                    "vram_percent", 99) <= 3 or
                not 0 <= row.get("post_run_gpu_state", {}).get(
                    "gpu_use_percent", 99) <= 6 for row in raw):
        errors.append("slot-ratio raw or idle-gate evidence changed")
    sweeps = summary.get("slot_ratio_sweeps", [])
    by_group = {}
    for row in sweeps:
        by_group.setdefault(row.get("group"), []).append(row)
    if summary.get("status") != "pass" or len(sweeps) != 4 or \
            any(row.get("generated_tokens_equal_across_ratios") is not True or
                [(point.get("small_slots"), point.get("large_slots"))
                 for point in row.get("rows", [])] != [(2, 6), (4, 4), (6, 2)]
                for row in sweeps) or \
            set(by_group) != {"short_heavy", "long_heavy"} or \
            any(len(rows) != 2 for rows in by_group.values()):
        errors.append("slot-ratio sweep contracts changed")
    if any(not 0.83 < row["rows"][2].get(
                "throughput_ratio_vs_uniform", 0.0) < 0.87 or
           not 0.39 < row["rows"][2].get(
                "focus_ttft_p95_ratio_vs_uniform", 9.0) < 0.42 or
           not 1.27 < row["rows"][2].get(
                "focus_completion_p95_ratio_vs_uniform", 0.0) < 1.32 or
           row["rows"][0].get("focus_ttft_p95_ratio_vs_uniform", 0.0) < 4.9
           for row in by_group.get("short_heavy", [])):
        errors.append("short-heavy 6:2 optimum changed")
    if any(not 0.86 < row["rows"][0].get(
                "throughput_ratio_vs_uniform", 0.0) < 0.89 or
           not 1.05 < row["rows"][0].get(
                "focus_ttft_p95_ratio_vs_uniform", 0.0) < 1.08 or
           not 1.14 < row["rows"][0].get(
                "focus_completion_p95_ratio_vs_uniform", 0.0) < 1.17 or
           row["rows"][2].get("focus_ttft_p95_ratio_vs_uniform", 0.0) < 4.0
           for row in by_group.get("long_heavy", [])):
        errors.append("long-heavy 2:6 optimum changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("slot-ratio preflight changed")
    if gates.get("status") != "keep_explicit_workload_matched_slot_ratios" or \
            gates.get("official_matrix", {}).get("passed") != 48 or \
            gates.get("official_matrix", {}).get("token_exact_sweeps") != 4 or \
            gates.get("python_contract", {}).get("passed") != 16:
        errors.append("slot-ratio final gates changed")
    return len(raw), len(sweeps), len(preflight)


def validate_mi300_precision_roofline(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "119-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    sizes = {row.get("size") for row in raw}
    dtypes = {row.get("dtype") for row in raw}
    expected_dtypes = {"fp32_readable", "fp32", "fp16", "bf16",
                       "fp8_e4m3_fnuz"}
    if len(raw) != 20 or sizes != {128, 256, 512, 1024} or \
            dtypes != expected_dtypes or any(
                row.get("accuracy_passed") is not True or
                row.get("achieved_tflops", 0.0) <= 0.0 or
                not 0.0 < row.get("official_peak_utilization", 0.0) < 0.1 or
                not 0.0 < row.get("roofline_utilization", 0.0) < 0.1 or
                row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
                row.get("post_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
                for row in raw):
        errors.append("MI300 precision roofline raw evidence changed")
    by_size_dtype = {(row["size"], row["dtype"]): row for row in raw}
    fp8_ratios = {
        size: by_size_dtype[size, "fp32"]["median_ms"] /
              by_size_dtype[size, "fp8_e4m3_fnuz"]["median_ms"]
        for size in sizes
    }
    if any(fp8_ratios[size] >= 1.0 for size in (128, 256, 512)) or \
            not 1.05 < fp8_ratios[1024] < 1.15:
        errors.append("FP8 non-universal speedup evidence changed")
    best = summary.get("by_dtype", {})
    if summary.get("status") != "pass" or set(best) != expected_dtypes or \
            best.get("fp16", {}).get("best_size") != 1024 or \
            not 18.0 < best.get("fp16", {}).get(
                "best_achieved_tflops", 0.0) < 19.5 or \
            not 13.0 < best.get("fp8_e4m3_fnuz", {}).get(
                "best_achieved_tflops", 0.0) < 14.5 or \
            best.get("fp8_e4m3_fnuz", {}).get(
                "best_official_peak_utilization", 1.0) >= 0.006:
        errors.append("MI300 precision roofline summary changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("MI300 precision roofline preflight changed")
    if gates.get("status") != "keep_executed_precision_roofline" or \
            gates.get("formal_matrix", {}).get("passed") != 20 or \
            gates.get("decision", {}).get("fp8_universal_speedup") is not False or \
            gates.get("decision", {}).get("int8_executed") is not False:
        errors.append("MI300 precision roofline gates changed")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "mi300_precision_roofline.py").read_text(encoding="utf-8")
    if '"fp8_e4m3_fnuz": 2614.9' not in runner or \
            "INT8/INT4 are not executed" not in runner:
        errors.append("MI300 roofline peak or execution boundary is missing")
    return len(raw), len(sizes), len(dtypes)


def validate_large_precision_roofline(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "120-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    sizes = {row.get("size") for row in raw}
    dtypes = {row.get("dtype") for row in raw}
    expected_dtypes = {"fp32_readable", "fp32", "fp16", "bf16",
                       "fp8_e4m3_fnuz"}
    if len(raw) != 10 or sizes != {2048, 4096} or dtypes != expected_dtypes or \
            any(row.get("reference") != "fp32" or
                row.get("accuracy_passed") is not True or
                row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
                row.get("post_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 3
                for row in raw):
        errors.append("large precision roofline raw/reference evidence changed")
    by_key = {(row["size"], row["dtype"]): row for row in raw}
    if not 1.65 < by_key[2048, "fp8_e4m3_fnuz"]["achieved_tflops"] / \
            by_key[2048, "fp32"]["achieved_tflops"] < 1.80 or \
            not 4.20 < by_key[4096, "fp8_e4m3_fnuz"]["achieved_tflops"] / \
            by_key[4096, "fp32"]["achieved_tflops"] < 4.40 or \
            not 1.35 < by_key[4096, "fp8_e4m3_fnuz"]["achieved_tflops"] / \
            by_key[4096, "fp16"]["achieved_tflops"] < 1.50 or \
            not 470.0 < by_key[4096, "fp8_e4m3_fnuz"][
                "achieved_tflops"] < 485.0 or \
            not 0.17 < by_key[4096, "fp8_e4m3_fnuz"][
                "official_peak_utilization"] < 0.19 or \
            not 0.65 < by_key[4096, "fp32"][
                "official_peak_utilization"] < 0.70:
        errors.append("large precision roofline speedup/utilization changed")
    if summary.get("status") != "pass" or summary.get("reference") != "fp32" or \
            summary.get("sizes") != [2048, 4096]:
        errors.append("large precision roofline summary changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("large precision roofline preflight changed")
    if gates.get("status") != \
            "keep_large_shape_fp8_speedup_with_reference_boundary" or \
            gates.get("formal_matrix", {}).get("passed") != 10 or \
            gates.get("decision", {}).get("independent_large_fp32_reference") is not False:
        errors.append("large precision roofline gates changed")
    worker = (REPOSITORY / "benchmarks" / "micro" /
              "benchmark_precision.cpp").read_text(encoding="utf-8")
    if 'result.reference = argv[index + 1]' not in worker or \
            'result.reference != "cpu" && result.reference != "fp32"' not in worker:
        errors.append("large precision explicit reference mode is missing")
    return len(raw), len(sizes), len(dtypes)


def validate_int8_executed_probe(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "121-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    sizes = {row.get("size") for row in raw}
    if len(raw) != 6 or sizes != {128, 256, 512, 1024, 2048, 4096} or any(
            row.get("op") != "int8_matmul" or
            row.get("input_dtype") != "int8" or
            row.get("output_dtype") != "int32" or
            row.get("sample_count") != 5 or
            row.get("maximum_sample_error") != 0 or
            row.get("accuracy_passed") is not True or
            row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
            row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
            row.get("post_run_gpu_state", {}).get("vram_percent") != 0 or
            row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
            for row in raw):
        errors.append("INT8 executed probe raw/exact evidence changed")
    by_size = {row["size"]: row for row in raw}
    if not 290.0 < by_size[2048]["achieved_tops"] < 310.0 or \
            not 405.0 < by_size[4096]["achieved_tops"] < 425.0 or \
            not 0.15 < by_size[4096]["official_peak_utilization"] < 0.17:
        errors.append("INT8 executed probe TOPS/utilization changed")
    if summary.get("status") != "pass" or \
            summary.get("best", {}).get("size") != 4096 or \
            "no public Tensor dtype" not in summary.get("boundary", ""):
        errors.append("INT8 executed probe summary/boundary changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("INT8 executed probe preflight changed")
    if gates.get("status") != "keep_raw_int8_executed_probe" or \
            gates.get("formal_matrix", {}).get("passed") != 6 or \
            gates.get("decision", {}).get("raw_int8_kernel_executed") is not True or \
            gates.get("decision", {}).get("public_tensor_int8") is not False:
        errors.append("INT8 executed probe gates changed")
    worker = (REPOSITORY / "benchmarks" / "micro" /
              "benchmark_int8.cpp").read_text(encoding="utf-8")
    if "HIPBLAS_COMPUTE_32I" not in worker or "HIP_R_8I" not in worker or \
            "maximum_sample_error" not in worker:
        errors.append("INT8 raw worker execution or exact-sample gate missing")
    return len(raw), sum(row["sample_count"] for row in raw), 1


def validate_official_fp8_static_scale(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "122-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    rejected = [json.loads(line) for line in
                (data / "rejected-worker-raw.jsonl").read_text(
                    encoding="utf-8").splitlines()]
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    if len(raw) != 36 or any(row.get("status") != "pass" for row in raw) or \
            any(row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
                row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
                row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
                for row in raw):
        errors.append("official FP8 raw or idle-gate evidence changed")
    aggregates = summary.get("aggregates", [])
    fp8_rows = [row for row in aggregates if row.get("policy") == "fp8"]
    bf16_rows = [row for row in aggregates if row.get("policy") == "bf16"]
    if summary.get("status") != "complete_with_recorded_accuracy_failures" or \
            summary.get("execution_status") != "pass" or len(aggregates) != 12 or \
            len(fp8_rows) != 4 or len(bf16_rows) != 4 or \
            summary.get("accuracy_failure_count") != 4 or any(
                row.get("successful_runs") != 3 for row in aggregates) or any(
                row.get("precision_gate_passed_all") is not False or
                row.get("maximum_absolute_error_max", 0.0) < 11.0 or
                row.get("root_mean_square_error_max", 0.0) < 2.0
                for row in fp8_rows) or any(
                row.get("precision_gate_passed_all") is not True
                for row in bf16_rows):
        errors.append("official FP8 aggregate precision evidence changed")
    by_key = {(row["model"], row["context"], row["policy"]): row
              for row in aggregates}
    qwen8 = by_key["qwen2.5-0.5b", 8, "fp8"]
    qwen512 = by_key["qwen2.5-0.5b", 512, "fp8"]
    deep8 = by_key["deepseek-r1-distill-qwen-1.5b", 8, "fp8"]
    deep512 = by_key["deepseek-r1-distill-qwen-1.5b", 512, "fp8"]
    if qwen8.get("top_token_equal_all") is not True or \
            qwen512.get("top_token_equal_all") is not False or \
            deep8.get("fp8_software_fallback_shapes_max") != 1 or \
            deep8.get("fp8_software_fallback_calls_p50") != 112 or \
            deep512.get("fp8_software_fallback_shapes_max") != 0 or \
            deep512.get("fp8_native_shapes_max") != 5:
        errors.append("official FP8 top-token or fallback evidence changed")
    if not 0.40 < qwen8["resident_weight_bytes"] / \
            by_key["qwen2.5-0.5b", 8, "fp32"]["resident_weight_bytes"] < 0.50 or \
            not 0.30 < deep8["resident_weight_bytes"] / \
            by_key["deepseek-r1-distill-qwen-1.5b", 8, "fp32"]["resident_weight_bytes"] < 0.40:
        errors.append("official FP8 residency reduction changed")
    if len(rejected) != 18 or len(preflight) != 3:
        errors.append("official FP8 rejected worker or preflight evidence changed")
    if gates.get("status") != \
            "keep_fp8_infrastructure_reject_static_scale_model_policy" or \
            gates.get("full_release", {}).get("passed") != 326 or \
            gates.get("sanitizer", {}).get("passed") != 219 or \
            gates.get("decision", {}).get("static_global_scale_policy_accepted") is not False:
        errors.append("official FP8 final gates changed")
    model_source = (REPOSITORY / "src" / "model" / "model.cpp").read_text(
        encoding="utf-8")
    op_source = (REPOSITORY / "src" / "ops" / "optimized.cpp").read_text(
        encoding="utf-8")
    if "prepare_fp8_inference_weights" not in model_source or \
            "fp8_native_matrix_registry" not in op_source or \
            "fp8_software_fallback_calls" not in op_source:
        errors.append("official FP8 weight cache or fallback source missing")
    return len(raw), len(fp8_rows), len(rejected)


def validate_fp8_global_scale_grid(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "123-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    candidates = [row for row in raw if row.get("policy") == "fp8"]
    references = [row for row in raw if row.get("policy") == "fp32"]
    if len(raw) != 34 or len(candidates) != 32 or len(references) != 2 or \
            any(row.get("status") != "pass" for row in raw) or any(
                row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
                row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
                row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 5
                for row in raw):
        errors.append("FP8 global scale raw or idle-gate evidence changed")
    if summary.get("status") != "complete_no_passing_scale" or \
            summary.get("execution_status") != "pass" or \
            summary.get("candidate_count") != 32 or \
            summary.get("precision_gate_pass_count") != 0 or \
            summary.get("activation_scales") != [0.00625, 0.0125, 0.025, 0.05] or \
            summary.get("weight_scales") != [0.00125, 0.0025, 0.005, 0.01]:
        errors.append("FP8 global scale summary contract changed")
    by_model = summary.get("by_model", {})
    qwen = by_model.get("qwen2.5-0.5b", {})
    deep = by_model.get("deepseek-r1-distill-qwen-1.5b", {})
    qwen_best = qwen.get("best_candidate", {})
    deep_best = deep.get("best_candidate", {})
    if qwen.get("precision_gate_pass_count") != 0 or \
            qwen.get("top_token_equal_count") != 10 or \
            qwen_best.get("fp8_activation_scale") != 0.05 or \
            qwen_best.get("fp8_weight_scale") != 0.0025 or \
            not 1.9 < qwen_best.get("root_mean_square_error", 0.0) < 2.0 or \
            deep.get("precision_gate_pass_count") != 0 or \
            deep.get("top_token_equal_count") != 4 or \
            deep_best.get("fp8_activation_scale") != 0.025 or \
            deep_best.get("fp8_weight_scale") != 0.005 or \
            not 2.5 < deep_best.get("root_mean_square_error", 0.0) < 2.6 or \
            deep_best.get("fp8_software_fallback_calls") != 112:
        errors.append("FP8 global scale best-candidate evidence changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("FP8 global scale preflight changed")
    if gates.get("status") != "reject_current_global_scale_grid_expand_boundary" or \
            gates.get("cpu_regression", {}).get("passed") != 225 or \
            gates.get("official_scale_grid", {}).get("precision_gate_passed") != 0 or \
            gates.get("decision", {}).get("all_global_scales_refuted") is not False or \
            gates.get("decision", {}).get("expand_activation_boundary") is not True:
        errors.append("FP8 global scale gates changed")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_fp8_scale_grid.py").read_text(encoding="utf-8")
    if "complete last-token vocabulary logits" not in runner or \
            "the grid is fixed before execution" not in runner:
        errors.append("FP8 global scale runner boundary missing")
    return len(raw), len(candidates), sum(
        row.get("precision_gate_passed", False) for row in candidates)


def validate_fp8_scale_boundary(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "124-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    candidates = [row for row in raw if row.get("policy") == "fp8"]
    if len(raw) != 18 or len(candidates) != 16 or \
            any(row.get("status") != "pass" for row in raw) or any(
                row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
                row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
                row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
                for row in raw):
        errors.append("FP8 boundary raw or idle-gate evidence changed")
    if summary.get("status") != "complete_no_passing_scale" or \
            summary.get("candidate_count") != 16 or \
            summary.get("precision_gate_pass_count") != 0 or \
            summary.get("activation_scales") != [0.1, 0.2] or \
            summary.get("weight_scales") != [0.00125, 0.0025, 0.005, 0.01]:
        errors.append("FP8 boundary summary contract changed")
    by_model = summary.get("by_model", {})
    qwen = by_model.get("qwen2.5-0.5b", {})
    deep = by_model.get("deepseek-r1-distill-qwen-1.5b", {})
    qwen_best = qwen.get("best_candidate", {})
    deep_best = deep.get("best_candidate", {})
    if qwen.get("top_token_equal_count") != 8 or \
            qwen_best.get("fp8_activation_scale") != 0.2 or \
            qwen_best.get("fp8_weight_scale") != 0.0025 or \
            not 0.65 < qwen_best.get("root_mean_square_error", 0.0) < 0.70 or \
            deep.get("top_token_equal_count") != 6 or \
            deep_best.get("fp8_activation_scale") != 0.2 or \
            deep_best.get("fp8_weight_scale") != 0.005 or \
            not 1.15 < deep_best.get("root_mean_square_error", 0.0) < 1.20 or \
            deep_best.get("fp8_software_fallback_calls") != 112:
        errors.append("FP8 boundary best-candidate evidence changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("FP8 boundary preflight changed")
    if gates.get("status") != "reject_second_boundary_keep_expanding" or \
            gates.get("official_boundary_grid", {}).get("workers_passed") != 18 or \
            gates.get("official_boundary_grid", {}).get("precision_gate_passed") != 0 or \
            gates.get("decision", {}).get("all_global_scales_refuted") is not False or \
            gates.get("decision", {}).get("both_models_best_on_activation_upper_boundary") \
            is not True:
        errors.append("FP8 boundary gates changed")
    return len(raw), len(candidates), sum(
        row.get("precision_gate_passed", False) for row in candidates)


def validate_fp8_scale_turn(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "125-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    candidates = [row for row in raw if row.get("policy") == "fp8"]
    if len(raw) != 18 or len(candidates) != 16 or \
            any(row.get("status") != "pass" for row in raw) or any(
                row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
                row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
                row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
                for row in raw):
        errors.append("FP8 scale turn raw or idle-gate evidence changed")
    if summary.get("status") != "complete_no_passing_scale" or \
            summary.get("candidate_count") != 16 or \
            summary.get("precision_gate_pass_count") != 0 or \
            summary.get("activation_scales") != [0.4, 0.8]:
        errors.append("FP8 scale turn summary contract changed")
    by_model = summary.get("by_model", {})
    qwen = by_model.get("qwen2.5-0.5b", {})
    deep = by_model.get("deepseek-r1-distill-qwen-1.5b", {})
    qwen_best = qwen.get("best_candidate", {})
    deep_best = deep.get("best_candidate", {})
    if qwen.get("top_token_equal_count") != 8 or \
            qwen_best.get("fp8_activation_scale") != 0.8 or \
            qwen_best.get("fp8_weight_scale") != 0.005 or \
            not 0.30 < qwen_best.get("root_mean_square_error", 0.0) < 0.31 or \
            deep.get("top_token_equal_count") != 5 or \
            deep_best.get("fp8_activation_scale") != 0.4 or \
            deep_best.get("fp8_weight_scale") != 0.00125 or \
            not 1.22 < deep_best.get("root_mean_square_error", 0.0) < 1.25 or \
            deep_best.get("fp8_software_fallback_calls") != 112:
        errors.append("FP8 scale turn best-candidate evidence changed")
    deep_low_rms_wrong_top = [row for row in candidates
                              if row.get("model") ==
                              "deepseek-r1-distill-qwen-1.5b" and
                              row.get("fp8_activation_scale") == 0.8 and
                              row.get("fp8_weight_scale") == 0.01]
    if len(deep_low_rms_wrong_top) != 1 or \
            deep_low_rms_wrong_top[0].get("top_token_equal") is not False or \
            not 0.62 < deep_low_rms_wrong_top[0].get(
                "root_mean_square_error", 0.0) < 0.64:
        errors.append("FP8 scale turn top-token counterexample changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("FP8 scale turn preflight changed")
    if gates.get("status") != "deepseek_turn_found_qwen_boundary_open" or \
            gates.get("decision", {}).get("deepseek_top_equal_turn_found") is not True or \
            gates.get("decision", {}).get("qwen_turn_found") is not False or \
            gates.get("decision", {}).get("continue_qwen_only_once") is not True:
        errors.append("FP8 scale turn gates changed")
    return len(raw), len(candidates), sum(
        row.get("precision_gate_passed", False) for row in candidates)


def validate_qwen_fp8_scale_closure(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "126-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    candidates = [row for row in raw if row.get("policy") == "fp8"]
    if len(raw) != 9 or len(candidates) != 8 or \
            any(row.get("status") != "pass" for row in raw) or any(
                row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
                row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
                row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 0 or
                row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 3
                for row in raw):
        errors.append("Qwen FP8 closure raw or idle-gate evidence changed")
    qwen = summary.get("by_model", {}).get("qwen2.5-0.5b", {})
    best = qwen.get("best_candidate", {})
    if summary.get("status") != "complete_no_passing_scale" or \
            summary.get("candidate_count") != 8 or \
            summary.get("precision_gate_pass_count") != 0 or \
            summary.get("activation_scales") != [1.6, 3.2] or \
            qwen.get("top_token_equal_count") != 8 or \
            best.get("fp8_activation_scale") != 3.2 or \
            best.get("fp8_weight_scale") != 0.0025 or \
            not 0.21 < best.get("root_mean_square_error", 0.0) < 0.22 or \
            not 1.0 < best.get("maximum_absolute_error", 0.0) < 1.01:
        errors.append("Qwen FP8 closure summary evidence changed")
    if len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("Qwen FP8 closure preflight changed")
    if gates.get("status") != "stop_cross_model_global_search_start_tensor_amax" or \
            gates.get("decision", {}).get("qwen_literal_turn_found") is not False or \
            gates.get("decision", {}).get(
                "all_global_scales_mathematically_refuted") is not False or \
            gates.get("decision", {}).get("cross_model_global_search_stopped") is not True or \
            gates.get("decision", {}).get("implement_weight_tensor_amax") is not True:
        errors.append("Qwen FP8 closure decision boundary changed")
    return len(raw), len(candidates), sum(
        row.get("precision_gate_passed", False) for row in candidates)


def validate_fp8_tensor_amax_weight(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "127-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    rejected = [json.loads(line) for line in
                (data / "rejected-raw.jsonl").read_text(
                    encoding="utf-8").splitlines()]
    pilot = [json.loads(line) for line in
             (data / "pilot-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    if len(raw) != 36 or any(row.get("status") != "pass" for row in raw) or any(
            row.get("pre_run_gpu_state", {}).get("vram_percent", 99) > 1 or
            row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
            row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 3 or
            row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 5
            for row in raw):
        errors.append("FP8 tensor-amax raw or idle-gate evidence changed")
    aggregates = summary.get("aggregates", [])
    fp8_rows = [row for row in aggregates if row.get("policy") == "fp8"]
    if summary.get("status") != "complete_with_recorded_accuracy_failures" or \
            summary.get("fp8_weight_scale_mode") != "tensor-amax" or \
            summary.get("accuracy_failure_count") != 4 or len(aggregates) != 12 or \
            len(fp8_rows) != 4 or any(
                row.get("precision_gate_passed_all") is not False or
                row.get("top_token_equal_all") is not True or
                row.get("weight_preparation_ms_p50", 0.0) < 2800.0
                for row in fp8_rows) or \
            "per-Tensor weight amax" not in summary.get("boundary", "") or \
            "fixed global activation scale" not in summary.get("boundary", ""):
        errors.append("FP8 tensor-amax aggregate or boundary evidence changed")
    by_key = {(row["model"], row["context"]): row for row in fp8_rows}
    qwen8 = by_key["qwen2.5-0.5b", 8]
    qwen512 = by_key["qwen2.5-0.5b", 512]
    deep8 = by_key["deepseek-r1-distill-qwen-1.5b", 8]
    deep512 = by_key["deepseek-r1-distill-qwen-1.5b", 512]
    if not 0.65 < qwen8["root_mean_square_error_max"] < 0.68 or \
            not 1.28 < qwen512["root_mean_square_error_max"] < 1.31 or \
            not 1.16 < deep8["root_mean_square_error_max"] < 1.19 or \
            not 1.30 < deep512["root_mean_square_error_max"] < 1.32 or \
            deep8.get("fp8_software_fallback_calls_p50") != 112 or \
            deep512.get("fp8_software_fallback_calls_p50") != 0:
        errors.append("FP8 tensor-amax precision or fallback evidence changed")
    fp8_raw = [row for row in raw if row.get("policy") == "fp8"]
    qwen_raw = next(row for row in fp8_raw if row["model"] == "qwen2.5-0.5b")
    deep_raw = next(row for row in fp8_raw if row["model"].startswith("deepseek"))
    if qwen_raw.get("fp8_weight_bytes_scanned") != 1431306240 or \
            not 0.00034 < qwen_raw.get("fp8_weight_scale_min", 0.0) < 0.00036 or \
            not 0.0069 < qwen_raw.get("fp8_weight_scale_max", 0.0) < 0.0070 or \
            deep_raw.get("fp8_weight_bytes_scanned") != 6174277632 or \
            not 0.00079 < deep_raw.get("fp8_weight_scale_min", 0.0) < 0.00081 or \
            not 0.0075 < deep_raw.get("fp8_weight_scale_max", 0.0) < 0.0076:
        errors.append("FP8 tensor-amax scale-range or scan evidence changed")
    if len(rejected) != 15 or len(pilot) != 3 or len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("FP8 tensor-amax rejected/pilot/preflight evidence changed")
    if gates.get("status") != \
            "keep_tensor_amax_infrastructure_reject_model_policy" or \
            gates.get("full_release", {}).get("passed") != 331 or \
            gates.get("decision", {}).get("tensor_amax_api_retained") is not True or \
            gates.get("decision", {}).get("tensor_amax_model_policy_accepted") is not False or \
            gates.get("decision", {}).get("activation_scale_requires_new_policy") is not True:
        errors.append("FP8 tensor-amax gates changed")
    model_source = (REPOSITORY / "src" / "model" / "model.cpp").read_text(
        encoding="utf-8")
    if "tensor_amax_scale" not in model_source or \
            "FP8 tensor-amax weight preparation requires finite values" not in model_source:
        errors.append("FP8 tensor-amax implementation boundary missing")
    return len(raw), len(fp8_rows), len(rejected)


def validate_fp8_activation_range(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "128-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    workers = [json.loads(line) for line in
               (data / "workers.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    pilot = [json.loads(line) for line in
             (data / "pilot-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected_trace = [json.loads(line) for line in
                      (data / "rejected-qwen-trace.jsonl").read_text(
                          encoding="utf-8").splitlines()]
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    qwen = [row for row in raw if row.get("model") == "qwen2.5-0.5b"]
    deep = [row for row in raw if row.get("model") ==
            "deepseek-r1-distill-qwen-1.5b"]
    if len(raw) != 208 or len(qwen) != 96 or len(deep) != 112 or any(
            row.get("status") != "pass" or row.get("dtype") != "float32" or
            row.get("representable_magnitude") != 48.0
            for row in raw):
        errors.append("FP8 activation-range raw contract changed")
    if len(workers) != 2 or any(
            row.get("status") != "pass" or
            row.get("selected_boundaries") not in (96, 112) or
            row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
            row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
            row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
            row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 3
            for row in workers):
        errors.append("FP8 activation-range worker or idle-gate evidence changed")
    aggregates = summary.get("aggregates", [])
    if summary.get("status") != "pass" or summary.get("rows") != 208 or \
            summary.get("potential_saturation_rows") != 16 or \
            summary.get("representable_magnitude") != 48.0 or \
            len(aggregates) != 8 or \
            "not performance evidence" not in summary.get("boundary", ""):
        errors.append("FP8 activation-range summary changed")
    by_key = {(row["model"], row["boundary"]): row for row in aggregates}
    qwen_activated = by_key["qwen2.5-0.5b", "ffn.activated"]
    deep_activated = by_key[
        "deepseek-r1-distill-qwen-1.5b", "ffn.activated"]
    if qwen_activated.get("potential_saturation_layers") != 4 or \
            qwen_activated.get("maximum_layer") != 21 or \
            not 35.0 < qwen_activated.get("range_ratio_max", 0.0) < 37.0 or \
            deep_activated.get("potential_saturation_layers") != 5 or \
            deep_activated.get("maximum_layer") != 2 or \
            not 63.0 < deep_activated.get("range_ratio_max", 0.0) < 65.0:
        errors.append("FP8 activation-range FFN outlier evidence changed")
    if any(by_key[model, boundary].get("potential_saturation_layers") != 0
           for model in ("qwen2.5-0.5b",
                         "deepseek-r1-distill-qwen-1.5b")
           for boundary in ("attention_norm", "attention.context")):
        errors.append("FP8 activation-range Attention boundary changed")
    if len(pilot) != 96 or len(rejected_trace) != 317 or len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("FP8 activation-range pilot/rejected/preflight evidence changed")
    if gates.get("status") != \
            "activation_global_scale_refuted_design_device_tensor_scale" or \
            gates.get("rejected_trace_attempt", {}).get(
                "missing_fp32_ffn_activated") != 24 or \
            gates.get("decision", {}).get(
                "fixed_global_activation_scale_accepted") is not False or \
            gates.get("decision", {}).get(
                "per_linear_input_tensor_scale_supported_by_evidence") is not True or \
            gates.get("decision", {}).get(
                "per_row_token_scale_required_yet") is not False:
        errors.append("FP8 activation-range gates changed")
    runner = (REPOSITORY / "benchmarks" / "single_gpu" /
              "hf_activation_range.py").read_text(encoding="utf-8")
    if "all-layer trace is missing a Linear input boundary" not in runner or \
            "synchronous diagnostic trace, not performance evidence" not in runner:
        errors.append("FP8 activation-range runner boundary missing")
    return len(raw), summary.get("potential_saturation_rows", 0), len(rejected_trace)


def validate_fp8_device_activation_amax(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "129-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    pilot = [json.loads(line) for line in
             (data / "pilot-raw.jsonl").read_text(encoding="utf-8").splitlines()]
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    preflight = [json.loads(line) for line in
                 (data / "gpu2-preflight.jsonl").read_text(
                     encoding="utf-8").splitlines()]
    if len(raw) != 36 or any(row.get("status") != "pass" for row in raw) or any(
            row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
            row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
            row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
            row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
            for row in raw):
        errors.append("FP8 device activation amax raw or idle-gate evidence changed")
    aggregates = summary.get("aggregates", [])
    fp8_rows = [row for row in aggregates if row.get("policy") == "fp8"]
    if summary.get("status") != "complete_with_recorded_accuracy_failures" or \
            summary.get("accuracy_failure_count") != 4 or len(aggregates) != 12 or \
            len(fp8_rows) != 4 or \
            summary.get("fp8_weight_scale_mode") != "tensor-amax" or \
            summary.get("fp8_activation_scale_mode") != "tensor-amax" or \
            "device per-input-Tensor activation amax" not in \
            summary.get("boundary", "") or any(
                row.get("precision_gate_passed_all") is not False or
                row.get("top_token_equal_all") is not True
                for row in fp8_rows):
        errors.append("FP8 device activation amax aggregate contract changed")
    by_key = {(row["model"], row["context"]): row for row in fp8_rows}
    expected = {
        ("qwen2.5-0.5b", 8): (0.18, 0.20, 1400.0, 1600.0),
        ("qwen2.5-0.5b", 512): (0.28, 0.31, 4700.0, 5100.0),
        ("deepseek-r1-distill-qwen-1.5b", 8): (0.42, 0.45, 800.0, 900.0),
        ("deepseek-r1-distill-qwen-1.5b", 512): (0.24, 0.26, 2100.0, 2250.0),
    }
    for key, (rms_low, rms_high, tps_low, tps_high) in expected.items():
        row = by_key[key]
        if not rms_low < row.get("root_mean_square_error_max", 0.0) < rms_high or \
                not tps_low < row.get("prefill_tokens_per_second_p50", 0.0) < tps_high:
            errors.append(f"FP8 device activation amax result changed for {key}")
    all_aggregates = {(row["model"], row["context"], row["policy"]): row
                      for row in aggregates}
    if by_key["qwen2.5-0.5b", 512]["prefill_tokens_per_second_p50"] / \
            all_aggregates["qwen2.5-0.5b", 512, "bf16"][
                "prefill_tokens_per_second_p50"] > 0.06 or \
            by_key["deepseek-r1-distill-qwen-1.5b", 512][
                "prefill_tokens_per_second_p50"] / \
            all_aggregates["deepseek-r1-distill-qwen-1.5b", 512, "bf16"][
                "prefill_tokens_per_second_p50"] > 0.05:
        errors.append("FP8 single-block long-context failure changed")
    if by_key["deepseek-r1-distill-qwen-1.5b", 8].get(
            "fp8_software_fallback_calls_p50") != 112 or \
            by_key["deepseek-r1-distill-qwen-1.5b", 512].get(
                "fp8_software_fallback_calls_p50") != 0:
        errors.append("FP8 device-scale fallback evidence changed")
    if len(pilot) != 3 or len(preflight) != 3 or any(
            row.get("card2", {}).get("GPU use (%)") != "0" or
            row.get("card2", {}).get("GPU Memory Allocated (VRAM%)") != "0"
            for row in preflight):
        errors.append("FP8 device activation pilot/preflight changed")
    if gates.get("status") != \
            "keep_device_scale_infrastructure_reject_current_model_policy" or \
            gates.get("full_release", {}).get("passed") != 336 or \
            gates.get("decision", {}).get("device_scale_contract_retained") is not True or \
            gates.get("decision", {}).get(
                "current_tensor_amax_model_policy_accepted") is not False or \
            gates.get("decision", {}).get("single_block_reduction_accepted") is not False:
        errors.append("FP8 device activation amax gates changed")
    op_source = (REPOSITORY / "src" / "ops" / "ops.cpp").read_text(
        encoding="utf-8")
    kernel_source = (REPOSITORY / "src" / "ops" / "hip" /
                     "basic_kernels.hip").read_text(encoding="utf-8")
    if "quantize_fp8_dynamic" not in op_source or \
            "host_scale_available" not in op_source or \
            "fp8_tensor_scale_kernel" not in kernel_source or \
            "dequantize_fp8_device_scale_kernel" not in kernel_source:
        errors.append("FP8 device activation amax implementation missing")
    return len(raw), len(fp8_rows), len(pilot)


def validate_fp8_activation_row_range(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "130-data"
    raw = [json.loads(line) for line in
           (data / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    workers = [json.loads(line) for line in
               (data / "workers.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "trace-manifest.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    qwen = [row for row in raw if row.get("model") == "qwen2.5-0.5b"]
    deep = [row for row in raw if row.get("model") ==
            "deepseek-r1-distill-qwen-1.5b"]
    if len(raw) != 208 or len(qwen) != 96 or len(deep) != 112 or any(
            row.get("status") != "pass" or row.get("rows") != 8 or
            len(row.get("row_amax", [])) != 8 or
            row.get("row_amax_max") != row.get("tensor_amax")
            for row in raw):
        errors.append("FP8 activation row-range raw contract changed")
    if len(workers) != 2 or any(
            row.get("selected_boundaries") not in (96, 112) or
            row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
            row.get("pre_run_gpu_state", {}).get("gpu_use_percent") != 0 or
            row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
            row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 2
            for row in workers):
        errors.append("FP8 activation row-range worker evidence changed")
    aggregates = summary.get("aggregates", [])
    if summary.get("status") != "pass" or summary.get("rows") != 208 or \
            len(aggregates) != 8 or \
            "not performance evidence" not in summary.get("boundary", ""):
        errors.append("FP8 activation row-range summary changed")
    by_key = {(row["model"], row["boundary"]): row for row in aggregates}
    qwen_norm = by_key["qwen2.5-0.5b", "ffn_norm"]
    qwen_act = by_key["qwen2.5-0.5b", "ffn.activated"]
    deep_act = by_key["deepseek-r1-distill-qwen-1.5b", "ffn.activated"]
    deep_attention = by_key[
        "deepseek-r1-distill-qwen-1.5b", "attention.context"]
    if not 4.0 < qwen_norm.get("row_spread_p50", 0.0) < 4.1 or \
            qwen_norm.get("quarter_range_rows") != 79 or \
            not 1100.0 < qwen_act.get("row_spread_max", 0.0) < 1110.0 or \
            not 2070.0 < deep_act.get("row_spread_max", 0.0) < 2080.0 or \
            deep_attention.get("quarter_range_rows") != 0 or \
            not 1.1 < deep_attention.get("row_spread_p50", 0.0) < 1.2:
        errors.append("FP8 activation row-range skew evidence changed")
    traces = manifest.get("traces", [])
    if manifest.get("source_controlled") is not False or len(traces) != 2 or \
            sum(row.get("records", 0) for row in traces) != 738 or \
            sum(row.get("bytes", 0) for row in traces) != 94989569:
        errors.append("FP8 activation row-range trace manifest changed")
    if gates.get("status") != "support_ffn_per_row_reject_universal_per_row" or \
            gates.get("decision", {}).get(
                "ffn_per_row_scale_supported_by_evidence") is not True or \
            gates.get("decision", {}).get(
                "attention_per_row_scale_supported_by_evidence") is not False or \
            gates.get("decision", {}).get("universal_per_row_policy_accepted") is not False:
        errors.append("FP8 activation row-range gates changed")
    return len(raw), qwen_norm.get("quarter_range_rows", 0), len(traces)


def validate_fp8_ffn_outer_row(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "131-data"
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    pilot = [json.loads(line) for line in (data / "pilot-raw.jsonl").read_text(
        encoding="utf-8").splitlines()]
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if len(raw) != 36 or any(row.get("status") != "pass" for row in raw) or any(
            row.get("pre_run_gpu_state", {}).get("vram_percent", 99) > 1 or
            row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
            row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
            row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
            for row in raw):
        errors.append("FFN outer-row raw or idle-gate evidence changed")
    aggregates = summary.get("aggregates", [])
    fp8 = [row for row in aggregates if row.get("policy") == "fp8"]
    if summary.get("status") != "complete_with_recorded_accuracy_failures" or \
            summary.get("accuracy_failure_count") != 4 or len(fp8) != 4 or \
            summary.get("fp8_activation_scale_mode") != "ffn-outer-row" or \
            summary.get("fp8_activation_scale") != 0.2 or \
            summary.get("fp8_activation_minimum_scale") != 0.0001 or any(
                row.get("precision_gate_passed_all") is not False or
                row.get("top_token_equal_all") is not True or
                row.get("fp8_outer_row_native_statuses") != [0]
                for row in fp8):
        errors.append("FFN outer-row aggregate contract changed")
    by_key = {(row["model"], row["context"]): row for row in fp8}
    expected = {
        ("qwen2.5-0.5b", 8): (0.18, 0.20, 1600, 1700, 288),
        ("qwen2.5-0.5b", 512): (0.39, 0.41, 68000, 69500, 288),
        ("deepseek-r1-distill-qwen-1.5b", 8): (0.21, 0.23, 940, 1000, 336),
        ("deepseek-r1-distill-qwen-1.5b", 512): (0.23, 0.24, 35000, 35800, 336),
    }
    for key, (rl, rh, tl, th, calls) in expected.items():
        row = by_key[key]
        if not rl < row.get("root_mean_square_error_max", 0) < rh or \
                not tl < row.get("prefill_tokens_per_second_p50", 0) < th or \
                row.get("fp8_outer_row_fallback_calls_p50") != calls:
            errors.append(f"FFN outer-row result changed for {key}")
    if len(pilot) != 3 or gates.get("full_release", {}).get("passed") != 344 or \
            gates.get("decision", {}).get("ffn_only_routing_retained") is not True or \
            gates.get("decision", {}).get("fp8_model_policy_accepted") is not False:
        errors.append("FFN outer-row pilot/gates changed")
    return len(raw), len(fp8), len(pilot)


def validate_fp8_device_weight_amax(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "132-data"
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    pilot = [json.loads(line) for line in (data / "pilot-raw.jsonl").read_text(
        encoding="utf-8").splitlines()]
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    fresh_build = (data / "fresh-build.log").read_text(encoding="utf-8")
    rejected_build = (data / "rejected-build.log").read_text(encoding="utf-8")
    if len(raw) != 36 or any(row.get("status") != "pass" for row in raw) or any(
            row.get("pre_run_gpu_state", {}).get("vram_percent") != 0 or
            row.get("pre_run_gpu_state", {}).get("gpu_use_percent", 99) > 1 or
            row.get("post_run_gpu_state", {}).get("vram_percent", 99) > 2 or
            row.get("post_run_gpu_state", {}).get("gpu_use_percent", 99) > 4
            for row in raw):
        errors.append("device weight amax raw or idle evidence changed")
    aggregates = summary.get("aggregates", [])
    fp8 = [row for row in aggregates if row.get("policy") == "fp8"]
    if summary.get("status") != "complete_with_recorded_accuracy_failures" or \
            summary.get("fp8_weight_scale_mode") != "device-tensor-amax" or \
            summary.get("accuracy_failure_count") != 4 or len(fp8) != 4 or any(
                row.get("precision_gate_passed_all") is not False or
                row.get("top_token_equal_all") is not True
                for row in fp8):
        errors.append("device weight amax aggregate contract changed")
    by_key = {(row["model"], row["context"]): row for row in fp8}
    expected = {
        ("qwen2.5-0.5b", 8): (480, 520, 1900, 2100, 0.65, 0.68),
        ("qwen2.5-0.5b", 512): (480, 520, 90000, 92000, 1.22, 1.24),
        ("deepseek-r1-distill-qwen-1.5b", 8): (2050, 2180, 1320, 1410, 1.10, 1.12),
        ("deepseek-r1-distill-qwen-1.5b", 512): (2050, 2180, 50500, 52500, 1.28, 1.30),
    }
    for key, (pl, ph, tl, th, rl, rh) in expected.items():
        row = by_key[key]
        if not pl < row.get("weight_preparation_ms_p50", 0) < ph or \
                not tl < row.get("prefill_tokens_per_second_p50", 0) < th or \
                not rl < row.get("root_mean_square_error_max", 0) < rh:
            errors.append(f"device weight amax result changed for {key}")
    fp8_raw = [row for row in raw if row.get("policy") == "fp8"]
    qwen = next(row for row in fp8_raw if row["model"] == "qwen2.5-0.5b")
    deep = next(row for row in fp8_raw if row["model"].startswith("deepseek"))
    if qwen.get("fp8_device_amax_tensors") != 168 or \
            qwen.get("fp8_device_weight_bytes_scanned") != 1431306240 or \
            deep.get("fp8_device_amax_tensors") != 197 or \
            deep.get("fp8_device_weight_bytes_scanned") != 6174277632 or any(
                row.get("fp8_weight_bytes_scanned") != 0 or
                row.get("fp8_host_scale_summary_available") is not False
                for row in fp8_raw):
        errors.append("device weight amax scan metadata changed")
    if len(pilot) != 3 or "[34/34] Linking HIP executable" not in fresh_build or \
            "invalid operands" not in rejected_build or \
            gates.get("decision", {}).get("device_weight_amax_retained") is not True or \
            gates.get("decision", {}).get("bit_equivalent_to_host_amax") is not False or \
            gates.get("decision", {}).get("fp8_model_policy_accepted") is not False:
        errors.append("device weight amax build/pilot/gates changed")
    return len(raw), len(fp8), len(pilot)


def validate_fp8_multiblock_amax(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "133-data"
    weight_raw = [json.loads(line) for line in (data / "weight" / "raw.jsonl").read_text(
        encoding="utf-8").splitlines()]
    activation_raw = [json.loads(line) for line in
                      (data / "activation" / "raw.jsonl").read_text(
                          encoding="utf-8").splitlines()]
    weight = json.loads((data / "weight" / "summary.json").read_text(encoding="utf-8"))
    activation = json.loads((data / "activation" / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    build = (data / "fresh-build.log").read_text(encoding="utf-8")
    if len(weight_raw) != 18 or len(activation_raw) != 18 or any(
            row.get("status") != "pass" for row in weight_raw + activation_raw):
        errors.append("multi-block amax raw contract changed")
    if weight.get("accuracy_failure_count") != 2 or \
            activation.get("accuracy_failure_count") != 2 or \
            weight.get("fp8_weight_scale_mode") != "device-tensor-amax" or \
            activation.get("fp8_activation_scale_mode") != "tensor-amax":
        errors.append("multi-block amax suite contract changed")
    weight_fp8 = {row["model"]: row for row in weight["aggregates"]
                  if row["policy"] == "fp8"}
    activation_fp8 = {row["model"]: row for row in activation["aggregates"]
                      if row["policy"] == "fp8"}
    q_weight = weight_fp8["qwen2.5-0.5b"]
    d_weight = weight_fp8["deepseek-r1-distill-qwen-1.5b"]
    q_activation = activation_fp8["qwen2.5-0.5b"]
    d_activation = activation_fp8["deepseek-r1-distill-qwen-1.5b"]
    if not 19 < q_weight["weight_preparation_ms_p50"] < 22 or \
            not 27 < d_weight["weight_preparation_ms_p50"] < 31 or \
            not 75000 < q_activation["prefill_tokens_per_second_p50"] < 76000 or \
            not 44000 < d_activation["prefill_tokens_per_second_p50"] < 46000 or \
            q_weight["root_mean_square_error_max"] != 0.6643638885201558 or \
            d_weight["root_mean_square_error_max"] != 1.111237406654093 or \
            q_activation["root_mean_square_error_max"] != 0.2925066092695958 or \
            d_activation["root_mean_square_error_max"] != 0.249140100254465:
        errors.append("multi-block amax performance or exact error evidence changed")
    if "[34/34] Linking HIP executable" not in build or \
            gates.get("decision", {}).get("multiblock_reduction_retained") is not True or \
            gates.get("decision", {}).get("weight_error_signature_unchanged") is not True or \
            gates.get("decision", {}).get("activation_error_signature_unchanged") is not True or \
            gates.get("decision", {}).get("fp8_model_policy_accepted") is not False:
        errors.append("multi-block amax build/gates changed")
    return len(weight_raw) + len(activation_raw), 2, 2


def validate_fp8_dynamic_profile(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "134-data"
    qwen = json.loads((data / "qwen" / "parsed-summary.json").read_text(encoding="utf-8"))
    deep = json.loads((data / "deepseek" / "parsed-summary.json").read_text(encoding="utf-8"))
    combined = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    for model, expected_calls, scale_ms, gemm_ms in (
            (qwen, 168, 2.122265, 3.118825),
            (deep, 197, 3.109078, 5.518931)):
        categories = {row["category"]: row for row in model["kernel_categories"]}
        scale_total = sum(categories[name]["total_time_ms"] for name in (
            "fp8_tensor_scale_kernel", "fp8_finalize_scale_kernel",
            "quantize_fp8_device_scale_kernel"))
        if any(categories[name]["calls"] != expected_calls for name in (
                "fp8_tensor_scale_kernel", "fp8_finalize_scale_kernel",
                "quantize_fp8_device_scale_kernel")) or \
                abs(scale_total - scale_ms) > 1.0e-6 or \
                abs(categories["hipBLASLt/Tensile GEMM"]["total_time_ms"] - gemm_ms) > 1.0e-6 or \
                categories["dequant/fallback"]["calls"] != 0:
            errors.append(f"dynamic profile categories changed for {model['model']}")
        attributable = model.get("measured_forward_attributable", {})
        if attributable.get("known_calls") != expected_calls * 3 + \
                categories["hipBLASLt/Tensile GEMM"]["calls"] or \
                "Whole-process other" not in attributable.get(
                    "classification_boundary", ""):
            errors.append(f"dynamic profile boundary changed for {model['model']}")
    if len(combined.get("profiles", [])) != 2 or \
            gates.get("decision", {}).get("shared_qkv_quantization_supported") is not True or \
            gates.get("decision", {}).get("shared_gate_up_quantization_supported") is not True or \
            gates.get("decision", {}).get("whole_process_other_is_forward_only") is not False:
        errors.append("dynamic profile combined/gates changed")
    return qwen["kernel_total"]["calls"] + deep["kernel_total"]["calls"], 168, 197


def validate_fp8_shared_activation(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "135-data"
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    build = (data / "fresh-build.log").read_text(encoding="utf-8")
    if len(raw) != 18 or any(row.get("status") != "pass" for row in raw) or \
            summary.get("accuracy_failure_count") != 2:
        errors.append("shared activation raw/summary contract changed")
    fp8 = {row["model"]: row for row in summary["aggregates"]
           if row["policy"] == "fp8"}
    qwen = fp8["qwen2.5-0.5b"]
    deep = fp8["deepseek-r1-distill-qwen-1.5b"]
    if qwen.get("fp8_dynamic_tensor_calls_p50") != 384 or \
            deep.get("fp8_dynamic_tensor_calls_p50") != 452 or \
            qwen.get("fp8_dynamic_row_calls_p50") != 0 or \
            deep.get("fp8_dynamic_row_calls_p50") != 0 or \
            not 84000 < qwen.get("prefill_tokens_per_second_p50", 0) < 86000 or \
            not 50000 < deep.get("prefill_tokens_per_second_p50", 0) < 51000 or \
            qwen.get("root_mean_square_error_max") != 0.2925066092695958 or \
            deep.get("root_mean_square_error_max") != 0.249140100254465:
        errors.append("shared activation calls/performance/error evidence changed")
    if verification.get("all_expected_checks_passed") is not True or \
            verification.get("build", {}).get("steps_completed") != 50 or \
            "[50/50] Linking HIP executable" not in build or \
            gates.get("decision", {}).get("shared_qkv_quantization_retained") is not True or \
            gates.get("decision", {}).get("complete_logit_errors_unchanged") is not True or \
            gates.get("decision", {}).get("fp8_model_policy_accepted") is not False:
        errors.append("shared activation build/verification/gates changed")
    return len(raw), int(qwen["fp8_dynamic_tensor_calls_p50"]), \
        int(deep["fp8_dynamic_tensor_calls_p50"])


def validate_fp8_shared_profile(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "136-data"
    qwen = json.loads((data / "qwen" / "parsed-summary.json").read_text(encoding="utf-8"))
    deep = json.loads((data / "deepseek" / "parsed-summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    for model, calls, scale_ms, known_ms, reduction, call_reduction in (
            (qwen, 96, 1.154471, 4.167181, -20.490184293725157, 216),
            (deep, 113, 1.754955, 7.155494, -17.066683634660095, 252)):
        comparison = model.get("comparison_to_exp134", {})
        categories = {row["category"]: row for row in model["kernel_categories"]}
        actual_scale = sum(categories[name]["total_time_ms"] for name in (
            "fp8_tensor_scale_kernel", "fp8_finalize_scale_kernel",
            "quantize_fp8_device_scale_kernel"))
        if any(categories[name]["calls"] != calls for name in (
                "fp8_tensor_scale_kernel", "fp8_finalize_scale_kernel",
                "quantize_fp8_device_scale_kernel")) or \
                abs(actual_scale - scale_ms) > 1.0e-6 or \
                abs(model["measured_forward_attributable"]["known_total_time_ms"] - known_ms) > 1.0e-6 or \
                abs(comparison.get("known_forward_time_percent_change", 0) - reduction) > 1.0e-9 or \
                comparison.get("kernel_call_reduction") != call_reduction:
            errors.append(f"shared profile evidence changed for {model['model']}")
    if gates.get("decision", {}).get("shared_quantization_attribution_confirmed") is not True or \
            gates.get("decision", {}).get("gemm_calls_unchanged") is not True or \
            gates.get("decision", {}).get("continue_performance_complexity_now") is not False:
        errors.append("shared profile gates changed")
    return qwen["kernel_total"]["calls"] + deep["kernel_total"]["calls"], 96, 113


def validate_fp8_layer_drift(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "137-data"
    analysis = json.loads((data / "analysis.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "trace-manifest.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if analysis.get("all_stage_counts_match") is not True or \
            analysis.get("all_values_complete_no_truncation") is not True or \
            len(summary.get("summaries", [])) != 2:
        errors.append("FP8 layer drift completeness changed")
    models = {row["model"]: row for row in analysis["models"]}
    qwen = models["qwen2.5-0.5b"]
    deep = models["deepseek-r1-distill-qwen-1.5b"]
    if qwen["expected_stage_count"] != 26 or \
            qwen["largest_relative_l2_jump"]["to"] != "inference.blocks.21" or \
            not 0.20 < qwen["largest_relative_l2_jump"]["delta_relative_l2"] < 0.21 or \
            deep["expected_stage_count"] != 30 or \
            deep["maximum_max_abs_stage"]["name"] != "inference.blocks.27" or \
            not 0.24 < deep["final_logits"]["relative_l2"] < 0.25:
        errors.append("FP8 layer drift localization changed")
    traces = manifest.get("traces", [])
    if len(traces) != 4 or sum(row["selected_stages"] for row in traces) != 112 or \
            sum(row["truncated"] for row in traces) != 0 or \
            sum(row["selected_values"] for row in traces) != 1678848:
        errors.append("FP8 layer drift trace manifest changed")
    if gates.get("decision", {}).get("qwen_detail_layer") != 21 or \
            gates.get("decision", {}).get("deepseek_detail_layer") != 27 or \
            gates.get("decision", {}).get("complete_values_verified") is not True:
        errors.append("FP8 layer drift gates changed")
    return 56, 21, 27


def validate_fp8_block_detail(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "138-data"
    analysis = json.loads((data / "analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "trace-manifest.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    build = (data / "fresh-build.log").read_text(encoding="utf-8")
    if analysis.get("all_models_have_expected_16_stages") is not True or \
            analysis.get("all_selected_values_complete_no_truncation") is not True:
        errors.append("FP8 block detail completeness changed")
    models = {row["model"]: row for row in analysis["models"]}
    qwen = models["qwen2.5-0.5b"]
    deep = models["deepseek-r1-distill-qwen-1.5b"]
    if qwen["layer"] != 21 or qwen["stage_count"] != 16 or \
            qwen["largest_positive_delta"]["previous_stage"] != \
            "inference.blocks.21.ffn.output" or \
            not 0.19 < qwen["largest_positive_delta"]["delta_relative_l2"] < 0.20 or \
            deep["layer"] != 27 or deep["stage_count"] != 16 or \
            deep["largest_positive_delta"]["previous_stage"] != \
            "inference.blocks.27.ffn.output" or \
            not 0.08 < deep["largest_positive_delta"]["delta_relative_l2"] < 0.09:
        errors.append("FP8 block detail residual boundary changed")
    if not 0.04 < qwen["gate_stage"]["relative_l2"] < 0.05 or \
            not 0.06 < qwen["up_stage"]["relative_l2"] < 0.07 or \
            not 0.04 < deep["gate_stage"]["relative_l2"] < 0.05 or \
            not 0.03 < deep["up_stage"]["relative_l2"] < 0.04:
        errors.append("FP8 block detail gate/up evidence changed")
    traces = manifest.get("traces", [])
    if len(traces) != 4 or sum(row["selected"] for row in traces) != 64 or \
            sum(row["values"] for row in traces) != 1038336 or \
            sum(row["truncated"] for row in traces) != 0 or \
            "[50/50] Linking HIP executable" not in build:
        errors.append("FP8 block detail trace/build manifest changed")
    if gates.get("decision", {}).get("gate_up_primary_explosion") is not False or \
            gates.get("decision", {}).get("residual_cancellation_supported") is not True or \
            gates.get("decision", {}).get("residual_cancellation_proven") is not False:
        errors.append("FP8 block detail gates changed")
    return 32, 21, 27


def validate_fp8_residual_cancellation(errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "139-data"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    if summary.get("all_complete_values_verified") is not True or \
            summary.get("cancellation_proven_for_all_selected_blocks") is not True or \
            len(summary.get("rows", [])) != 2:
        errors.append("residual cancellation summary contract changed")
    rows = {row["model"]: row for row in summary["rows"]}
    qwen = rows["qwen2.5-0.5b"]
    deep = rows["deepseek-r1-distill-qwen-1.5b"]
    if not 17.0 < qwen["fp32_reference"]["residual_plus_ffn_norm_over_sum"] < 17.1 or \
            not -1.0 < qwen["fp32_reference"]["cosine_residual_ffn"] < -0.99 or \
            qwen["relative_l2_decomposition"]["factor_product_minus_observed_ratio"] != 0.0 or \
            not 4.4 < deep["fp32_reference"]["residual_plus_ffn_norm_over_sum"] < 4.5 or \
            not -0.91 < deep["fp32_reference"]["cosine_residual_ffn"] < -0.89 or \
            deep["reconstruction"]["fp32_block_eq_f32_residual_plus_ffn_max_abs_error"] != 0.0 or \
            deep["reconstruction"]["fp8_block_eq_f32_residual_plus_ffn_max_abs_error"] != 0.0:
        errors.append("residual cancellation algebra changed")
    if gates.get("decision", {}).get("residual_cancellation_proven") is not True or \
            gates.get("decision", {}).get(
                "fp32_block_counterfactual_needed_to_prove_cancellation") is not False or \
            gates.get("decision", {}).get(
                "mixed_precision_counterfactual_needed_for_causal_fix") is not True:
        errors.append("residual cancellation gates changed")
    return 2, 21, 27


def validate_fp8_selective_block_counterfactual(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "140-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    build = (data / "fresh-build.log").read_text(encoding="utf-8")
    binary_contract = (data / "hf-cli-binary-contract.log").read_text(encoding="utf-8")
    if verification.get("all_suites_passed") is not True or \
            len(verification.get("suites", [])) != 2:
        errors.append("selective FP32 combined verification changed")
        return 0, 0, 0
    suites = {row["model"]: row for row in verification["suites"]}
    qwen = suites["qwen2.5-0.5b"]
    deep = suites["deepseek-r1-distill-qwen-1.5b"]
    formal_workers = sum(row["execution"]["worker_rows"] for row in suites.values())
    precision_failures = sum(
        row["execution"]["accuracy_failure_count"] for row in suites.values())
    for name, suite, layer, converted, calls in (
            ("qwen", qwen, "21", 161, 368),
            ("deepseek", deep, "27", 190, 436)):
        raw_rows = sum(1 for line in (data / name / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        preflight_rows = sum(1 for line in (data / name / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        if raw_rows != 18 or preflight_rows != 3 or \
                (data / name / "stderr.log").stat().st_size != 0 or \
                suite["fp8_fp32_layers"] != layer or \
                suite["counter_checks"]["expected_converted_tensors"] != converted or \
                suite["counter_checks"]["expected_t512_dynamic_tensor_calls"] != calls or \
                suite["complete_logits"]["logit_count_values"] != [151936] or \
                suite["complete_logits"]["top_token_equal_all"] is not True:
            errors.append(f"selective FP32 {name} execution contract changed")
    q_t8, q_t512 = qwen["comparisons"]
    d_t8, d_t512 = deep["comparisons"]
    if not 0.96 < q_t8["delta"]["root_mean_square_error_ratio"] < 0.98 or \
            not 1.03 < q_t512["delta"]["root_mean_square_error_ratio"] < 1.05 or \
            not 0.85 < d_t8["delta"]["root_mean_square_error_ratio"] < 0.88 or \
            not 1.05 < d_t512["delta"]["root_mean_square_error_ratio"] < 1.07 or \
            not 0.98 < q_t512["delta"]["tps_ratio"] < 1.0 or \
            not 0.96 < d_t512["delta"]["tps_ratio"] < 0.98:
        errors.append("selective FP32 precision/performance evidence changed")
    if formal_workers != 36 or precision_failures != 4 or \
            "[50/50] Linking HIP executable" not in build or \
            "binary contract: pass" not in binary_contract:
        errors.append("selective FP32 build/worker evidence changed")
    decision = gates.get("decision", {})
    if decision.get("selective_fp32_api_retained") is not True or \
            decision.get("critical_block_model_policy_accepted") is not False or \
            decision.get("critical_block_is_primary_error_source") is not False or \
            decision.get("enable_by_default") is not False:
        errors.append("selective FP32 decision gates changed")
    return formal_workers, precision_failures, len(suites)


def validate_fp8_error_source_isolation(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "141-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    build = (data / "fresh-build.log").read_text(encoding="utf-8")
    contract = (data / "hf-cli-binary-contract.log").read_text(encoding="utf-8")
    suites = verification.get("suites", [])
    if verification.get("all_suites_passed") is not True or len(suites) != 2:
        errors.append("FP8 error-source combined verification changed")
        return 0, 0, 0
    by_mode = {row["mode"]: row for row in suites}
    for mode, expected_converted, q_calls, deep_calls in (
            ("weight-only", {168, 197}, 0, 0),
            ("activation-only", {0}, 96, 113)):
        suite = by_mode[mode]
        directory = data / mode
        raw_rows = sum(1 for line in (directory / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        preflight_rows = sum(1 for line in (directory / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        checks = suite.get("fp8_checks", [])
        by_model = {row["model"]: row for row in checks if row["context"] == 8}
        if suite.get("all_contract_checks_passed") is not True or \
                suite.get("worker_rows") != 12 or suite.get("fp8_worker_rows") != 4 or \
                raw_rows != 12 or preflight_rows != 3 or \
                (directory / "stderr.log").stat().st_size != 0 or \
                (directory / "exit-code.txt").read_text(encoding="utf-8").strip() != "0" or \
                len(checks) != 4 or {row["converted_tensors"] for row in checks} != \
                expected_converted:
            errors.append(f"FP8 error-source {mode} execution contract changed")
            continue
        qwen = by_model["qwen2.5-0.5b"]
        deep = by_model["deepseek-r1-distill-qwen-1.5b"]
        if qwen["fp8_linears_covered"] != 168 or \
                deep["fp8_linears_covered"] != 197 or \
                qwen["dynamic_tensor_calls"] != q_calls or \
                deep["dynamic_tensor_calls"] != deep_calls or \
                any(row["logit_count"] != 151936 or
                    row["top_token_equal"] is not True or
                    row["native_shapes"] != 0 or
                    row["software_fallback_calls"] != 0
                    for row in checks):
            errors.append(f"FP8 error-source {mode} machine counters changed")
    comparisons = {(row["model"], row["context"]): row
                   for row in verification.get("comparisons", [])}
    q8 = comparisons[("qwen2.5-0.5b", 8)]
    q512 = comparisons[("qwen2.5-0.5b", 512)]
    d8 = comparisons[("deepseek-r1-distill-qwen-1.5b", 8)]
    d512 = comparisons[("deepseek-r1-distill-qwen-1.5b", 512)]
    if q8["dominant_source_by_rms"] != "weight-only" or \
            q512["dominant_source_by_max_abs"] != "weight-only" or \
            not 1.60 < q512["weight_over_activation"][
                "root_mean_square_error_ratio"] < 1.63 or \
            d8["dominant_source_by_max_abs"] != "comparable-within-5-percent" or \
            d8["dominant_source_by_rms"] != "activation-only" or \
            d512["dominant_source_by_max_abs"] != "weight-only" or \
            d512["dominant_source_by_rms"] != "activation-only":
        errors.append("FP8 error-source attribution changed")
    decision = gates.get("decision", {})
    if decision.get("qwen_weight_rounding_dominates_all_reported_metrics") is not True or \
            decision.get("deepseek_has_one_universal_dominant_source") is not False or \
            decision.get("diagnostic_throughput_is_fp8_performance_evidence") is not False or \
            decision.get("combined_roundtrip_fp32_gemm_counterfactual_required") is not True or \
            "[50/50]" not in build or "binary contract: pass" not in contract:
        errors.append("FP8 error-source decision/build gates changed")
    return sum(row["worker_rows"] for row in suites), 8, 151936


def validate_fp8_native_roundtrip(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "142-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    raw_rows = sum(1 for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    pair_rows = sum(1 for line in (data / "pairs.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    execution = verification.get("execution", {})
    if verification.get("all_contract_checks_passed") is not True or \
            execution.get("worker_rows") != 12 or \
            execution.get("comparison_rows") != 4 or \
            execution.get("exit_code") != 0 or execution.get("stderr_bytes") != 0 or \
            raw_rows != 12 or pair_rows != 4 or preflight_rows != 3 or \
            (data / "stderr.log").stat().st_size != 0:
        errors.append("FP8 native-roundtrip execution contract changed")
    workers = verification.get("worker_checks", [])
    if len(workers) != 4 or any(
            row.get("passed") is not True or row.get("same_converted") is not True or
            row.get("same_dynamic") is not True or row.get("same_linears") is not True or
            row.get("complete_logits") is not True or
            row.get("both_roundtrip_zero_native_and_fallback") is not True
            for row in workers):
        errors.append("FP8 native-roundtrip worker counters changed")
    pairs = {(row["model"], row["context"]): row
             for row in verification.get("pair_checks", [])}
    expected_keys = {
        ("qwen2.5-0.5b", 8), ("qwen2.5-0.5b", 512),
        ("deepseek-r1-distill-qwen-1.5b", 8),
        ("deepseek-r1-distill-qwen-1.5b", 512)}
    if set(pairs) != expected_keys or any(
            row.get("passed") is not True or row.get("logit_count") != 151936 or
            row.get("all_top_tokens_equal") is not True or
            row.get("all_precision_gates_passed") is not False or
            row.get("native_is_major_additional_vector_perturbation") is not True or
            row.get("native_materially_increases_final_total_rms") is not False
            for row in pairs.values()):
        errors.append("FP8 native-roundtrip pair gates changed")
    if pairs and (
            not 0.57 < pairs[("qwen2.5-0.5b", 8)][
                "direct_native_rms_over_full_total_rms"] < 0.59 or
            not 0.75 < pairs[("qwen2.5-0.5b", 512)][
                "direct_native_rms_over_full_total_rms"] < 0.76 or
            not 0.76 < pairs[("deepseek-r1-distill-qwen-1.5b", 8)][
                "direct_native_rms_over_full_total_rms"] < 0.78 or
            not 0.54 < pairs[("deepseek-r1-distill-qwen-1.5b", 512)][
                "direct_native_rms_over_full_total_rms"] < 0.56):
        errors.append("FP8 native-roundtrip direct ratios changed")
    decision = gates.get("decision", {})
    build = (data / "fresh-build.log").read_text(encoding="utf-8")
    cli_contract = (data / "hf-cli-binary-contract.log").read_text(encoding="utf-8")
    runner_contract = (data / "benchmark-hf-fp8-native-roundtrip-contract.log").read_text(
        encoding="utf-8")
    if decision.get("native_gemm_is_major_additional_vector_perturbation") is not True or \
            decision.get("native_gemm_materially_increases_final_total_rms") is not False or \
            decision.get("replace_native_gemm_with_fp32_is_accepted_fix") is not False or \
            decision.get("finer_quantization_scale_remains_required") is not True or \
            "[50/50]" not in build or "binary contract: pass" not in cli_contract or \
            "OK" not in runner_contract:
        errors.append("FP8 native-roundtrip decision/build gates changed")
    return execution.get("worker_rows", 0), execution.get("comparison_rows", 0), 151936


def validate_fp8_output_channel_policy(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "143-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    execution = verification.get("execution", {})
    raw_rows = sum(1 for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if verification.get("all_execution_contract_checks_passed") is not True or \
            execution.get("worker_rows") != 36 or \
            execution.get("target_fp8_rows") != 12 or \
            execution.get("exit_code") != 0 or execution.get("stderr_bytes") != 0 or \
            raw_rows != 36 or preflight_rows != 3 or \
            (data / "stderr.log").stat().st_size != 0:
        errors.append("FP8 output-channel execution contract changed")
    checks = verification.get("counter_checks", [])
    if len(checks) != 12 or any(
            row.get("passed") is not True or row.get("logit_count") != 151936 or
            row.get("top_token_equal") is not True or
            row.get("fp8_dynamic_column_calls") != 0 or
            row.get("precision_gate_passed") is not False
            for row in checks):
        errors.append("FP8 output-channel worker checks changed")
    qwen = [row for row in checks if row["model"] == "qwen2.5-0.5b"]
    deep = [row for row in checks
            if row["model"] == "deepseek-r1-distill-qwen-1.5b"]
    if not qwen or not deep or any(
            row["fp8_linears_covered"] != 168 or
            row["converted_tensors"] != 168 or
            row["fp8_scale_bytes_retained"] != 1216512 or
            row["fp8_dynamic_tensor_calls_per_forward"] != 96.0
            for row in qwen) or any(
            row["fp8_linears_covered"] != 197 or
            row["converted_tensors"] != 197 or
            row["fp8_scale_bytes_retained"] != 3188224 or
            row["fp8_dynamic_tensor_calls_per_forward"] != 113.0
            for row in deep):
        errors.append("FP8 output-channel scale/count contract changed")
    comparisons = {(row["model"], row["context"]): row
                   for row in verification.get("comparisons", [])}
    q8 = comparisons.get(("qwen2.5-0.5b", 8), {})
    q512 = comparisons.get(("qwen2.5-0.5b", 512), {})
    d8 = comparisons.get(("deepseek-r1-distill-qwen-1.5b", 8), {})
    d512 = comparisons.get(("deepseek-r1-distill-qwen-1.5b", 512), {})
    if not 1.28 < q8.get("delta", {}).get("root_mean_square_error_ratio", 0) < 1.30 or \
            not 1.27 < q512.get("delta", {}).get("root_mean_square_error_ratio", 0) < 1.29 or \
            not 0.40 < d8.get("delta", {}).get("root_mean_square_error_ratio", 0) < 0.42 or \
            not 0.66 < d512.get("delta", {}).get("root_mean_square_error_ratio", 0) < 0.67 or \
            not 0.86 < q512.get("delta", {}).get("tps_ratio", 0) < 0.88 or \
            not 0.86 < d512.get("delta", {}).get("tps_ratio", 0) < 0.88:
        errors.append("FP8 output-channel precision/performance evidence changed")
    keep = verification.get("keep_gate", {})
    decision = gates.get("decision", {})
    build = (data / "fresh-build.log").read_text(encoding="utf-8")
    cli_contract = (data / "hf-cli-binary-contract.log").read_text(encoding="utf-8")
    matrix_contract = (data / "benchmark-hf-fp8-matrix-contract.log").read_text(
        encoding="utf-8")
    if keep.get("keep") is not False or keep.get("precision_gate_pass_count") != 0 or \
            keep.get("t512_tps_gate_pass_count") != 0 or \
            decision.get("accept_as_cross_model_default") is not False or \
            decision.get("retain_output_column_operator") is not True or \
            decision.get("qwen_precision_improved") is not False or \
            decision.get("deepseek_precision_improved") is not True or \
            "[50/50]" not in build or "binary contract: pass" not in cli_contract or \
            "OK" not in matrix_contract:
        errors.append("FP8 output-channel keep/build gates changed")
    return execution.get("worker_rows", 0), execution.get("target_fp8_rows", 0), \
        keep.get("precision_gate_pass_count", -1)


def validate_fp8_output_column_native_probe(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "144-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    execution = verification.get("execution", {})
    raw_rows = sum(1 for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if verification.get("all_checks_passed") is not True or \
            execution.get("worker_rows") != 6 or \
            execution.get("target_fp8_rows") != 2 or \
            execution.get("exit_code") != 0 or execution.get("stderr_bytes") != 0 or \
            raw_rows != 6 or preflight_rows != 3 or \
            (data / "stderr.log").stat().st_size != 0:
        errors.append("FP8 output-column native execution contract changed")
    gtest = verification.get("gtest", {})
    gtest_json = json.loads((data / "output-column-native-gtest.json").read_text(
        encoding="utf-8"))
    case = gtest_json["testsuites"][0]["testsuite"][0]
    if gtest.get("tests") != 1 or gtest.get("failures") != 0 or \
            gtest.get("output_column_native_status") != 0 or \
            gtest.get("output_column_scale_calls") != 1 or \
            case.get("output_column_native_status") != "0" or \
            case.get("output_column_scale_calls") != "1":
        errors.append("FP8 output-column native GTest evidence changed")
    checks = {row["model"]: row for row in verification.get("fp8_checks", [])}
    qwen = checks.get("qwen2.5-0.5b", {})
    deep = checks.get("deepseek-r1-distill-qwen-1.5b", {})
    if qwen.get("passed") is not True or qwen.get("fp8_linears_covered") != 168 or \
            qwen.get("fp8_output_column_native_status") != 0 or \
            qwen.get("fp8_output_column_scale_calls") != 336 or \
            qwen.get("fp8_dynamic_column_calls") != 0 or \
            deep.get("passed") is not True or deep.get("fp8_linears_covered") != 197 or \
            deep.get("fp8_output_column_native_status") != 0 or \
            deep.get("fp8_output_column_scale_calls") != 394 or \
            deep.get("fp8_dynamic_column_calls") != 0 or \
            any(row.get("logit_count") != 151936 or
                row.get("fp8_software_fallback_calls") != 0
                for row in checks.values()):
        errors.append("FP8 output-column native model counters changed")
    build = verification.get("build", {})
    decision = gates.get("decision", {})
    if build.get("base_steps_completed") != 50 or \
            build.get("hip_tests_incremental_steps") != 12 or \
            decision.get("native_output_column_vector_scale_supported_on_stack") is not False or \
            decision.get("scalar_native_gemm_plus_device_post_scale_supported") is not True or \
            decision.get("known_failed_submission_is_cached") is not True or \
            decision.get("direct_library_scale_can_remove_post_launch") is not False or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8") or \
            "binary contract: pass" not in (data / "hf-cli-binary-contract.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 output-column native capability gates changed")
    return execution.get("worker_rows", 0), execution.get("target_fp8_rows", 0), \
        gtest.get("output_column_native_status", -1)


def validate_fp8_weight_reconstruction_audit(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "145-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    execution = verification.get("execution", {})
    raw_rows = sum(1 for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if verification.get("all_checks_passed") is not True or \
            execution.get("tensor_rows") != 365 or execution.get("exit_code") != 0 or \
            execution.get("stderr_bytes") != 0 or raw_rows != 365 or \
            preflight_rows != 3 or (data / "stderr.log").stat().st_size != 0 or \
            (data / "benchmark-hf-fp8-weight-audit-contract-exit-code.txt").read_text(
                encoding="utf-8").strip() != "0" or \
            "OK" not in (data / "benchmark-hf-fp8-weight-audit-contract.stderr.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 weight audit execution/contract changed")
    models = {row["model"]: row for row in verification.get("model_checks", [])}
    qwen = models.get("qwen2.5-0.5b", {})
    deep = models.get("deepseek-r1-distill-qwen-1.5b", {})
    if qwen.get("actual_counts") != {
            "all": 168, "attention": 96, "ffn": 72, "output_head": 0} or \
            deep.get("actual_counts") != {
                "all": 197, "attention": 112, "ffn": 84, "output_head": 1} or \
            qwen.get("all_tensor_contracts_passed") is not True or \
            deep.get("all_tensor_contracts_passed") is not True:
        errors.append("FP8 weight audit model counts changed")
    groups = {(row["model"], row["group"]): row["reconstruction"]
              for row in verification.get("group_summaries", [])}
    q_all = groups.get(("qwen2.5-0.5b", "all_linear"), {})
    q_attention = groups.get(("qwen2.5-0.5b", "attention"), {})
    d_all = groups.get(("deepseek-r1-distill-qwen-1.5b", "all_linear"), {})
    d_head = groups.get(("deepseek-r1-distill-qwen-1.5b", "output_head"), {})
    if not 0.992 < q_all.get("column_over_scalar_relative_l2", 0) < 0.994 or \
            not 0.989 < q_attention.get("column_over_scalar_relative_l2", 0) < 0.991 or \
            not 0.995 < d_all.get("column_over_scalar_relative_l2", 0) < 0.997 or \
            not 0.989 < d_head.get("column_over_scalar_relative_l2", 0) < 0.992:
        errors.append("FP8 weight audit aggregate ratios changed")
    extremes = {row["model"]: row for row in verification.get("tensor_extremes", [])}
    if extremes.get("qwen2.5-0.5b", {}).get(
            "largest_relative_l2_improvement", {}).get("name") != \
            "model.layers.9.self_attn.k_proj.weight" or \
            extremes.get("deepseek-r1-distill-qwen-1.5b", {}).get(
                "smallest_relative_l2_improvement_or_largest_regression", {}).get("name") != \
            "model.layers.3.mlp.down_proj.weight":
        errors.append("FP8 weight audit tensor extremes changed")
    decision = gates.get("decision", {})
    if decision.get("external_audit_proves_microllm_model_precision") is not False or \
            decision.get("deepseek_output_head_is_best_group") is not True or \
            decision.get("global_ffn_scope_is_priority") is not False or \
            decision.get("output_head_only_is_next_minimal_counterfactual") is not True or \
            "External PyTorch ROCm" not in verification.get("boundary", ""):
        errors.append("FP8 weight audit scope/boundary gates changed")
    return execution.get("tensor_rows", 0), len(groups), sum(
        len(row.get("invalid_tensors", [])) for row in models.values())


def validate_fp8_output_head_only(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "146-data"
    verification = json.loads((data / "candidate" / "verification.json").read_text(
        encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    suites = verification.get("suites", {})
    for name in ("candidate", "control"):
        raw_rows = sum(1 for line in (data / name / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        preflight_rows = sum(1 for line in (data / name / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        if raw_rows != 36 or preflight_rows != 3 or \
                (data / name / "stderr.log").stat().st_size != 0 or \
                (data / name / "exit-code.txt").read_text(
                    encoding="utf-8").strip() != "0":
            errors.append(f"FP8 output-head-only {name} execution changed")
    if verification.get("all_execution_contract_checks_passed") is not True or \
            suites.get("combined_worker_rows") != 72 or \
            suites.get("combined_fp8_rows") != 24:
        errors.append("FP8 output-head-only combined execution changed")
    checks = verification.get("output_head_counter_checks", [])
    if len(checks) != 12 or any(
            row.get("passed") is not True or row.get("logit_count") != 151936 or
            row.get("fp8_dynamic_column_calls") != 0 or
            row.get("precision_gate_passed") is not False or
            row.get("top_token_equal") is not True
            for row in checks):
        errors.append("FP8 output-head-only worker checks changed")
    qwen = [row for row in checks if row["model"] == "qwen2.5-0.5b"]
    deep = [row for row in checks
            if row["model"] == "deepseek-r1-distill-qwen-1.5b"]
    if not qwen or not deep or any(
            row["fp8_scale_bytes_retained"] != 672 or
            row["fp8_output_column_scale_calls"] != 0 or
            row["fp8_output_column_native_status"] != -1
            for row in qwen) or any(
            row["fp8_scale_bytes_retained"] != 608528 or
            row["fp8_output_column_scale_calls"] != 4 or
            row["fp8_output_column_native_status"] != 0
            for row in deep):
        errors.append("FP8 output-head-only scope counters changed")
    comparisons = verification.get("same_revision_comparisons", [])
    if len(comparisons) != 4 or any(
            row["delta"]["maximum_absolute_error"] != 0.0 or
            row["delta"]["root_mean_square_error"] != 0.0 or
            row["delta"]["maximum_absolute_error_exactly_equal"] is not True or
            row["delta"]["root_mean_square_error_exactly_equal"] is not True
            for row in comparisons):
        errors.append("FP8 output-head-only numerical control changed")
    t512 = [row for row in comparisons if row["context"] == 512]
    if len(t512) != 2 or any(
            row["delta"]["t512_tps_degradation_le_5_percent"] is not True or
            not 0.99 < row["delta"]["tps_ratio"] < 1.0
            for row in t512):
        errors.append("FP8 output-head-only T512 performance gate changed")
    keep = verification.get("targeted_keep_gate", {})
    complete = verification.get("complete_precision_gates", {})
    historical = verification.get("historical_context_not_keep_evidence", {})
    decision = gates.get("decision", {})
    if keep.get("keep") is not False or \
            keep.get("deepseek_max_rms_both_improve") is not False or \
            keep.get("both_t512_tps_degradation_le_5_percent") is not True or \
            complete.get("pass_count") != 0 or \
            "do not determine keep" not in historical.get("boundary", "") or \
            decision.get("historical_host_tensor_baseline_is_valid_keep_evidence") is not False or \
            decision.get("remove_rejected_output_head_scope") is not True or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8"):
        errors.append("FP8 output-head-only decision/build gates changed")
    return suites.get("combined_worker_rows", 0), suites.get("combined_fp8_rows", 0), \
        complete.get("pass_count", -1)


def validate_fp8_attention_only(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "147-data"
    verification = json.loads((data / "candidate" / "verification.json").read_text(
        encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    for name in ("candidate", "control"):
        raw_rows = sum(1 for line in (data / name / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        preflight_rows = sum(1 for line in (data / name / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        if raw_rows != 36 or preflight_rows != 3 or \
                (data / name / "stderr.log").stat().st_size != 0 or \
                (data / name / "exit-code.txt").read_text(
                    encoding="utf-8").strip() != "0":
            errors.append(f"FP8 attention-only {name} execution changed")
    suites = verification.get("suites", {})
    if verification.get("all_execution_contract_checks_passed") is not True or \
            suites.get("combined_workers") != 72 or \
            suites.get("combined_fp8_rows") != 24:
        errors.append("FP8 attention-only combined execution changed")
    candidate = suites.get("attention_only", {}).get("counter_checks", [])
    if len(candidate) != 12 or any(
            row.get("passed") is not True or row.get("logit_count") != 151936 or
            row.get("fp8_dynamic_column_calls") != 0 or
            row.get("precision_gate_passed") is not False or
            row.get("fp8_output_column_native_status") != 0
            for row in candidate):
        errors.append("FP8 attention-only worker checks changed")
    qwen = [row for row in candidate if row["model"] == "qwen2.5-0.5b"]
    deep = [row for row in candidate
            if row["model"] == "deepseek-r1-distill-qwen-1.5b"]
    if not qwen or not deep or any(
            row["fp8_scale_bytes_retained"] != 196896 or
            row["fp8_output_column_scale_calls"] != 384
            for row in qwen) or any(
            row["fp8_scale_bytes_retained"] != 401748 or
            row["fp8_output_column_scale_calls"] != 448
            for row in deep):
        errors.append("FP8 attention-only scope counters changed")
    comparisons = {(row["model"], row["context"]): row
                   for row in verification.get("same_revision_comparisons", [])}
    q8 = comparisons.get(("qwen2.5-0.5b", 8), {}).get("delta", {})
    q512 = comparisons.get(("qwen2.5-0.5b", 512), {}).get("delta", {})
    d8 = comparisons.get(("deepseek-r1-distill-qwen-1.5b", 8), {}).get("delta", {})
    d512 = comparisons.get(("deepseek-r1-distill-qwen-1.5b", 512), {}).get("delta", {})
    if not 0.89 < q8.get("root_mean_square_error_ratio", 0) < 0.90 or \
            not 1.08 < q512.get("root_mean_square_error_ratio", 0) < 1.10 or \
            not 0.92 < d8.get("root_mean_square_error_ratio", 0) < 0.93 or \
            not 0.85 < d512.get("root_mean_square_error_ratio", 0) < 0.87 or \
            not 0.95 < q512.get("tps_ratio", 0) < 0.97 or \
            not 0.95 < d512.get("tps_ratio", 0) < 0.97:
        errors.append("FP8 attention-only precision/performance evidence changed")
    keep = verification.get("keep_gate", {})
    complete = verification.get("complete_precision_gates", {})
    decision = gates.get("decision", {})
    if keep.get("keep") is not False or \
            keep.get("all_four_cases_max_and_rms_not_worse") is not False or \
            keep.get("at_least_one_strict_improvement") is not True or \
            keep.get("both_t512_tps_degradation_le_5_percent") is not True or \
            complete.get("pass_count") != 0 or \
            decision.get("qwen_t512_rms_regressed") is not True or \
            decision.get("attention_output_only_is_next_minimal_scope") is not True or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8"):
        errors.append("FP8 attention-only decision/build gates changed")
    return suites.get("combined_workers", 0), suites.get("combined_fp8_rows", 0), \
        complete.get("pass_count", -1)


def validate_fp8_attention_output_only(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "148-data"
    verification = json.loads((data / "candidate" / "verification.json").read_text(
        encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    for name in ("candidate", "control"):
        raw_rows = sum(1 for line in (data / name / "raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        preflight_rows = sum(1 for line in (data / name / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
        if raw_rows != 36 or preflight_rows != 3 or \
                (data / name / "stderr.log").stat().st_size != 0 or \
                (data / name / "exit-code.txt").read_text(
                    encoding="utf-8").strip() != "0":
            errors.append(f"FP8 attention-output {name} execution changed")
    suites = verification.get("suites", {})
    if verification.get("all_checks_passed") is not True or \
            suites.get("combined_workers") != 72 or suites.get("combined_fp8") != 24:
        errors.append("FP8 attention-output combined execution changed")
    checks = suites.get("candidate", {}).get("checks", [])
    if len(checks) != 12 or any(
            row.get("passed") is not True or row.get("logit_count") != 151936 or
            row.get("hot_column") != 0 or row.get("status") != 0
            for row in checks):
        errors.append("FP8 attention-output worker checks changed")
    qwen = [row for row in checks if row["model"] == "qwen2.5-0.5b"]
    deep = [row for row in checks
            if row["model"] == "deepseek-r1-distill-qwen-1.5b"]
    if not qwen or not deep or any(
            row["scale"] != 86592 or row["post"] != 96
            for row in qwen) or any(
            row["scale"] != 172708 or row["post"] != 112
            for row in deep):
        errors.append("FP8 attention-output scope counters changed")
    comparisons = {(row["model"], row["context"]): row
                   for row in verification.get("comparisons", [])}
    q8 = comparisons.get(("qwen2.5-0.5b", 8), {})
    q512 = comparisons.get(("qwen2.5-0.5b", 512), {})
    d8 = comparisons.get(("deepseek-r1-distill-qwen-1.5b", 8), {})
    d512 = comparisons.get(("deepseek-r1-distill-qwen-1.5b", 512), {})
    if q8.get("delta", {}).get("max") != 0.0 or \
            q8.get("delta", {}).get("rms") != 0.0 or \
            q512.get("delta", {}).get("max") != 0.0 or \
            q512.get("delta", {}).get("rms") != 0.0 or \
            not -0.08 < d8.get("delta", {}).get("max", 0) < -0.07 or \
            not -0.02 < d8.get("delta", {}).get("rms", 0) < -0.01 or \
            not -0.18 < d512.get("delta", {}).get("max", 0) < -0.17 or \
            not -0.03 < d512.get("delta", {}).get("rms", 0) < -0.02 or \
            not -5.0 < q512.get("delta", {}).get("tps_percent", -100) < 0.0 or \
            not -5.0 < d512.get("delta", {}).get("tps_percent", -100) < 0.0:
        errors.append("FP8 attention-output precision/performance evidence changed")
    keep = verification.get("keep_gate", {})
    complete = verification.get("complete_precision", {})
    decision = gates.get("decision", {})
    if keep.get("keep") is not True or keep.get("eight_metrics_not_worse") is not True or \
            keep.get("at_least_one_improved") is not True or \
            keep.get("both_t512_pass") is not True or complete.get("pass_count") != 0 or \
            decision.get("targeted_keep") is not True or \
            decision.get("remove_broader_attention_scope") is not True or \
            decision.get("complete_fp8_precision_available") is not False or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8"):
        errors.append("FP8 attention-output keep/build gates changed")
    return suites.get("combined_workers", 0), suites.get("combined_fp8", 0), \
        complete.get("pass_count", -1)


def validate_fp8_clipped_pilot_invalid(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "149-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    build = verification.get("build", {})
    invalid = verification.get("invalid_artifacts", [])
    if verification.get("status") != "invalid_due_to_external_gpu_contention" or \
            verification.get("valid_fraction_suites") != 0 or \
            verification.get("required_fraction_suites") != 4 or \
            verification.get("valid_fp8_rows") != 0 or \
            verification.get("throughput_is_performance_evidence") is not False or \
            build.get("valid") is not True or build.get("steps_completed") != 50 or \
            len(invalid) != 2:
        errors.append("FP8 clipped invalid verification changed")
    by_fraction = {row["fraction"]: row for row in invalid}
    one = by_fraction.get(1.0, {})
    point75 = by_fraction.get(0.75, {})
    raw_rows = sum(1 for line in (data / "fraction-1-invalid" / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if one.get("runner_exit_code") != 1 or one.get("recorded_raw_rows") != 3 or \
            one.get("post_gate_state") != {"gpu_use_percent": 22, "vram_percent": 9} or \
            raw_rows != 3 or \
            (data / "fraction-1-invalid" / "exit-code.txt").read_text(
                encoding="utf-8").strip() != "1" or \
            (data / "fraction-1-invalid" / "stderr.log").stat().st_size == 0 or \
            point75.get("runner_launched") is not False or \
            len(point75.get("preflight", [])) != 1:
        errors.append("FP8 clipped invalid artifact evidence changed")
    sequences = verification.get("gpu_occupation_sequences", [])
    samples = [sample for row in sequences for sample in row.get("samples", [])]
    if len(sequences) != 3 or len(samples) != 18 or \
            max(sample["use"] for sample in samples) != 100 or \
            {sample["vram"] for sample in samples} != {57}:
        errors.append("FP8 clipped contention monitoring changed")
    decision = gates.get("decision", {})
    if decision.get("numerical_fraction_selected") is not False or \
            decision.get("contaminated_rows_may_be_merged_into_retry") is not False or \
            decision.get("retry_must_start_from_fraction_one") is not True or \
            decision.get("gpu_idle_gate_should_be_relaxed") is not False or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8") or \
            "binary contract: pass" not in (data / "hf-cli-binary-contract.log").read_text(
                encoding="utf-8") or \
            "OK" not in (data / "hf-fp8-matrix-contract.log").read_text(encoding="utf-8"):
        errors.append("FP8 clipped invalid decision/build gates changed")
    return verification.get("valid_fraction_suites", -1), \
        verification.get("required_fraction_suites", -1), raw_rows


def validate_fp8_fraction_workload_invalid(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "150-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    execution = verification.get("execution", {})
    raw_rows = sum(1 for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if verification.get("status") != "invalid_workload_mismatch" or \
            verification.get("valid_fraction_conclusion") is not False or \
            verification.get("valid_selection") is not None or \
            verification.get("worker_contract_checks_passed") is not True or \
            execution.get("workers") != 20 or execution.get("comparisons") != 16 or \
            execution.get("exit") != 0 or execution.get("stderr") != 0 or \
            raw_rows != 36 or preflight_rows != 3 or \
            (data / "stderr.log").stat().st_size != 0:
        errors.append("FP8 fraction workload-invalid execution changed")
    mismatch = verification.get("mismatch", {})
    cases = mismatch.get("fraction1_vs_exp148", [])
    if mismatch.get("runner_hardcoded_weight_scale") != 0.0001 or \
            mismatch.get("requested_retained_policy_weight_scale") != 0.005 or \
            mismatch.get("all_fraction1_metrics_match_exp148") is not False or \
            len(cases) != 4 or any(
                row.get("max_exact") is not False or row.get("rms_exact") is not False
                for row in cases):
        errors.append("FP8 fraction workload mismatch evidence changed")
    decision = gates.get("decision", {})
    if decision.get("runner_reported_selection_is_valid") is not False or \
            decision.get("numerical_fraction_selected") is not False or \
            decision.get("exp150_rows_may_be_merged_into_retry") is not False or \
            decision.get("corrected_runner_must_pin_or_expose_weight_scale") is not True or \
            decision.get("exp149_data_was_merged") is not False or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8") or \
            "binary contract: pass" not in (data / "hf-cli-binary-contract.log").read_text(
                encoding="utf-8") or \
            "OK" not in (data / "hf-fp8-fraction-pilot-contract.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 fraction workload-invalid decision/build gates changed")
    return execution.get("workers", 0), execution.get("comparisons", 0), len(cases)


def validate_fp8_clipped_coarse_grid(
        errors: list[str]) -> tuple[int, int, float]:
    data = ROOT / "experiments" / "151-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    execution = verification.get("execution", {})
    raw_rows = sum(1 for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if verification.get("all_checks_passed") is not True or \
            verification.get("all_fraction1_exact") is not True or \
            execution.get("workers") != 20 or execution.get("comparisons") != 16 or \
            execution.get("exit") != 0 or execution.get("stderr") != 0 or \
            raw_rows != 36 or preflight_rows != 3 or \
            verification.get("summary_weight_scale") != 0.005 or \
            (data / "stderr.log").stat().st_size != 0:
        errors.append("FP8 clipped coarse execution/baseline changed")
    matches = verification.get("fraction1_exp148_match", [])
    if len(matches) != 4 or any(
            row.get("max_exact") is not True or row.get("rms_exact") is not True
            for row in matches):
        errors.append("FP8 clipped coarse Exp148 match changed")
    table = {row["fraction"]: row for row in verification.get("fraction_table", [])}
    if set(table) != {1.0, 0.75, 0.5, 0.25} or \
            not 6.5 < table[0.75]["rms_over_f1"] < 6.6 or \
            not 9.5 < table[0.5]["rms_over_f1"] < 9.6 or \
            not 12.1 < table[0.25]["rms_over_f1"] < 12.3 or \
            table[0.75]["top_token_equal_all"] is not True or \
            table[0.5]["top_token_equal_all"] is not False or \
            verification.get("selection", {}).get("selected_fraction") != 1.0:
        errors.append("FP8 clipped coarse fraction results changed")
    checks = verification.get("worker_checks", [])
    if len(checks) != 20 or any(
            row.get("passed") is not True or row.get("logits") != 151936
            for row in checks):
        errors.append("FP8 clipped coarse worker checks changed")
    clipped = [row for row in checks if row.get("fraction") not in (None, 1.0)]
    if len(clipped) != 12 or any(
            row["clipped"] != row["dynamic"] or row["dynamic"] not in (96, 113)
            for row in clipped):
        errors.append("FP8 clipped coarse call counters changed")
    decision = gates.get("decision", {})
    if decision.get("coarse_selected_fraction") != 1.0 or \
            decision.get("fraction_075_or_lower_viable") is not False or \
            decision.get("fraction_095_09_085_refinement_required") is not True or \
            decision.get("model_clipping_default_changed") is not False or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8") or \
            "OK" not in (data / "hf-fp8-fraction-pilot-contract.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 clipped coarse decision/build gates changed")
    return execution.get("workers", 0), execution.get("comparisons", 0), \
        verification.get("selection", {}).get("selected_fraction", -1.0)


def validate_fp8_clipped_fine_grid(
        errors: list[str]) -> tuple[int, int, float]:
    data = ROOT / "experiments" / "152-data"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    execution = verification.get("execution", {})
    raw_rows = sum(1 for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    if verification.get("all_checks_passed") is not True or \
            verification.get("all_fraction1_exact") is not True or \
            execution.get("workers") != 20 or execution.get("comparisons") != 16 or \
            execution.get("exit") != 0 or execution.get("stderr") != 0 or \
            raw_rows != 36 or preflight_rows != 3 or \
            verification.get("weight_scale") != 0.005 or \
            verification.get("comparisons", {}).get("all_151936") is not True or \
            verification.get("comparisons", {}).get("finite") is not True or \
            (data / "stderr.log").stat().st_size != 0:
        errors.append("FP8 clipped fine execution/baseline changed")
    matches = verification.get("fraction1_exp148", [])
    if len(matches) != 4 or any(
            row.get("max_exact") is not True or row.get("rms_exact") is not True
            for row in matches):
        errors.append("FP8 clipped fine Exp148 match changed")
    table = {row["fraction"]: row for row in verification.get("fraction_table", [])}
    if set(table) != {1.0, 0.95, 0.9, 0.85} or \
            not 2.1 < table[0.95]["rms_over_f1"] < 2.2 or \
            not 4.9 < table[0.9]["rms_over_f1"] < 5.1 or \
            not 8.2 < table[0.85]["rms_over_f1"] < 8.3 or \
            any(row["top_token_equal_all"] is not True for row in table.values()) or \
            verification.get("selection", {}).get("selected_fraction") != 1.0:
        errors.append("FP8 clipped fine fraction results changed")
    decision = gates.get("decision", {})
    if decision.get("selected_fraction") != 1.0 or \
            decision.get("fraction_025_through_095_closed") is not True or \
            decision.get("remove_model_clipping_policy") is not True or \
            decision.get("retain_low_level_clipped_operator") is not True or \
            decision.get("complete_fp8_precision_available") is not False or \
            "[50/50]" not in (data / "fresh-build.log").read_text(encoding="utf-8") or \
            "OK" not in (data / "hf-fp8-fraction-pilot-contract.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 clipped fine decision/build gates changed")
    return execution.get("workers", 0), execution.get("comparisons", 0), \
        verification.get("selection", {}).get("selected_fraction", -1.0)


def validate_fp8_e5_activation_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "153-data"
    candidate = data / "candidate"
    control = data / "control"
    verification = json.loads(
        (candidate / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads(
        (candidate / "summary.json").read_text(encoding="utf-8"))
    control_summary = json.loads(
        (control / "summary.json").read_text(encoding="utf-8"))
    candidate_raw = sum(1 for line in (candidate / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    control_raw = sum(1 for line in (control / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    candidate_preflight = sum(1 for line in (
        candidate / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
    control_preflight = sum(1 for line in (
        control / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
    suites = verification.get("suites", {})
    if verification.get("all_checks_passed") is not True or \
            verification.get("revision") != \
            "821e7b8ba9ef8b3d7396dbd74c72bc78b114ac49" or \
            suites.get("combined_workers") != 72 or \
            suites.get("combined_fp8") != 24 or \
            candidate_raw != 36 or control_raw != 36 or \
            candidate_preflight != 3 or control_preflight != 3 or \
            (candidate / "stderr.log").stat().st_size != 0 or \
            (control / "stderr.log").stat().st_size != 0 or \
            (candidate / "exit-code.txt").read_text(encoding="utf-8").strip() != "0" or \
            (control / "exit-code.txt").read_text(encoding="utf-8").strip() != "0":
        errors.append("FP8 E5 activation execution contract changed")
    if candidate_summary.get("fp8_activation_format") != "e5m2-fnuz" or \
            control_summary.get("fp8_activation_format") != "e4m3-fnuz" or \
            len(candidate_summary.get("rows", [])) != 36 or \
            len(control_summary.get("rows", [])) != 36:
        errors.append("FP8 E5/E4 format or row identity changed")
    comparisons = verification.get("comparisons", [])
    if len(comparisons) != 4 or any(
            row.get("delta", {}).get("max_not_worse") is not False or
            row.get("delta", {}).get("rms_not_worse") is not False or
            row.get("candidate", {}).get("resident") !=
            row.get("control", {}).get("resident") or
            row.get("candidate", {}).get("peak") !=
            row.get("control", {}).get("peak")
            for row in comparisons):
        errors.append("FP8 E5 complete-logit regression evidence changed")
    ratios = [
        (row["delta"]["max_ratio"], row["delta"]["rms_ratio"])
        for row in comparisons
    ]
    if not ratios or min(value for pair in ratios for value in pair) < 1.5 or \
            max(value for pair in ratios for value in pair) < 3.4 or \
            verification.get("keep", {}).get("keep") is not False or \
            verification.get("keep", {}).get("both_t512_pass") is not True or \
            verification.get("complete_precision", {}).get("passes") != 0 or \
            verification.get("complete_precision", {}).get("total") != 4:
        errors.append("FP8 E5 rejection gate changed")
    for format_name, expected_dynamic, expected_post in (
            ("e4", {384, 452}, {96, 112}),
            ("e5", {384, 452}, {96, 112})):
        checks = suites.get(format_name, {}).get("checks", [])
        if len(checks) != 12 or any(
                row.get("passed") is not True or row.get("logits") != 151936
                for row in checks) or \
                {row.get("dynamic") for row in checks} != expected_dynamic or \
                {row.get("post") for row in checks} != expected_post:
            errors.append(f"FP8 {format_name} worker counters changed")
    decision = gates.get("decision", {})
    evidence = gates.get("evidence", {})
    active_surfaces = "\n".join(
        (REPOSITORY / name).read_text(encoding="utf-8")
        for name in ("include/microllm/model/config.h", "src/model/config.cpp",
                     "src/model/model.cpp", "apps/hf_infer.cpp",
                     "benchmarks/single_gpu/hf_fp8_matrix.py"))
    primitive_surfaces = "\n".join(
        (REPOSITORY / name).read_text(encoding="utf-8")
        for name in ("include/microllm/base/low_precision.h",
                     "tests/autograd/autograd_test.cpp",
                     "tests/ops/hip_ops_test.cpp"))
    if decision.get("targeted_keep") is not False or \
            decision.get("remove_model_e5_policy") is not True or \
            decision.get("retain_low_level_e5_dtype_and_ops") is not True or \
            decision.get("retain_mixed_operand_autograd_api") is not True or \
            evidence.get("max_rms_metrics_worse") != 8 or \
            evidence.get("complete_precision_gates_passed") != 0 or \
            "[50/50]" not in (data / "fresh-build.log").read_text(
                encoding="utf-8") or \
            "contract: pass" not in (data / "hf-cli-binary-contract.log").read_text(
                encoding="utf-8") or \
            "OK" not in (data / "hf-fp8-matrix-contract.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 E5 decision/build gates changed")
    if "Fp8ActivationFormat" in active_surfaces or \
            "--fp8-activation-format" in active_surfaces or \
            "Float8E5M2FNUZ" not in primitive_surfaces or \
            "MixedFp8FormatsUseE5ActivationE4WeightAndFp32Gradients" not in \
            primitive_surfaces or \
            "MixedE5ActivationAndE4WeightExecuteWithExplicitDispatch" not in \
            primitive_surfaces:
        errors.append("FP8 E5 model removal or primitive retention changed")
    return suites.get("combined_workers", 0), suites.get("combined_fp8", 0), \
        verification.get("complete_precision", {}).get("passes", -1)


def validate_fp8_layer_leave_one_out(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "154-data"
    verification = json.loads(
        (data / "verification.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    raw_rows = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    stdout_rows = sum(1 for line in (data / "stdout.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    preflight_rows = sum(1 for line in (data / "gpu2-preflight.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    execution = verification.get("execution", {})
    if verification.get("all_checks_passed") is not True or \
            verification.get("revision") != \
            "147864a0374fc51e27bfe4df5001652ec7e9c16d" or \
            execution.get("rows") != 56 or execution.get("candidates") != 52 or \
            execution.get("all_logits151936") is not True or \
            execution.get("all_finite") is not True or \
            execution.get("exit") != 0 or execution.get("stderr") != 0 or \
            len(raw_rows) != 56 or stdout_rows != 56 or preflight_rows != 3 or \
            (data / "stderr.log").stat().st_size != 0 or \
            (data / "exit-code.txt").read_text(encoding="utf-8").strip() != "0":
        errors.append("FP8 layer leave-one-out execution contract changed")
    models = verification.get("models", {})
    expected = {
        "qwen2.5-0.5b": {
            "layers": 24, "linears": 161, "dynamic": 92, "post": 23,
            "both": 20, "max_low": 0.71, "max_high": 0.72,
            "rms_low": 0.66, "rms_high": 0.67,
        },
        "deepseek-r1-distill-qwen-1.5b": {
            "layers": 28, "linears": 190, "dynamic": 109, "post": 27,
            "both": 0, "max_low": 1.02, "max_high": 1.03,
            "rms_low": 0.99, "rms_high": 1.0,
        },
    }
    if set(models) != set(expected):
        errors.append("FP8 layer leave-one-out model set changed")
    for name, contract in expected.items():
        model = models.get(name, {})
        best = model.get("best_candidate", {})
        routing = model.get("routing_contract", {})
        if model.get("candidate_rows") != contract["layers"] or \
                model.get("layer_values") != list(range(contract["layers"])) or \
                model.get("each_layer_once") is not True or \
                model.get("both_non_worse_count") != contract["both"] or \
                model.get("precision_gate_pass_count") != 0 or \
                model.get("top_equal_count") != contract["layers"] or \
                best.get("fp32_layer") != 9 or \
                not contract["max_low"] < best.get("maximum_over_baseline", 0) < \
                    contract["max_high"] or \
                not contract["rms_low"] < best.get("rms_over_baseline", 0) < \
                    contract["rms_high"] or \
                routing.get("passed") is not True or \
                routing.get("linears_values") != [contract["linears"]] or \
                routing.get("dynamic_values") != [contract["dynamic"]] or \
                routing.get("post_values") != [contract["post"]]:
            errors.append(f"FP8 layer leave-one-out {name} result changed")
    if summary.get("candidate_count") != 52 or len(summary.get("rows", [])) != 56 or \
            any(row.get("logit_count") != 151936 or
                row.get("top_token_equal") is not True for row in raw_rows):
        errors.append("FP8 layer leave-one-out summary/logit contract changed")
    decision = gates.get("decision", {})
    evidence = gates.get("evidence", {})
    if decision.get("screening_keep") is not False or \
            decision.get("shared_single_fp32_layer_viable") is not False or \
            decision.get("deepseek_single_fp32_layer_closed") is not True or \
            decision.get("qwen_layer9_requires_formal_short_long_matrix") is not True or \
            decision.get("throughput_used_for_selection") is not False or \
            evidence.get("rows") != 56 or evidence.get("candidates") != 52 or \
            "[50/50]" not in (data / "fresh-build.log").read_text(
                encoding="utf-8") or \
            "contract: pass" not in (data / "hf-cli-binary-contract.log").read_text(
                encoding="utf-8") or \
            "OK" not in (data / "hf-fp8-matrix-contract.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 layer leave-one-out decision/build gates changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "hf_fp8_layer_leave_one_out.py").read_text(encoding="utf-8")
    for token in ("attention-output-only", "tensor-amax", "rank_candidates",
                  "throughput is diagnostic only"):
        if token not in runner:
            errors.append(f"FP8 layer leave-one-out runner lost contract: {token}")
    return execution.get("rows", 0), execution.get("candidates", 0), \
        models.get("deepseek-r1-distill-qwen-1.5b", {}).get(
            "both_non_worse_count", -1)


def validate_fp8_qwen_layer9_formal(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "155-data"
    candidate = data / "candidate"
    control = data / "control"
    verification = json.loads(
        (candidate / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads(
        (candidate / "summary.json").read_text(encoding="utf-8"))
    control_summary = json.loads(
        (control / "summary.json").read_text(encoding="utf-8"))
    candidate_raw = sum(1 for line in (candidate / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    control_raw = sum(1 for line in (control / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    candidate_preflight = sum(1 for line in (
        candidate / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
    control_preflight = sum(1 for line in (
        control / "gpu2-preflight.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip())
    suites = verification.get("suites", {})
    if verification.get("all_checks_passed") is not True or \
            verification.get("revision") != \
            "147864a0374fc51e27bfe4df5001652ec7e9c16d" or \
            suites.get("combined_workers") != 36 or \
            suites.get("combined_fp8") != 12 or \
            candidate_raw != 18 or control_raw != 18 or \
            candidate_preflight != 3 or control_preflight != 3 or \
            (candidate / "stderr.log").stat().st_size != 0 or \
            (control / "stderr.log").stat().st_size != 0 or \
            (candidate / "exit-code.txt").read_text(encoding="utf-8").strip() != "0" or \
            (control / "exit-code.txt").read_text(encoding="utf-8").strip() != "0":
        errors.append("FP8 Qwen layer9 formal execution contract changed")
    if candidate_summary.get("fp8_fp32_layers") != "9" or \
            control_summary.get("fp8_fp32_layers") != "" or \
            len(candidate_summary.get("rows", [])) != 18 or \
            len(control_summary.get("rows", [])) != 18:
        errors.append("FP8 Qwen layer9 policy identity changed")
    comparisons = {row["context"]: row for row in verification.get("comparisons", [])}
    short = comparisons.get(8, {})
    long = comparisons.get(512, {})
    short_delta = short.get("delta", {})
    long_delta = long.get("delta", {})
    if set(comparisons) != {8, 512} or \
            not 0.71 < short_delta.get("max_ratio", 0) < 0.72 or \
            not 0.66 < short_delta.get("rms_ratio", 0) < 0.67 or \
            not 1.05 < long_delta.get("max_ratio", 0) < 1.06 or \
            not 1.36 < long_delta.get("rms_ratio", 0) < 1.37 or \
            long_delta.get("max_not_worse") is not False or \
            long_delta.get("rms_not_worse") is not False or \
            long_delta.get("t512_pass") is not True or \
            short.get("candidate", {}).get("resident", 0) - \
            short.get("control", {}).get("resident", 0) != 44724712 or \
            long.get("candidate", {}).get("peak", 0) - \
            long.get("control", {}).get("peak", 0) != 44724712:
        errors.append("FP8 Qwen layer9 short/long comparison changed")
    candidate_counts = suites.get("candidate", {}).get("counts", {})
    control_counts = suites.get("control", {}).get("counts", {})
    if candidate_counts.get("passed") is not True or \
            candidate_counts.get("rows") != 6 or \
            candidate_counts.get("layers") != ["9"] or \
            candidate_counts.get("linears") != [161] or \
            candidate_counts.get("dynamic") != [368] or \
            candidate_counts.get("post") != [92] or \
            control_counts.get("passed") is not True or \
            control_counts.get("rows") != 6 or \
            control_counts.get("layers") != [""] or \
            control_counts.get("linears") != [168] or \
            control_counts.get("dynamic") != [384] or \
            control_counts.get("post") != [96] or \
            candidate_counts.get("native") != [4] or \
            control_counts.get("native") != [4] or \
            candidate_counts.get("fallback_calls") != [0] or \
            control_counts.get("fallback_calls") != [0]:
        errors.append("FP8 Qwen layer9 routing counters changed")
    keep = verification.get("keep", {})
    precision = verification.get("complete_precision", {})
    decision = gates.get("decision", {})
    if keep.get("keep") is not False or \
            keep.get("four_metrics_not_worse") is not False or \
            keep.get("one_improved") is not True or \
            keep.get("t512_pass") is not True or \
            precision.get("passes") != 0 or precision.get("total") != 2 or \
            decision.get("targeted_keep") is not False or \
            decision.get("single_fp32_block_direction_closed") is not True or \
            decision.get("retain_fp32_layer_diagnostic_api") is not True or \
            "[50/50]" not in (data / "fresh-build.log").read_text(
                encoding="utf-8") or \
            "contract: pass" not in (data / "hf-cli-binary-contract.log").read_text(
                encoding="utf-8") or \
            "OK" not in (data / "hf-fp8-matrix-contract.log").read_text(
                encoding="utf-8"):
        errors.append("FP8 Qwen layer9 decision/build gates changed")
    return suites.get("combined_workers", 0), suites.get("combined_fp8", 0), \
        precision.get("passes", -1)


def validate_block_reduction_determinism(
        errors: list[str]) -> tuple[int, int, int]:
    data = ROOT / "experiments" / "156-data"
    verification = json.loads(
        (data / "verification.json").read_text(encoding="utf-8"))
    gates = json.loads((data / "gates.json").read_text(encoding="utf-8"))
    before_direct = (data / "before-direct-determinism.log").read_text(
        encoding="utf-8")
    direct_differences = [float(value) for value in re.findall(
        r"Which is: ([0-9.eE+-]+)\n  0\.0F", before_direct)]
    fixed_logs = sorted((data / "fixed-shape-runs").glob("*.log"))
    if verification.get("status") != "pass" or \
            verification.get("experiment") != "Exp156" or \
            verification.get("before", {}).get("baseline_revision_token_failures") != 1 or \
            verification.get("before", {}).get("numeric_shape_failures") != 18 or \
            verification.get("after", {}).get("numeric_shape_passes") != 20 or \
            verification.get("after", {}).get("direct_attention_bit_exact") != 20 or \
            verification.get("after", {}).get("full_cpu_hip_tests_passed") != 370 or \
            len(direct_differences) != 20 or \
            not 0.067 < max(direct_differences) < 0.068 or \
            "context=128 batch=8 dtype=float32" not in (
                data / "baseline-token-failure.log").read_text(encoding="utf-8") or \
            "context=127 batch=8 dtype=float32" not in (
                data / "before-numeric-failure.log").read_text(encoding="utf-8"):
        errors.append("block reduction pre-fix evidence changed")
    if len(fixed_logs) != 20 or any(
            "[  PASSED  ] 1 test." not in path.read_text(encoding="utf-8") or
            "[  FAILED  ]" in path.read_text(encoding="utf-8")
            for path in fixed_logs):
        errors.append("block reduction 20-process fixed shape gate changed")
    after_log = (data / "after-direct-and-shape.log").read_text(encoding="utf-8")
    if "Running 2 tests from 2 test suites" not in after_log or \
            "[  PASSED  ] 2 tests." not in after_log or \
            "LongBatchFusedForwardIsDeterministic" not in after_log or \
            "CpuLogitsMatchAcrossBoundaryContextBatchAndCacheDtype" not in after_log:
        errors.append("block reduction final direct/shape gate changed")
    before_performance = [json.loads(line) for line in (
        data / "before-performance.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip()]
    after_performance = [json.loads(line) for line in (
        data / "after-performance.jsonl").read_text(encoding="utf-8").splitlines()
                         if line.strip()]
    before_median = statistics.median(
        row["tokens_per_second"] for row in before_performance)
    after_median = statistics.median(
        row["tokens_per_second"] for row in after_performance)
    if len(before_performance) != 3 or len(after_performance) != 3 or \
            not 231600 < before_median < 231700 or \
            not 231900 < after_median < 232000 or \
            after_median < before_median * 0.95:
        errors.append("block reduction performance gate changed")
    decision = gates.get("decision", {})
    kernel = (REPOSITORY / "src/ops/hip/basic_kernels.hip").read_text(
        encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(
        encoding="utf-8")
    if decision.get("retain_post_read_barrier") is not True or \
            decision.get("retain_parallel_causal_gqa") is not True or \
            decision.get("performance_regression") is not False or \
            "Every lane must read the result" not in kernel or \
            kernel.count("const auto result = scratch[0]") < 3 or \
            "LongBatchFusedForwardIsDeterministic" not in tests:
        errors.append("block reduction source/decision gate changed")
    return verification.get("before", {}).get("numeric_shape_failures", -1), \
        verification.get("after", {}).get("numeric_shape_passes", -1), \
        verification.get("after", {}).get("full_cpu_hip_tests_passed", -1)


def validate_matmul_exact_registry(errors: list[str]) -> tuple[int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-matmul-exact-key"
    verification = json.loads(
        (data / "verification.json").read_text(encoding="utf-8"))
    cpu_log = (data / "cpu-key.log").read_text(encoding="utf-8")
    hip_log = (data / "hip-isolation.log").read_text(encoding="utf-8")
    header = (REPOSITORY / "include/microllm/ops/ops.h").read_text(
        encoding="utf-8")
    implementation = (REPOSITORY / "src/ops/optimized.cpp").read_text(
        encoding="utf-8")
    fields = verification.get("isolated_fields", [])
    required_fields = {
        "dtype", "transpose_layout", "strides", "architecture",
        "hip_runtime_version", "hip_driver_version", "hipblaslt_version",
        "mode", "workspace_limit",
    }
    if verification.get("status") != "pass" or set(fields) != required_fields or \
            verification.get("full_cpu_hip_tests") != "370/370" or \
            "[  PASSED  ] 1 test." not in cpu_log or \
            "[  PASSED  ] 1 test." not in hip_log or \
            "MatmulTuningKey" not in header or \
            "std::atomic<std::size_t> registry_entries" not in implementation or \
            "registry_entries.load(std::memory_order_acquire) != 0" not in implementation:
        errors.append("exact matmul registry evidence/source changed")
    return len(fields), int(verification.get("status") == "pass")


def validate_matmul_persistent_cache(errors: list[str]) -> tuple[int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-matmul-persistent-cache"
    verification = json.loads(
        (data / "verification.json").read_text(encoding="utf-8"))
    cpu_log = (data / "cpu-roundtrip.log").read_text(encoding="utf-8")
    hip_log = (data / "hip-version-filter.log").read_text(encoding="utf-8")
    implementation = (REPOSITORY / "src/ops/optimized.cpp").read_text(
        encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp"))
    contracts = (
        "cpu_roundtrip", "deterministic_serialization", "atomic_replace",
        "wrong_schema_transactional_rejection",
        "malformed_scalar_transactional_rejection",
        "duplicate_key_transactional_rejection",
        "stale_architecture_filtered", "hip_current_environment_restored",
        "hip_runtime_version_mismatch_filtered",
    )
    if verification.get("status") != "pass" or \
            any(verification.get(name) is not True for name in contracts) or \
            verification.get("full_cpu_hip_tests") != "372/372" or \
            "[  PASSED  ] 1 test." not in cpu_log or \
            "[  PASSED  ] 1 test." not in hip_log or \
            "save_matmul_tuning_cache" not in implementation or \
            "load_matmul_tuning_cache" not in implementation or \
            "std::filesystem::rename(temporary, path" not in implementation or \
            "matmul tuning cache contains a duplicate key" not in implementation or \
            "MatmulTuningCacheRoundTripsAndRejectsStaleCorruptData" not in tests or \
            "MatmulTuningCacheRestoresOnlyCurrentEnvironment" not in tests:
        errors.append("persistent matmul tuning cache evidence/source changed")
    return len(contracts), int(verification.get("status") == "pass")


def validate_matmul_correctness_before_timing(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-23-matmul-correctness-before-timing")
    verification = json.loads(
        (data / "verification.json").read_text(encoding="utf-8"))
    small = json.loads((data / "fp32-64-screen.json").read_text(encoding="utf-8"))
    accepted = json.loads((data / "fp32-128-accepted.json").read_text(
        encoding="utf-8"))
    strict = json.loads((data / "fp16-strict-rejection.json").read_text(
        encoding="utf-8"))
    cache_lines = [json.loads(line) for line in (
        data / "accepted-cache.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    if verification.get("status") != "pass" or \
            verification.get("automatic_registration") is not False or \
            verification.get("end_to_end_acceptance_external") is not True or \
            verification.get("full_cpu_hip_tests") != "375/375" or \
            small.get("accepted") is not False or \
            len(small.get("candidates", [])) != 2 or \
            any(candidate.get("correctness_passed") is not True or
                not candidate.get("event_ms_p50", 0) > 0 or
                not candidate.get("event_ms_p95", 0) >=
                    candidate.get("event_ms_p50", 0)
                for candidate in small.get("candidates", [])):
        errors.append("matmul autotune FP32 screening evidence changed")
    if accepted.get("accepted") is not True or \
            accepted.get("recommended") != "hipblaslt" or \
            len(cache_lines) != 2 or cache_lines[0].get("schema_version") != 1 or \
            cache_lines[1].get("implementation") != "hipblaslt" or \
            cache_lines[1].get("rows") != 128:
        errors.append("matmul autotune explicit acceptance/cache evidence changed")
    strict_by_name = {
        candidate["implementation"]: candidate
        for candidate in strict.get("candidates", [])
    }
    rejected = strict_by_name.get("hipblaslt", {})
    if strict.get("recommended") != "readable" or \
            strict_by_name.get("readable", {}).get("correctness_passed") is not True or \
            rejected.get("correctness_passed") is not False or \
            rejected.get("event_ms_p50") != 0.0 or \
            rejected.get("event_ms_p95") != 0.0 or \
            rejected.get("wall_ms_p50") != 0.0 or \
            rejected.get("failure") != "complete-output correctness gate failed":
        errors.append("matmul autotune correctness-before-timing rejection changed")
    source = (REPOSITORY / "src/ops/tuning.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "benchmarks/micro/tune_matmul.cpp").read_text(
        encoding="utf-8")
    if "const auto error = compare_complete" not in source or \
            source.find("const auto error = compare_complete") > \
                source.find("start.record_default_stream") or \
            "runtime::Stream" in source or \
            "register_matmul_autotune_winner" not in source or \
            "MatmulAutotuneChecksCorrectnessBeforeTiming" not in tests or \
            "end-to-end acceptance remains external" not in cli:
        errors.append("matmul autotune source/test ordering contract changed")
    return len(small.get("candidates", [])), \
        int(rejected.get("correctness_passed") is False), \
        len(cache_lines) - 1


def validate_adamw_correctness_before_timing(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-23-adamw-correctness-before-timing")
    verification = json.loads(
        (data / "verification.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    accepted = json.loads((data / "accepted-report.json").read_text(
        encoding="utf-8"))
    cache = [json.loads(line) for line in (data / "accepted-cache.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    end_to_end = [json.loads(line) for line in (data / "end-to-end.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    summaries = summary.get("summaries", [])
    measured_tps = statistics.median(
        row.get("tokens_per_second", 0) for row in end_to_end)
    unaligned = next((row for row in raw if not row.get("aligned16")), {})
    unaligned_vector = next((candidate for candidate in unaligned.get("candidates", [])
                             if candidate.get("implementation") == "vectorized"), {})
    aligned = [row for row in summaries if row.get("aligned16")]
    if verification.get("status") != "pass" or \
            verification.get("matrix", {}).get("fresh_processes") != 15 or \
            verification.get("vectorized_speedup", {}).get(
                "cases_at_or_above_1_05") != 0 or \
            verification.get("regression", {}).get("full_cpu_hip") != "378/378" or \
            verification.get("regression", {}).get("asan_ubsan") != "253/253" or \
            verification.get("regression", {}).get(
                "pytorch_enabled_cpu") != "229/229" or \
            summary.get("status") != "pass" or len(summaries) != 5 or \
            len(raw) != 15 or len(end_to_end) != 3 or len(aligned) != 4 or \
            not math.isclose(measured_tps, verification.get(
                "end_to_end", {}).get("current_tokens_per_second_p50", 0),
                rel_tol=0, abs_tol=1e-6) or \
            abs(verification.get("end_to_end", {}).get(
                "change_percent", 100)) >= 1.0 or \
            any(row.get("complete_state_passed") is not True for row in summaries) or \
            any(row.get("vectorized_speedup", 0) >= 1.05 for row in aligned):
        errors.append("AdamW complete-state matrix evidence changed")
    if unaligned_vector.get("supported") is not False or \
            unaligned_vector.get("correctness_passed") is not False or \
            unaligned_vector.get("event_ms_p50") != 0.0 or \
            unaligned_vector.get("event_ms_p95") != 0.0 or \
            unaligned_vector.get("wall_ms_p50") != 0.0 or \
            "16-byte aligned" not in unaligned_vector.get("failure", ""):
        errors.append("AdamW pre-timing alignment rejection changed")
    if accepted.get("accepted") is not True or \
            accepted.get("recommended") != "scalar" or len(cache) != 2 or \
            cache[0].get("kind") != "microllm_adamw_tuning_cache" or \
            cache[1].get("implementation") != "scalar" or \
            cache[1].get("elements") != 4099 or \
            cache[1].get("parameter_aligned16") is not False:
        errors.append("AdamW explicit acceptance/cache evidence changed")
    source = (REPOSITORY / "src/ops/adamw_tuning.cpp").read_text(encoding="utf-8")
    header = (REPOSITORY / "include/microllm/ops/tuning.h").read_text(
        encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / (
        "benchmarks/single_gpu/adamw_autotune_matrix.py")).read_text(encoding="utf-8")
    compare_position = source.find("candidate.parameter = compare_complete")
    timing_position = source.find("start.record_default_stream", compare_position)
    if compare_position < 0 or timing_position < 0 or compare_position > timing_position or \
            "std::atomic<std::size_t> adamw_registry_entries" not in source or \
            "std::filesystem::rename(temporary, path" not in source or \
            "register_adamw_autotune_winner" not in header or \
            "AdamWAutotuneChecksEveryStateBeforeTimingAndRegistration" not in tests or \
            "DEFAULT_CASES" not in runner:
        errors.append("AdamW tuner source/test ordering contract changed")
    return len(raw), sum(row.get("complete_state_passed") is True for row in summaries), \
        sum(row.get("vectorized_speedup", 0) >= 1.05 for row in aligned)


def validate_cooperative_bias_gradient(
        errors: list[str]) -> tuple[int, int, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-cooperative-bias-gradient"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    operator = json.loads((data / "operator-summary.json").read_text(encoding="utf-8"))
    operator_raw = [json.loads(line) for line in (data / "operator-raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    training = json.loads((data / "training-summary.json").read_text(encoding="utf-8"))
    baseline = [json.loads(line) for line in (data / "baseline.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    candidate = [json.loads(line) for line in (data / "candidate.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    summaries = operator.get("summaries", [])
    row16 = [row for row in summaries if row.get("rows") == 16]
    row32 = [row for row in summaries if row.get("rows") == 32]
    comparisons = training.get("comparisons", [])
    if verification.get("status") != "pass" or len(operator_raw) != 78 or \
            verification.get("regression", {}).get("full_cpu_hip") != "380/380" or \
            verification.get("regression", {}).get("hip_label") != "121/121" or \
            operator.get("raw_rows") != 78 or len(summaries) != 13 or \
            len(row16) != 1 or not row16[0].get("cooperative_speedup", 0) < 1.05 or \
            len(row32) != 4 or \
            any(row.get("cooperative_speedup", 0) < 1.05 for row in row32) or \
            any(row.get("maximum_absolute_error", 1) > 3e-5 or
                row.get("rms_error", 1) > 1e-5 for row in summaries):
        errors.append("cooperative bias-gradient operator evidence changed")
    if len(baseline) != 6 or len(candidate) != 6 or len(comparisons) != 2 or \
            any(row.get("throughput_speedup", 0) < 1.05 or
                row.get("peak_ratio") != 1.0 or
                row.get("final_loss_relative_difference", 1) > 0.005 or
                row.get("observed_parameter_after_equal") is not True
                for row in comparisons):
        errors.append("cooperative bias-gradient training gate changed")
    profile_values = profile.get("profile", {})
    if profile.get("status") != "pass" or \
            profile_values.get("baseline", {}).get("calls") != 216 or \
            profile_values.get("candidate", {}).get("calls") != 216 or \
            not profile_values.get("speedup", 0) > 6.0:
        errors.append("cooperative bias-gradient profile evidence changed")
    kernel = (REPOSITORY / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    dispatch = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    if "bias_gradient_cooperative_kernel" not in kernel or \
            "dim3(columns, row_lanes)" not in kernel or \
            "rows >= 32" not in dispatch or \
            "CooperativeBiasGradientCoversThresholdTailsAndModelWidths" not in tests:
        errors.append("cooperative bias-gradient source/test contract changed")
    return len(operator_raw), len(comparisons), float(profile_values.get("speedup", 0))


def validate_post_bias_training_profile(
        errors: list[str]) -> tuple[int, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-post-bias-training-profile"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    delta = json.loads((data / "profile-delta.json").read_text(encoding="utf-8"))
    categories = {row["category"]: row for row in delta.get("categories", [])}
    gemm = categories.get("hipBLASLt GEMM", {})
    excluded = delta.get("excluded_nonpositive_delta_names", [])
    if verification.get("status") != "pass" or delta.get("status") != "pass" or \
            delta.get("derived_steps") != 2 or len(categories) != 10 or \
            not math.isclose(delta.get("total_kernel_ns_per_step", 0), 35497419,
                             rel_tol=0, abs_tol=1) or \
            not math.isclose(gemm.get("kernel_share", 0), 0.534697818,
                             rel_tol=0, abs_tol=1e-8) or \
            not any("cast_transpose_2d" in name for name in excluded) or \
            verification.get("next_open_hypothesis") != \
                "enumerate and persist exact hipBLASLt solution indices for training GEMM shapes":
        errors.append("post-bias training phase-delta evidence changed")
    script = (REPOSITORY / "benchmarks/single_gpu/profile_step_delta.py").read_text(
        encoding="utf-8")
    if "many_step_count - 1" not in script or \
            "duration_ns_per_step" not in script or \
            "hipBLASLt GEMM" not in script:
        errors.append("training phase-delta runner contract changed")
    return len(categories), float(gemm.get("kernel_share", 0))


def validate_bf16_training_solution_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-bf16-training-solutions"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    matrix = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    training = json.loads((data / "training-comparison.json").read_text(
        encoding="utf-8"))
    summaries = matrix.get("summaries", [])
    candidate_evaluations = sum(len(row.get("candidates", [])) for row in raw)
    if verification.get("status") != "pass" or matrix.get("status") != "pass" or \
            verification.get("regression", {}).get("full_cpu_hip") != "381/381" or \
            verification.get("regression", {}).get("hip_label") != "122/122" or \
            len(raw) != 24 or len(summaries) != 8 or candidate_evaluations != 1536 or \
            any(row.get("registry_entries_after_screening") != 0 or
                row.get("passing_candidates") != 64 or
                any(candidate.get("correctness_passed") is not True or
                    candidate.get("event_ms_p50", 0) <= 0 or
                    candidate.get("wall_ms_p50", 0) <= 0
                    for candidate in row.get("candidates", []))
                for row in raw) or \
            any(row.get("common_passing_candidates") != 64 or
                row.get("maximum_absolute_error", 1) > 1e-4 or
                row.get("maximum_rms_error", 1) > 1e-5
                for row in summaries):
        errors.append("BF16 training solution operator evidence changed")
    comparisons = training.get("comparisons", [])
    if training.get("status") != "pass" or len(comparisons) != 2 or \
            training.get("policies_passing_both_models") != [] or \
            any(row.get("all_shapes_speedup", 2) >= 1.05 or
                row.get("selective_speedup", 2) >= 1.05 or
                row.get("all_shapes_parameter_equal") is not True or
                row.get("selective_parameter_equal") is not True
                for row in comparisons):
        errors.append("BF16 training solution model rejection changed")
    tuner = (REPOSITORY / "benchmarks/micro/tune_bf16_algorithms.cpp").read_text(
        encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_train_step.cpp").read_text(encoding="utf-8")
    compare_position = tuner.find("const auto actual = checked.to_vector")
    timing_position = tuner.find("start.record_default_stream", compare_position)
    if compare_position < 0 or timing_position < 0 or compare_position > timing_position or \
            "clear_bf16_algorithm_registry" not in tuner or \
            "--bf16-algorithms" not in cli or \
            "rows:inner:columns:index" not in cli:
        errors.append("BF16 training solution tuner/CLI contract changed")
    return len(raw), candidate_evaluations, len(comparisons)


def validate_tied_embedding_sparse_add(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-tied-embedding-sparse-add"
    verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row["model"]: row for row in summary.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    diagnostics = summary.get("diagnostics", {})
    dense_records = diagnostics.get("dense_qwen", {}).get(
        "gradient_accumulation", {}).get("records", [])
    sparse_records = diagnostics.get("sparse_qwen", {}).get(
        "gradient_accumulation", {}).get("records", [])
    tied_shape = [151936, 896]
    dense_tied = next((row for row in dense_records
                       if row.get("target_operation") == "leaf" and
                       row.get("shape") == tied_shape), {})
    sparse_tied = next((row for row in sparse_records
                        if row.get("target_operation") == "leaf" and
                        row.get("shape") == tied_shape), {})
    deep_sparse_calls = diagnostics.get("sparse_deepseek", {}).get(
        "gradient_accumulation", {}).get("sparse_embedding_add_calls", -1)
    profile = summary.get("profile", {})
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            verification.get("regression", {}).get("full_cpu_hip") != "387/387" or \
            verification.get("regression", {}).get("asan_ubsan") != "257/257" or \
            verification.get("regression", {}).get(
                "pytorch_enabled_cpu") != "233/233" or \
            len(training) != 12 or len(comparisons) != 2 or \
            qwen.get("peak_ratio", 1) > 0.95 or \
            qwen.get("throughput_speedup", 0) < 0.98 or \
            deep.get("throughput_speedup", 0) < 0.98 or \
            qwen.get("loss_relative_difference", 1) > 0.005 or \
            qwen.get("observed_parameter_after_equal") is not True or \
            deep.get("observed_parameter_after_equal") is not True:
        errors.append("tied embedding sparse model/memory gate changed")
    if dense_tied.get("first_source") != "matmul_right" or \
            dense_tied.get("last_add_source") != "embedding_backward" or \
            sparse_tied.get("last_add_source") != "embedding_backward_sparse_add" or \
            sparse_tied.get("sparse_embedding_add_calls") != 1 or \
            deep_sparse_calls != 0:
        errors.append("tied embedding source/routing attribution changed")
    if profile.get("add", {}).get("dense", {}).get("calls") != 507 or \
            profile.get("add", {}).get("sparse", {}).get("calls") != 504 or \
            profile.get("fill", {}).get("dense", {}).get("calls") != 586 or \
            profile.get("fill", {}).get("sparse", {}).get("calls") != 583:
        errors.append("tied embedding sparse profile changed")
    autograd = (REPOSITORY / "src/autograd/autograd.cpp").read_text(encoding="utf-8")
    ops_source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/autograd/autograd_test.cpp", "tests/ops/hip_ops_test.cpp"))
    if "storage.use_count() == 2" not in autograd or \
            "embedding_backward_add_" not in ops_source or \
            "TiedEmbeddingUsesSparseAddAfterDenseHeadGradient" not in tests or \
            "EmbeddingBackwardAddMatchesDenseReferenceWithoutTransfers" not in tests:
        errors.append("tied embedding sparse source/test contract changed")
    return len(training), int(sparse_tied.get("sparse_embedding_add_calls", 0)), \
        int(qwen.get("peak_bytes_saved", 0))


def validate_attention_interleaved_pv(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-attention-interleaved-pv"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = {row["name"]: row for row in summary.get("rows", [])}
    qwen = rows.get("qwen_t512", {})
    deep = rows.get("deepseek_t512", {})
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            verification.get("regression", {}).get("full_cpu_hip") != "393/393" or \
            verification.get("regression", {}).get("hip_label") != "125/125" or \
            len(raw) != 30 or len(rows) != 5 or \
            any(row.get("finite") is not True or
                row.get("maximum_absolute_error") != 0 or
                row.get("rms_error") != 0 or
                row.get("host_to_device_calls") != 0 or
                row.get("device_to_host_calls") != 0
                for row in raw) or \
            qwen.get("event_speedup", 0) < 1.05 or \
            deep.get("event_speedup", 0) < 1.05:
        errors.append("interleaved Attention P*V evidence changed")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    benchmark = (REPOSITORY / "benchmarks/micro/benchmark_attention_layout.cpp").read_text(
        encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp",
        "python/tests/test_operator_parity.py"))
    if "matrix_right_.set_batch(batch_count, width)" not in source or \
            "matrix_output_.set_batch" not in source or \
            "Attention layout complete-output gate failed" not in benchmark or \
            "AttentionProbabilityValueInterleavedLayoutMatchesCpuWithoutTransfers" not in tests or \
            "attention_probability_value_bthd" not in tests:
        errors.append("interleaved Attention P*V source/test contract changed")
    return len(raw), len(rows), float(qwen.get("event_speedup", 0)), \
        float(deep.get("event_speedup", 0))


def validate_attention_context_layout_fusion(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-attention-context-layout-fusion"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row["model"]: row for row in summary.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            verification.get("regression", {}).get("full_cpu_hip") != "397/397" or \
            verification.get("regression", {}).get("hip_label") != "126/126" or \
            verification.get("regression", {}).get("pytorch_enabled_cpu") != "241/241" or \
            len(training) != 12 or len(comparisons) != 2 or \
            qwen.get("throughput_speedup", 0) < 0.98 or \
            deep.get("throughput_speedup", 0) < 0.98 or \
            qwen.get("strided_copy", {}).get("fused", {}).get("calls") != 0 or \
            deep.get("strided_copy", {}).get("fused", {}).get("calls") != 0 or \
            qwen.get("strided_copy", {}).get("fused", {}).get("bytes") != 0 or \
            deep.get("strided_copy", {}).get("fused", {}).get("bytes") != 0 or \
            qwen.get("loss_relative_difference", 1) > 0.005 or \
            deep.get("loss_relative_difference", 1) > 0.005 or \
            qwen.get("observed_parameter_after_equal") is not True or \
            deep.get("observed_parameter_after_equal") is not True:
        errors.append("complete Attention context layout model gate changed")
    if profile.get("materialized", {}).get("strided_copy_calls") != 288 or \
            profile.get("fused", {}).get("strided_copy_calls") != 0 or \
            profile.get("fused", {}).get("kernel_dispatches", 10000) >= \
            profile.get("materialized", {}).get("kernel_dispatches", 0):
        errors.append("complete Attention context layout profile changed")
    ops_source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    model_source = (REPOSITORY / "src/model/model.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/graph/graph_gradient_alignment_test.cpp",
        "tests/ops/hip_ops_test.cpp", "python/tests/test_operator_parity.py"))
    if "causal_gqa_attention_bthd_backward_saved" not in ops_source or \
            "attention_context_layout_fusion_enabled" not in model_source or \
            "BthdCausalGqaMatchesComposedGraphAndAllGradients" not in tests or \
            "LongCausalGqaBthdForwardBackwardMatchCpuWithoutTransfers" not in tests or \
            "graph_causal_gqa_bthd_value_grad" not in tests:
        errors.append("complete Attention context layout source/test contract changed")
    return len(training), int(qwen.get("strided_copy", {}).get("fused", {}).get("calls", -1)), \
        int(deep.get("strided_copy", {}).get("fused", {}).get("calls", -1))


def validate_post_layout_training_profile(
        errors: list[str]) -> tuple[int, float, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-post-layout-training-profile"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    delta = json.loads((data / "profile-delta.json").read_text(encoding="utf-8"))
    categories = delta.get("categories", [])
    by_name = {row["category"]: row for row in categories}
    gemm = by_name.get("hipBLASLt GEMM", {})
    next_hypothesis = verification.get("next_open_hypothesis", {})
    if verification.get("status") != "pass" or delta.get("status") != "pass" or \
            delta.get("derived_steps") != 2 or len(categories) != 9 or \
            delta.get("total_kernel_ns_per_step") != 33349282.0 or \
            not math.isclose(gemm.get("kernel_share", 0), 0.565474993,
                             rel_tol=0, abs_tol=1e-9) or \
            "strided materialization" in by_name or \
            next_hypothesis.get("qwen_interleaved_calls_per_step") != 72 or \
            next_hypothesis.get("deepseek_interleaved_calls_per_step") != 84 or \
            next_hypothesis.get("layouts_created_per_call") != 3:
        errors.append("post-layout training phase profile changed")
    excluded = delta.get("excluded_nonpositive_delta_names", [])
    if not any("cast_transpose_2d" in name for name in excluded):
        errors.append("post-layout load-only exclusion changed")
    return len(categories), float(gemm.get("kernel_share", 0)), \
        int(delta.get("total_kernel_ns_per_step", 0))


def validate_attention_layout_plan_cache_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-attention-layout-plan-cache"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    operator = json.loads((data / "operator-summary.json").read_text(
        encoding="utf-8"))
    operator_raw = [json.loads(line) for line in (data / "operator-raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    model = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    route = [json.loads(line) for line in (data / "route-smoke/training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    operator_rows = {row["name"]: row for row in operator.get("rows", [])}
    comparisons = {row["model"]: row for row in model.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    cached_route = [row for row in route if row.get("attention_layout_plan_cache") is True]
    if verification.get("status") != "pass" or operator.get("status") != "pass" or \
            model.get("status") != "pass" or model.get("decision") != "reject plan cache" or \
            verification.get("defaults") != {
                "engine": False, "training_cli": False,
                "operator_benchmark": False} or \
            verification.get("regression", {}).get("full_cpu_hip") != "399/399" or \
            verification.get("regression", {}).get("hip_label") != "127/127" or \
            len(operator_raw) != 24 or len(operator_rows) != 4 or len(training) != 12 or \
            len(cached_route) != 2 or \
            operator_rows.get("qwen_t512", {}).get("wall_speedup", 0) < 1.01 or \
            operator_rows.get("deepseek_t512", {}).get("wall_speedup", 0) < 1.01 or \
            qwen.get("throughput_speedup", 2) >= 1.01 or \
            deep.get("throughput_speedup", 2) >= 1.01 or \
            any(row.get("maximum_absolute_error") != 0 or row.get("rms_error") != 0
                for row in operator_raw) or \
            sorted((row.get("attention_layout_plan_cache_entries"),
                    row.get("attention_layout_plan_cache_misses"),
                    row.get("attention_layout_plan_cache_hits"))
                   for row in cached_route) != [(3, 3, 69), (3, 3, 81)]:
        errors.append("Attention layout plan-cache discard evidence changed")
    optimized = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_train_step.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    if "attention_layout_cache_enabled = false" not in optimized or \
            "bool attention_layout_plan_cache = false" not in cli or \
            "ExactModesAndShapesHitWithoutChangingOutputs" not in tests or \
            "AttentionLayoutPlanKey" not in optimized:
        errors.append("Attention layout plan-cache default/source contract changed")
    return len(operator_raw), len(training), len(route)


def validate_attention_gemm_scale_fusion_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-attention-gemm-scale-fusion"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row["model"]: row for row in summary.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            summary.get("decision") != "reject GEMM scale fusion" or \
            verification.get("defaults") != {
                "engine_attention_policy": False, "training_cli": False} or \
            verification.get("regression", {}).get("full_cpu_hip") != "401/401" or \
            verification.get("regression", {}).get("hip_label") != "128/128" or \
            len(training) != 12 or len(comparisons) != 2 or \
            qwen.get("throughput_speedup", 2) >= 1.01 or \
            deep.get("throughput_speedup", 0) < 1.01 or \
            qwen.get("observed_parameter_after_equal") is not True or \
            deep.get("observed_parameter_after_equal") is not False or \
            qwen.get("allocation_calls_saved") != 96 or \
            deep.get("allocation_calls_saved") != 112 or \
            profile.get("explicit", {}).get("scale_calls") != 144 or \
            profile.get("fused", {}).get("scale_calls") != 0:
        errors.append("Attention GEMM scale-fusion rejection evidence changed")
    ops_source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    optimized = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_train_step.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp",
        "python/tests/test_operator_parity.py"))
    if "attention_gemm_scale_fusion = false" not in ops_source or \
            "bool attention_gemm_scale_fusion = false" not in cli or \
            "matmul_scaled_with_implementation" not in optimized or \
            "ScaledHipblasLtMatmulUsesAlphaWithoutPayloadTransfers" not in tests or \
            "invalid_matmul_scaled_factor" not in tests:
        errors.append("Attention GEMM scale-fusion default/source contract changed")
    return len(training), int(qwen.get("allocation_calls_saved", 0)), \
        int(deep.get("allocation_calls_saved", 0))


def validate_paired_gqa_repeat_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-paired-gqa-repeat"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row["model"]: row for row in summary.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    separate_calls = (profile.get("separate", {}).get("repeat_forward_calls", 0) +
                      profile.get("separate", {}).get("repeat_backward_calls", 0))
    paired_calls = (profile.get("paired", {}).get("repeat_forward_calls", 0) +
                    profile.get("paired", {}).get("repeat_backward_calls", 0))
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            summary.get("decision") != "reject paired GQA repeat" or \
            verification.get("defaults") != {"engine": False, "training_cli": False} or \
            verification.get("regression", {}).get("full_cpu_hip") != "403/403" or \
            verification.get("regression", {}).get("hip_label") != "129/129" or \
            len(training) != 12 or len(comparisons) != 2 or \
            qwen.get("throughput_speedup", 2) >= 1.01 or \
            deep.get("throughput_speedup", 2) >= 1.01 or \
            qwen.get("observed_parameter_after_equal") is not True or \
            deep.get("observed_parameter_after_equal") is not True or \
            separate_calls != 432 or paired_calls != 216 or \
            profile.get("paired", {}).get("kernel_dispatches", 10000) >= \
            profile.get("separate", {}).get("kernel_dispatches", 0):
        errors.append("paired GQA repeat rejection evidence changed")
    source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_train_step.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp",
        "python/tests/test_operator_parity.py"))
    if "attention_paired_gqa_repeat = false" not in source or \
            "bool attention_paired_gqa_repeat = false" not in cli or \
            "repeat_gqa_kv_bthd_backward" not in source or \
            "PairedGqaRepeatForwardBackwardMatchSeparateWithoutTransfers" not in tests or \
            "repeat_gqa_kv_bthd_backward_key" not in tests:
        errors.append("paired GQA repeat default/source contract changed")
    return len(training), separate_calls, paired_calls


def validate_gqa_zero_stride_value_broadcast(
        errors: list[str]) -> tuple[int, int, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-attention-gqa-zero-stride-broadcast"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = {row["name"]: row for row in summary.get("rows", [])}
    qwen = rows.get("qwen_t512", {})
    deep = rows.get("deepseek_t512", {})
    mha = rows.get("mha_counterexample", {})
    if verification.get("status") != "pass" or summary.get("status") != "reject" or \
            verification.get("model_default_changed") is not False or \
            verification.get("regression", {}).get("full_cpu_hip") != "405/405" or \
            verification.get("regression", {}).get("hip_label") != "130/130" or \
            len(raw) != 30 or len(rows) != 5 or \
            qwen.get("wall_speedup", 2) >= 1.0 or \
            deep.get("wall_speedup", 0) < 1.5 or \
            mha.get("wall_speedup", 2) >= 1.0 or \
            any(row.get("maximum_absolute_error", 1) > 1e-6 or
                row.get("rms_error", 1) > 1e-7 or
                row.get("host_to_device_calls") != 0 or
                row.get("device_to_host_calls") != 0 for row in raw):
        errors.append("GQA zero-stride Value broadcast evidence changed")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp",
        "python/tests/test_operator_parity.py"))
    if "matrix_value.set_batch(batch_count, 0)" not in source or \
            "GqaProbabilityValueZeroStrideBroadcastMatchesCpuForBatchTwo" not in tests or \
            "attention_probability_value_gqa_bthd" not in tests:
        errors.append("GQA zero-stride Value source/test contract changed")
    return len(raw), len(rows), float(deep.get("wall_speedup", 0))


def validate_selective_gqa_value_broadcast_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-selective-gqa-value-broadcast"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row["model"]: row for row in summary.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            summary.get("decision") != "reject GQA Value broadcast" or \
            verification.get("defaults") != {"engine": False, "training_cli": False} or \
            verification.get("regression", {}).get("full_cpu_hip") != "406/406" or \
            verification.get("regression", {}).get("hip_label") != "131/131" or \
            len(training) != 12 or len(comparisons) != 2 or \
            qwen.get("allocation_calls_saved") != 0 or \
            deep.get("allocation_calls_saved") != 112 or \
            qwen.get("throughput_speedup", 2) >= 1.01 or \
            deep.get("throughput_speedup", 2) >= 1.01 or \
            profile.get("baseline", {}).get("repeat_forward_calls") != 336 or \
            profile.get("broadcast", {}).get("repeat_forward_calls") != 168 or \
            profile.get("baseline", {}).get("kernel_dispatches") != \
            profile.get("broadcast", {}).get("kernel_dispatches") or \
            profile.get("broadcast", {}).get("kernel_time_ns", 0) <= \
            profile.get("baseline", {}).get("kernel_time_ns", 1):
        errors.append("selective GQA Value broadcast rejection evidence changed")
    source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_train_step.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp",
        "python/tests/test_operator_parity.py"))
    if "attention_gqa_value_broadcast = false" not in source or \
            "bool attention_gqa_value_broadcast = false" not in cli or \
            "query.shape()[3] >= 128" not in source or \
            "WideGqaValueBroadcastFullSavedPathMatchesExpandedControl" not in tests or \
            "attention_probability_gradient_gqa_bthd" not in tests:
        errors.append("selective GQA Value broadcast default/source contract changed")
    return len(training), int(qwen.get("allocation_calls_saved", -1)), \
        int(deep.get("allocation_calls_saved", -1))


def validate_forward_only_gqa_value_broadcast_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-23-forward-only-gqa-value-broadcast"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row["model"]: row for row in summary.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            summary.get("decision") != "reject forward-only GQA Value broadcast" or \
            verification.get("zero_stride_model_search_closed") is not True or \
            verification.get("regression", {}).get("full_cpu_hip") != "406/406" or \
            verification.get("regression", {}).get("hip_label") != "131/131" or \
            len(training) != 12 or len(comparisons) != 2 or \
            qwen.get("allocation_calls_saved") != 0 or \
            deep.get("allocation_calls_saved") != 56 or \
            qwen.get("throughput_speedup", 2) >= 1.01 or \
            deep.get("throughput_speedup", 2) >= 1.01 or \
            deep.get("observed_parameter_after_equal") is not False or \
            profile.get("baseline", {}).get("repeat_forward_calls") != 336 or \
            profile.get("forward_only", {}).get("repeat_forward_calls") != 252 or \
            profile.get("baseline", {}).get("kernel_dispatches") != \
            profile.get("forward_only", {}).get("kernel_dispatches") or \
            profile.get("forward_only", {}).get("kernel_time_ns", 0) <= \
            profile.get("baseline", {}).get("kernel_time_ns", 1):
        errors.append("forward-only GQA Value broadcast rejection evidence changed")
    source = (REPOSITORY / "src/ops/ops.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_train_step.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    if "attention_gqa_forward_value_broadcast = false" not in source or \
            "bool attention_gqa_forward_value_broadcast = false" not in cli or \
            "forward_only_gradients" not in tests:
        errors.append("forward-only GQA Value broadcast default/source contract changed")
    return len(training), int(qwen.get("allocation_calls_saved", -1)), \
        int(deep.get("allocation_calls_saved", -1))


def validate_unique_gradient_inplace_add_discard(
        errors: list[str]) -> tuple[int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-unique-gradient-inplace-add"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    training = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row["model"]: row for row in summary.get("comparisons", [])}
    qwen = comparisons.get("qwen2.5-0.5b", {})
    deep = comparisons.get("deepseek-r1-distill-qwen-1.5b", {})
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    allocating = profile.get("policies", {}).get("allocating", {})
    inplace = profile.get("policies", {}).get("inplace", {})
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            summary.get("decision") != "reject unique-gradient in-place accumulation" or \
            len(training) != 12 or len(comparisons) != 2 or \
            qwen.get("allocation_calls_saved") != 144 or \
            deep.get("allocation_calls_saved") != 168 or \
            qwen.get("throughput_speedup", 2) >= 1.01 or \
            deep.get("throughput_speedup", 2) >= 1.01 or \
            allocating.get("kernel_calls") != inplace.get("kernel_calls") or \
            allocating.get("add_kernel_calls") != inplace.get("add_kernel_calls") or \
            allocating.get("engine_backend_allocation_calls") != \
            inplace.get("engine_backend_allocation_calls"):
        errors.append("unique-gradient in-place rejection evidence changed")
    source = (REPOSITORY / "src/autograd/autograd.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_train_step.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/autograd/autograd_test.cpp", "tests/ops/ops_test.cpp",
        "tests/ops/hip_ops_test.cpp"))
    if "thread_local bool unique_gradient_inplace_add = false" not in source or \
            "bool unique_gradient_inplace_add = false" not in cli or \
            "DenseAddEligibilityRequiresUniqueDestinationStorage" not in tests or \
            "InPlaceAddPreservesStorageWithoutPayloadTransfers" not in tests:
        errors.append("unique-gradient in-place source/test contract changed")
    return len(training), int(qwen.get("allocation_calls_saved", -1)), \
        int(deep.get("allocation_calls_saved", -1))


def validate_hip_graph_runtime(
        errors: list[str]) -> tuple[int, int, float, float, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-hip-graph-runtime"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    matrix = [json.loads(line) for line in (data / "matrix.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    accepted = [row for row in comparisons if row.get("nodes", 0) >= 32]
    rejected = [row for row in comparisons if row.get("nodes") in (1, 8)]
    minimum = min((float(row.get("wall_speedup", 0)) for row in accepted),
                  default=0.0)
    maximum = max((float(row.get("wall_speedup", 0)) for row in accepted),
                  default=0.0)
    eager = profile.get("policies", {}).get("eager", {})
    graph = profile.get("policies", {}).get("graph", {})
    api_saved = int(profile.get("delta", {}).get("hip_api_calls_saved", -1))
    if verification.get("status") != "pass" or summary.get("status") != "pass" or \
            summary.get("decision") != "keep caller-owned HIP Graph runtime primitive" or \
            len(matrix) != 60 or len(comparisons) != 10 or len(accepted) != 6 or \
            len(rejected) != 4 or minimum < 1.05 or \
            any(float(row.get("wall_speedup", 2)) >= 1.0 for row in rejected) or \
            any(row.get("maximum_absolute_error") != 0 or
                row.get("host_to_device_calls") != 0 or
                row.get("device_to_host_calls") != 0 or
                row.get("device_to_device_calls") != 0 for row in matrix) or \
            eager.get("kernel_calls") != graph.get("kernel_calls") or \
            eager.get("hip_launch_kernel_calls") != 2580 or \
            graph.get("hip_launch_kernel_calls") != 129 or \
            graph.get("hip_graph_launch_calls") != 20 or api_saved != 12188:
        errors.append("HIP Graph runtime evidence changed")
    runtime_header = (REPOSITORY / "include/microllm/runtime/runtime.h").read_text(
        encoding="utf-8")
    runtime_source = (REPOSITORY / "src/runtime/runtime.cpp").read_text(
        encoding="utf-8")
    tests = (REPOSITORY / "tests/runtime/runtime_test.cpp").read_text(
        encoding="utf-8")
    benchmark = (REPOSITORY / "benchmarks/micro/benchmark_hip_graph.cpp").read_text(
        encoding="utf-8")
    if "class HipGraphExecutable" not in runtime_header or \
            "hipStreamBeginCapture" not in runtime_source or \
            "hipGraphInstantiate" not in runtime_source or \
            "hipGetLastError" not in runtime_source or \
            "CapturesReplaysAndMovesCallerOwnedOperatorChain" not in tests or \
            "captured_nodes" not in benchmark:
        errors.append("HIP Graph runtime source/test contract changed")
    return len(matrix), len(comparisons), minimum, maximum, api_saved


def validate_hip_graph_gemm_discard(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-hip-graph-gemm"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    matrix = [json.loads(line) for line in (data / "matrix.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    indexed = {(row.get("shape_name"), row.get("calls")): row
               for row in comparisons}
    qwen32 = float(indexed.get(("qwen", 32), {}).get("wall_speedup", 0))
    deep32 = float(indexed.get(("deepseek", 32), {}).get("wall_speedup", 2))
    eager = profile.get("policies", {}).get("eager", {})
    graph = profile.get("policies", {}).get("graph", {})
    if verification.get("status") != "pass" or \
            verification.get("tests", {}).get("hip_full_configuration", {}).get("total") != 420 or \
            summary.get("status") != "pass" or \
            summary.get("decision") != "reject caller-owned hipBLASLt Graph boundary" or \
            len(matrix) != 36 or len(comparisons) != 6 or \
            qwen32 < 1.02 or deep32 >= 1.0 or \
            any(row.get("maximum_absolute_error") != 0 or
                row.get("rms_error") != 0 or
                row.get("output_address_stable") is not True or
                row.get("host_to_device_calls") != 0 or
                row.get("device_to_host_calls") != 0 or
                row.get("device_to_device_calls") != 0 for row in matrix) or \
            eager.get("kernel_calls") != graph.get("kernel_calls") or \
            eager.get("hip_ext_module_launch_calls") != 321 or \
            graph.get("hip_ext_module_launch_calls") != 33 or \
            graph.get("hip_graph_launch_calls") != 10:
        errors.append("HIP Graph GEMM rejection evidence changed")
    header = (REPOSITORY / "include/microllm/ops/ops.h").read_text(encoding="utf-8")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp"))
    if "void matmul_out_" not in header or \
            "void hipblaslt_matmul_out" not in source or \
            "MatmulOutPreservesCallerStorageAndChecksAliases" not in tests or \
            "CallerOwnedHipblasLtOutputCapturesAndReplays" not in tests:
        errors.append("HIP Graph GEMM source/test contract changed")
    return len(matrix), len(comparisons), qwen32, deep32


def validate_scoped_model_stream_discard(
        errors: list[str]) -> tuple[int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-scoped-model-stream-discard"
    failure = json.loads((data / "failure.json").read_text(encoding="utf-8"))
    repetitions = failure.get("repetitions", [])
    maxima = [float(row.get("maximum_absolute_logit_difference", 0))
              for row in repetitions]
    rms = [float(row.get("rms_logit_difference", 0)) for row in repetitions]
    if failure.get("status") != "stable_failure" or len(repetitions) != 3 or \
            min(maxima, default=0) < 1.4 or max(maxima, default=0) < 3.8 or \
            min(rms, default=0) < 0.47 or \
            failure.get("all_64_logits_within_1e_5") is not False or \
            failure.get("capture_failure_recovery", {}).get("status") != "fail" or \
            "previous error during capture" not in \
            failure.get("capture_failure_recovery", {}).get("next_error", ""):
        errors.append("scoped model Stream failure evidence changed")
    context = (REPOSITORY / "include/microllm/ops/context.h").read_text(
        encoding="utf-8")
    tests = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "tests/ops/ops_test.cpp", "tests/ops/hip_ops_test.cpp",
        "tests/graph/hip_graph_alignment_test.cpp"))
    package = (REPOSITORY / "tests/package/consumer/main.cpp").read_text(
        encoding="utf-8")
    if "ScopedOpStream" in context or "current_op_stream" in context or \
            "ScopedStreamRoutes" in tests or "ScopedOpStream" in package:
        errors.append("unsafe scoped model Stream candidate returned")
    return len(repetitions), max(maxima, default=0), max(rms, default=0)


def validate_deferred_hip_deallocation(
        errors: list[str]) -> tuple[int, int, float, float, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-deferred-hip-deallocation"
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    matrix = [json.loads(line) for line in (data / "matrix.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    speedups = [float(row.get("wall_speedup", 0)) for row in comparisons]
    deferred_bytes = [
        int(row.get("policies", {}).get("deferred", {}).get("deferred_bytes", -1))
        for row in comparisons]
    immediate = profile.get("policies", {}).get("immediate_sync", {})
    deferred = profile.get("policies", {}).get("deferred", {})
    if verification.get("status") != "pass" or \
            verification.get("tests", {}).get("hip_full_configuration", {}).get("total") != 428 or \
            summary.get("status") != "pass" or \
            summary.get("decision") != "keep explicit deferred HIP deallocation scope" or \
            len(matrix) != 36 or len(comparisons) != 6 or \
            min(speedups, default=0) < 2.28 or max(speedups, default=0) < 2.73 or \
            max(deferred_bytes, default=0) != 2080768 or \
            any(row.get("maximum_absolute_error") != 0 or
                row.get("host_to_device_calls") != 0 or
                row.get("device_to_host_calls") != 0 or
                row.get("device_to_device_calls") != 0 for row in matrix) or \
            immediate.get("kernel_calls") != deferred.get("kernel_calls") or \
            immediate.get("stream_synchronize_calls") != 320 or \
            deferred.get("stream_synchronize_calls") != 10 or \
            immediate.get("hip_malloc_calls") != deferred.get("hip_malloc_calls") or \
            immediate.get("hip_free_calls") != deferred.get("hip_free_calls"):
        errors.append("deferred HIP deallocation evidence changed")
    header = (REPOSITORY / "include/microllm/runtime/runtime.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/runtime/runtime.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/runtime/runtime_test.cpp").read_text(encoding="utf-8")
    if "class DeferredHipDeallocationScope" not in header or \
            "active_deferred_scope" not in source or \
            "KeepsTemporaryChainAliveUntilOneStreamSync" not in tests or \
            "CapacityOverflowFlushesSafelyAndContinues" not in tests:
        errors.append("deferred HIP deallocation source/test contract changed")
    return len(matrix), len(comparisons), min(speedups, default=0.0), \
        max(speedups, default=0.0), max(deferred_bytes, default=0)


def validate_scoped_deferred_model_stream(
        errors: list[str]) -> tuple[int, int, float, float, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-scoped-deferred-model-stream"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    pairs = [json.loads(line) for line in (data / "pairs.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    ratios = [float(row.get("deferred_speedup", 0.0)) for row in comparisons]
    maximum_bytes = max((int(row.get("maximum_deferred_bytes", 0))
                         for row in comparisons), default=0)
    legacy = profile.get("policies", {}).get("legacy", {})
    deferred = profile.get("policies", {}).get("deferred", {})
    if verification.get("status") != "pass_with_preexisting_rccl_failure" or \
            verification.get("tests", {}).get("cpu_debug") != \
                    {"passed": 281, "total": 281} or \
            verification.get("tests", {}).get("asan_ubsan") != \
                    {"passed": 279, "total": 279} or \
            verification.get("tests", {}).get("pytorch_enabled_cpu") != \
                    {"passed": 255, "total": 255} or \
            verification.get("tests", {}).get("hip_label") != \
                    {"passed": 146, "total": 146} or \
            verification.get("tests", {}).get("rccl_multi_gpu") != \
                    {"passed": 6, "total": 11} or \
            verification.get("rccl_baseline_check", {}).get("revision") != "adcd642" or \
            summary.get("status") != "pass" or \
            summary.get("raw_processes") != 48 or len(raw) != 48 or \
            len(pairs) != 24 or len(comparisons) != 8 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not False or \
            summary.get("decision") != "keep safe infrastructure; default off" or \
            not (0.12 <= min(ratios, default=0.0) <= 0.13) or \
            not (0.86 <= max(ratios, default=0.0) <= 0.87) or \
            any(ratio >= 1.0 for ratio in ratios) or \
            maximum_bytes != 15591456776 or \
            any(row.get("maximum_absolute_error", 0.0) != 0.0 or
                row.get("rms_error", 0.0) != 0.0 or
                row.get("loss_absolute_difference", 0.0) != 0.0 or
                row.get("parameter_absolute_difference", 0.0) != 0.0
                for row in pairs) or \
            any(row.get("deferred_overflow_flushes", -1) != 0
                for row in raw if row.get("policy") == "deferred") or \
            profile.get("status") != "pass" or \
            legacy.get("kernel_calls") != deferred.get("kernel_calls") or \
            legacy.get("hip_launch_kernel_calls") != \
                    deferred.get("hip_launch_kernel_calls") or \
            legacy.get("hip_ext_launch_kernel_calls") != \
                    deferred.get("hip_ext_launch_kernel_calls") or \
            deferred.get("hip_malloc_calls", 0) <= legacy.get("hip_malloc_calls", 0) or \
            deferred.get("hip_free_calls", 0) <= legacy.get("hip_free_calls", 0):
        errors.append("scoped deferred model Stream evidence changed")
    header = (REPOSITORY / "include/microllm/runtime/runtime.h").read_text(
        encoding="utf-8")
    runtime = (REPOSITORY / "src/runtime/runtime.cpp").read_text(encoding="utf-8")
    context = (REPOSITORY / "include/microllm/ops/context.h").read_text(
        encoding="utf-8")
    tests = (REPOSITORY / "tests/graph/hip_graph_alignment_test.cpp").read_text(
        encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "scoped_deferred_model_matrix.py").read_text(encoding="utf-8")
    if "class ScopedDeferredHipStream" not in header or \
            "active_scoped_deferred_stream" not in runtime or \
            "resolve_deferred_hip_stream" not in context or \
            "DeferredScopedStreamRestoresCompleteInferenceLogits" not in tests or \
            "DeferredScopedStreamMatchesForwardBackwardGradients" not in tests or \
            "inference_error" not in runner:
        errors.append("scoped deferred model Stream source/test contract changed")
    return len(raw), len(pairs), min(ratios, default=0.0), \
        max(ratios, default=0.0), maximum_bytes


def validate_per_device_hipblaslt_handles(
        errors: list[str]) -> tuple[int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-per-device-hipblaslt-handles"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    ratios = [float(row.get("throughput_ratio", 0.0)) for row in comparisons]
    if summary.get("status") != "pass" or len(raw) != 12 or \
            summary.get("raw_processes") != 12 or len(comparisons) != 4 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not True or \
            summary.get("decision") != "keep per-device handles" or \
            not (0.997 <= min(ratios, default=0.0) <= 0.999) or \
            not (1.022 <= max(ratios, default=0.0) <= 1.024) or \
            any(row.get("output_contract_exact") is not True
                for row in comparisons) or \
            verification.get("status") != "pass" or \
            verification.get("tests", {}).get("rccl_multi_gpu") != \
                    {"passed": 11, "total": 11} or \
            verification.get("tests", {}).get("hip_full_configuration") != \
                    {"passed": 436, "total": 436, "conditional_skips": 3}:
        errors.append("per-device hipBLASLt handle evidence changed")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "per_device_handle_regression.py").read_text(encoding="utf-8")
    if "Handle& handle_for_device(Device device)" not in source or \
            "static Handle handle" in source or \
            "PerDeviceHandlesSurviveAlternatingGpuMatmuls" not in tests or \
            "minimum_throughput_ratio" not in runner:
        errors.append("per-device hipBLASLt handle source/test contract changed")
    return len(raw), min(ratios, default=0.0), max(ratios, default=0.0)


def validate_stream_ordered_allocator(
        errors: list[str]) -> tuple[int, float, float, float, float, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-stream-ordered-allocator"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    async_ratios = [float(row.get("async_speedup_vs_deferred", 0.0))
                    for row in comparisons]
    graph_ratios = [float(row.get("graph_speedup_vs_deferred", 0.0))
                    for row in comparisons]
    maximum_pool = max((int(row.get("async_pool_reserved_high_bytes", 0))
                        for row in comparisons), default=0)
    modes = profile.get("modes", {})
    deferred = modes.get("deferred", {})
    async_mode = modes.get("async", {})
    graph = modes.get("graph", {})
    if summary.get("status") != "pass" or len(raw) != 72 or \
            summary.get("raw_processes") != 72 or len(comparisons) != 8 or \
            summary.get("correctness_gate") is not True or \
            summary.get("address_contract") is not True or \
            summary.get("async_performance_gate") is not False or \
            summary.get("graph_performance_gate") is not False or \
            summary.get("decision") != \
                    "keep explicit primitive; reject eager and graph policies" or \
            not (0.61 <= min(async_ratios, default=0.0) <= 0.63) or \
            not (0.70 <= max(async_ratios, default=0.0) <= 0.72) or \
            not (0.03 <= min(graph_ratios, default=0.0) <= 0.04) or \
            not (0.04 <= max(graph_ratios, default=0.0) <= 0.05) or \
            maximum_pool != 134217728 or \
            any(row.get("async_unique_addresses") != 2 or
                row.get("graph_unique_addresses") != row.get("nodes") or
                row.get("graph_node_count") != row.get("nodes") * 3 + 1
                for row in comparisons) or \
            profile.get("status") != "pass" or \
            deferred.get("kernel_calls") != async_mode.get("kernel_calls") or \
            deferred.get("kernel_calls") != graph.get("kernel_calls") or \
            graph.get("host_kernel_launch_calls", 0) >= \
                    deferred.get("host_kernel_launch_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests", {}).get("hip_full_configuration") != \
                    {"passed": 442, "total": 442, "conditional_skips": 3}:
        errors.append("Stream ordered allocator evidence changed")
    header = (REPOSITORY / "include/microllm/runtime/runtime.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/runtime/runtime.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/runtime/runtime_test.cpp").read_text(
        encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "stream_ordered_allocator_matrix.py").read_text(encoding="utf-8")
    if "class StreamOrderedHipBuffer" not in header or \
            "hipMallocAsync" not in source or "hipFreeAsync" not in source or \
            "CaptureCreatesSafeAllocationAndFreeNodes" not in tests or \
            "graph_node_count" not in runner:
        errors.append("Stream ordered allocator source/test contract changed")
    return len(raw), min(async_ratios, default=0.0), \
        max(async_ratios, default=0.0), min(graph_ratios, default=0.0), \
        max(graph_ratios, default=0.0), maximum_pool


def validate_activation_arena(
        errors: list[str]) -> tuple[int, float, float, float, float, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-activation-arena"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    eager = [float(row.get("arena_speedup", 0.0)) for row in comparisons]
    graph = [float(row.get("arena_graph_speedup", 0.0)) for row in comparisons]
    breaks = [int(row.get("arena_graph_break_even_replays", 0))
              for row in comparisons]
    modes = profile.get("modes", {})
    deferred_profile = modes.get("deferred", {})
    arena_profile = modes.get("arena", {})
    graph_profile = modes.get("arena_graph", {})
    if summary.get("status") != "pass" or len(raw) != 72 or \
            summary.get("raw_processes") != 72 or len(comparisons) != 8 or \
            summary.get("correctness_gate") is not True or \
            summary.get("layout_contract") is not True or \
            summary.get("arena_performance_gate") is not True or \
            summary.get("arena_graph_performance_gate") is not True or \
            summary.get("decision") != "keep arena and arena Graph candidate" or \
            not (1.07 <= min(eager, default=0.0) <= 1.08) or \
            not (1.76 <= max(eager, default=0.0) <= 1.78) or \
            not (1.31 <= min(graph, default=0.0) <= 1.32) or \
            not (3.06 <= max(graph, default=0.0) <= 3.07) or \
            min(breaks, default=0) != 9 or max(breaks, default=0) != 1280 or \
            any(row.get("arena_unique_addresses") != 2 or
                row.get("arena_graph_unique_addresses") != 2 or
                row.get("arena_graph_node_count") != row.get("nodes") + 1 or
                row.get("arena_capacity_bytes") !=
                    row.get("expected_arena_capacity_bytes")
                for row in comparisons) or \
            profile.get("status") != "pass" or \
            deferred_profile.get("kernel_calls") != arena_profile.get("kernel_calls") or \
            deferred_profile.get("kernel_calls") != graph_profile.get("kernel_calls") or \
            arena_profile.get("hip_malloc_calls", 0) >= \
                    deferred_profile.get("hip_malloc_calls", 0) or \
            graph_profile.get("host_kernel_launch_calls", 0) >= \
                    arena_profile.get("host_kernel_launch_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests", {}).get("hip_full_configuration") != \
                    {"passed": 447, "total": 447, "conditional_skips": 3}:
        errors.append("activation arena evidence changed")
    header = (REPOSITORY / "include/microllm/runtime/runtime.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/runtime/runtime.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/runtime/runtime_test.cpp").read_text(
        encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "activation_arena_matrix.py").read_text(encoding="utf-8")
    if "class HipActivationArena" not in header or \
            "HipActivationArena::allocate_slice" not in source or \
            "GraphReplaysStableTwoSlotLivenessPlan" not in tests or \
            "arena_graph_break_even_replays" not in runner:
        errors.append("activation arena source/test contract changed")
    return len(raw), min(eager, default=0.0), max(eager, default=0.0), \
        min(graph, default=0.0), max(graph, default=0.0), \
        min(breaks, default=0), max(breaks, default=0)


def validate_arena_ffn(
        errors: list[str]) -> tuple[int, float, float, float, float, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-arena-ffn"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    eager = [float(row.get("arena_speedup", 0.0)) for row in comparisons]
    graph = [float(row.get("arena_graph_speedup", 0.0)) for row in comparisons]
    breaks = [int(row.get("graph_break_even_replays", 0))
              for row in comparisons]
    modes = profile.get("modes", {})
    deferred_profile = modes.get("deferred", {})
    arena_profile = modes.get("arena", {})
    graph_profile = modes.get("arena_graph", {})
    expected_tests = {
        "cpu_debug": {"passed": 286, "total": 286},
        "asan_ubsan": {"passed": 284, "total": 284},
        "pytorch_enabled_cpu": {"passed": 260, "total": 260},
        "hip_full_configuration": {
            "passed": 449, "total": 449, "conditional_skips": 3},
        "hip_label": {"passed": 152, "total": 152},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    profile_files = (
        "deferred-hip-api-stats.csv", "deferred-kernel-stats.csv",
        "arena-hip-api-stats.csv", "arena-kernel-stats.csv",
        "graph-hip-api-stats.csv", "graph-kernel-stats.csv")
    if summary.get("status") != "pass" or len(raw) != 36 or \
            summary.get("raw_processes") != 36 or len(comparisons) != 4 or \
            summary.get("correctness_gate") is not True or \
            summary.get("layout_contract") is not True or \
            summary.get("arena_keep_rows") != 3 or \
            summary.get("arena_graph_keep_rows") != 3 or \
            summary.get("decision") != \
                    "keep shape-selective FFN arena and Graph candidate" or \
            not (1.04 <= min(eager, default=0.0) <= 1.05) or \
            not (3.03 <= max(eager, default=0.0) <= 3.04) or \
            not (1.00 <= min(graph, default=0.0) <= 1.01) or \
            not (2.96 <= max(graph, default=0.0) <= 2.98) or \
            min(breaks, default=0) != 1 or max(breaks, default=0) != 568 or \
            any(row.get("graph_node_count") != 4 for row in comparisons) or \
            profile.get("status") != "pass" or \
            deferred_profile.get("kernel_calls") != arena_profile.get("kernel_calls") or \
            deferred_profile.get("kernel_calls") != graph_profile.get("kernel_calls") or \
            arena_profile.get("hip_malloc_calls", 0) >= \
                    deferred_profile.get("hip_malloc_calls", 0) or \
            graph_profile.get("host_kernel_launch_calls", 0) >= \
                    arena_profile.get("host_kernel_launch_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 63 or \
            verification.get("formal_processes") != 36 or \
            any(row.get("record_type") != "arena_ffn_measurement" or
                row.get("status") != "pass" or
                row.get("maximum_absolute_error") != 0 or
                row.get("rms_error") != 0 for row in raw) or \
            any(not (data / name).is_file() for name in profile_files):
        errors.append("arena FFN evidence changed")
    storage = (REPOSITORY / "include/microllm/core/storage.h").read_text(
        encoding="utf-8")
    ops = (REPOSITORY / "include/microllm/ops/ops.h").read_text(encoding="utf-8")
    benchmark = (REPOSITORY / "benchmarks/single_gpu/"
                 "benchmark_arena_ffn.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/arena_ffn_matrix.py").read_text(
        encoding="utf-8")
    if "static Storage from_external" not in storage or \
            "void swiglu_out_" not in ops or \
            "MatmulImplementation::HipBLASLt" not in benchmark or \
            "graph_break_even_replays" not in runner:
        errors.append("arena FFN source/test contract changed")
    return len(raw), min(eager, default=0.0), max(eager, default=0.0), \
        min(graph, default=0.0), max(graph, default=0.0), \
        min(breaks, default=0), max(breaks, default=0)


def validate_bf16_arena_ffn(
        errors: list[str]) -> tuple[int, float, float, float, float, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-arena-ffn"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    eager = [float(row.get("arena_speedup", 0.0)) for row in comparisons]
    graph = [float(row.get("arena_graph_speedup", 0.0)) for row in comparisons]
    nodes = {int(row.get("graph_node_count", 0)) for row in comparisons}
    modes = profile.get("modes", {})
    baseline_profile = modes.get("baseline", {})
    arena_profile = modes.get("arena", {})
    graph_profile = modes.get("arena_graph", {})
    expected_tests = {
        "cpu_debug": {"passed": 287, "total": 287},
        "asan_ubsan": {"passed": 285, "total": 285},
        "pytorch_enabled_cpu": {"passed": 261, "total": 261},
        "hip_full_configuration": {
            "passed": 450, "total": 450, "conditional_skips": 3},
        "hip_label": {"passed": 152, "total": 152},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    profile_files = (
        "baseline-hip-api-stats.csv", "baseline-kernel-stats.csv",
        "arena-hip-api-stats.csv", "arena-kernel-stats.csv",
        "arena_graph-hip-api-stats.csv", "arena_graph-kernel-stats.csv")
    deep_r32 = next((row for row in comparisons
                     if row.get("model") == "deepseek" and
                     row.get("rows") == 32), {})
    if summary.get("status") != "pass" or len(raw) != 54 or \
            summary.get("raw_processes") != 54 or len(comparisons) != 6 or \
            summary.get("correctness_gate") is not True or \
            summary.get("graph_layout_contract") is not True or \
            summary.get("arena_keep_rows") != 5 or \
            summary.get("arena_graph_keep_rows") != 5 or \
            summary.get("decision") != \
                    "measure complete-model BF16 FFN arena" or \
            not (1.03 <= min(eager, default=0.0) <= 1.04) or \
            not (5.54 <= max(eager, default=0.0) <= 5.56) or \
            not (0.96 <= min(graph, default=0.0) <= 0.98) or \
            not (5.04 <= max(graph, default=0.0) <= 5.06) or \
            nodes != {5, 6} or \
            not (0.96 <= float(deep_r32.get("arena_graph_speedup", 0.0)) <= 0.98) or \
            any(int(row.get("baseline_allocation_calls", 0)) < 100 or
                int(row.get("arena_allocation_calls", -1)) != 0 or
                int(row.get("graph_allocation_calls", -1)) != 0
                for row in comparisons) or \
            profile.get("status") != "pass" or \
            baseline_profile.get("kernel_calls") != arena_profile.get("kernel_calls") or \
            baseline_profile.get("kernel_calls") != graph_profile.get("kernel_calls") or \
            arena_profile.get("hip_malloc_calls", 0) >= \
                    baseline_profile.get("hip_malloc_calls", 0) or \
            graph_profile.get("hip_launch_kernel_calls", 0) + \
                    graph_profile.get("hip_ext_launch_calls", 0) >= \
                    arena_profile.get("hip_launch_kernel_calls", 0) + \
                    arena_profile.get("hip_ext_launch_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 64 or \
            verification.get("formal_processes") != 54 or \
            any(row.get("record_type") != "bf16_arena_ffn_measurement" or
                row.get("status") != "pass" or
                row.get("maximum_absolute_error") != 0 or
                row.get("rms_error") != 0 for row in raw) or \
            any(not (data / name).is_file() for name in profile_files):
        errors.append("BF16 arena FFN evidence changed")
    ops = (REPOSITORY / "include/microllm/ops/ops.h").read_text(encoding="utf-8")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "bf16_arena_ffn_matrix.py").read_text(encoding="utf-8")
    if "struct Bf16FfnWorkspace" not in ops or \
            "void bf16_ffn_out_" not in ops or \
            "hipblaslt_bf16_matmul_out" not in source or \
            "QwenDecodeShapeFallsBackToDeviceCastAndRemainsReusable" not in tests or \
            "measured_allocation_calls" not in runner:
        errors.append("BF16 arena FFN source/test contract changed")
    return len(raw), min(eager, default=0.0), max(eager, default=0.0), \
        min(graph, default=0.0), max(graph, default=0.0), \
        min(nodes, default=0), max(nodes, default=0)


def validate_bf16_ffn_arena_model(
        errors: list[str]) -> tuple[int, float, float, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-ffn-arena-model"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    speedups = [float(row.get("arena_speedup", 0.0)) for row in comparisons]
    modes = profile.get("modes", {})
    baseline_profile = modes.get("baseline", {})
    arena_profile = modes.get("arena", {})
    expected_tests = {
        "cpu_debug": {"passed": 288, "total": 288},
        "asan_ubsan": {"passed": 286, "total": 286},
        "pytorch_enabled_cpu": {"passed": 262, "total": 262},
        "hip_full_configuration": {
            "passed": 451, "total": 451, "conditional_skips": 3},
        "hip_label": {"passed": 152, "total": 152},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    profile_files = (
        "baseline-hip-api-stats.csv", "baseline-kernel-stats.csv",
        "arena-hip-api-stats.csv", "arena-kernel-stats.csv")
    if summary.get("status") != "pass" or len(raw) != 60 or \
            summary.get("raw_processes") != 60 or len(comparisons) != 10 or \
            summary.get("correctness_gate") is not True or \
            summary.get("keep_rows") != 3 or \
            summary.get("regression_rows") != 0 or \
            summary.get("decision") != \
                    "reject universal model Arena; inspect shape selection" or \
            not (0.99 <= min(speedups, default=0.0) <= 1.00) or \
            not (1.03 <= max(speedups, default=0.0) <= 1.04) or \
            any(row.get("maximum_absolute_logit_difference") != 0 or
                row.get("exact_expected_tokens") is not True or
                int(row.get("arena_engine_allocation_calls", 0)) >=
                    int(row.get("baseline_engine_allocation_calls", 0)) or
                int(row.get("arena_entries", 0)) <= 0 or
                int(row.get("arena_misses", 0)) <= 0
                for row in comparisons) or \
            profile.get("status") != "pass" or \
            baseline_profile.get("kernel_calls") != arena_profile.get("kernel_calls") or \
            arena_profile.get("hip_malloc_calls", 0) >= \
                    baseline_profile.get("hip_malloc_calls", 0) or \
            arena_profile.get("hip_free_calls", 0) >= \
                    baseline_profile.get("hip_free_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 65 or \
            verification.get("formal_processes") != 60 or \
            any(row.get("record_type") !=
                    "bf16_ffn_arena_model_measurement" or
                row.get("status") != "pass" for row in raw) or \
            any(not (data / name).is_file() for name in profile_files):
        errors.append("BF16 FFN Arena model evidence changed")
    header = (REPOSITORY / "include/microllm/model/model.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/model/model.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_ffn_arena_models.py").read_text(encoding="utf-8")
    if "struct Bf16FfnArenaStats" not in header or \
            "set_bf16_ffn_arena_enabled" not in header or \
            "class Bf16FfnArenaCache" not in source or \
            "--bf16-ffn-arena" not in cli or \
            "maximum_absolute_logit_difference" not in runner:
        errors.append("BF16 FFN Arena model source/test contract changed")
    return len(raw), min(speedups, default=0.0), max(speedups, default=0.0), \
        int(summary.get("keep_rows", 0)), int(summary.get("regression_rows", 0))


def validate_bf16_ffn_arena_selective(
        errors: list[str]) -> tuple[int, float, float, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-ffn-arena-selective"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    speedups = [float(row.get("arena_speedup", 0.0)) for row in comparisons]
    eligible = [row for row in comparisons
                if int(row.get("flattened_rows", 0)) >= 512]
    bypassed = [row for row in comparisons
                if int(row.get("flattened_rows", 0)) < 512]
    modes = profile.get("modes", {})
    baseline_profile = modes.get("baseline", {})
    selective_profile = modes.get("selective", {})
    expected_tests = {
        "cpu_debug": {"passed": 288, "total": 288},
        "asan_ubsan": {"passed": 286, "total": 286},
        "pytorch_enabled_cpu": {"passed": 262, "total": 262},
        "hip_full_configuration": {
            "passed": 451, "total": 451, "conditional_skips": 3},
        "hip_label": {"passed": 152, "total": 152},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    profile_files = (
        "baseline-hip-api-stats.csv", "baseline-kernel-stats.csv",
        "selective-hip-api-stats.csv", "selective-kernel-stats.csv")
    if summary.get("status") != "pass" or len(raw) != 60 or \
            summary.get("raw_processes") != 60 or len(comparisons) != 10 or \
            summary.get("correctness_gate") is not True or \
            summary.get("arena_minimum_rows") != 512 or \
            summary.get("eligible_rows") != 2 or \
            summary.get("bypassed_rows") != 8 or \
            summary.get("keep_rows") != 2 or \
            summary.get("regression_rows") != 0 or \
            summary.get("decision") != \
                    "keep rows>=512 selective model Arena" or \
            not (0.99 <= min(speedups, default=0.0) <= 1.00) or \
            not (1.02 <= max(speedups, default=0.0) <= 1.03) or \
            any(float(row.get("arena_speedup", 0.0)) < 1.01 or
                int(row.get("arena_entries", 0)) != 1 or
                int(row.get("arena_eligible_calls", 0)) <= 0 or
                int(row.get("arena_bypassed_calls", -1)) != 0 or
                int(row.get("arena_engine_allocation_calls", 0)) >=
                    int(row.get("baseline_engine_allocation_calls", 0))
                for row in eligible) or \
            any(int(row.get("arena_entries", -1)) != 0 or
                int(row.get("arena_capacity_bytes", -1)) != 0 or
                int(row.get("arena_eligible_calls", -1)) != 0 or
                int(row.get("arena_bypassed_calls", 0)) <= 0 or
                int(row.get("arena_engine_allocation_calls", -1)) !=
                    int(row.get("baseline_engine_allocation_calls", 0)) or
                int(row.get("arena_engine_peak_bytes", -1)) !=
                    int(row.get("baseline_engine_peak_bytes", 0))
                for row in bypassed) or \
            any(row.get("maximum_absolute_logit_difference") != 0 or
                row.get("exact_expected_tokens") is not True
                for row in comparisons) or \
            profile.get("status") != "pass" or \
            baseline_profile.get("kernel_calls") != \
                    selective_profile.get("kernel_calls") or \
            selective_profile.get("hip_malloc_calls", 0) >= \
                    baseline_profile.get("hip_malloc_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 65 or \
            verification.get("formal_processes") != 60 or \
            any(row.get("status") != "pass" for row in raw) or \
            any(not (data / name).is_file() for name in profile_files):
        errors.append("selective BF16 FFN Arena evidence changed")
    header = (REPOSITORY / "include/microllm/model/model.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/model/model.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_ffn_arena_models.py").read_text(encoding="utf-8")
    if "minimum_rows = 1" not in header or \
            "bypassed_calls_" not in source or \
            "--bf16-ffn-arena-minimum-rows" not in cli or \
            "--arena-minimum-rows" not in runner:
        errors.append("selective BF16 FFN Arena source/test contract changed")
    return len(raw), min(speedups, default=0.0), max(speedups, default=0.0), \
        len(eligible), len(bypassed)


def validate_bf16_qkv_arena_discard(
        errors: list[str]) -> tuple[int, float, float, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-qkv-arena-discard"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    speedups = [float(row.get("arena_speedup", 0.0)) for row in comparisons]
    eligible = [row for row in comparisons
                if int(row.get("flattened_rows", 0)) >= 512]
    bypassed = [row for row in comparisons
                if int(row.get("flattened_rows", 0)) < 512]
    modes = profile.get("modes", {})
    baseline_profile = modes.get("baseline", {})
    qkv_profile = modes.get("qkv", {})
    expected_tests = {
        "cpu_debug": {"passed": 288, "total": 288},
        "asan_ubsan": {"passed": 286, "total": 286},
        "pytorch_enabled_cpu": {"passed": 262, "total": 262},
        "hip_full_configuration": {
            "passed": 451, "total": 451, "conditional_skips": 3},
        "hip_label": {"passed": 152, "total": 152},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    profile_files = (
        "baseline-hip-api-stats.csv", "baseline-kernel-stats.csv",
        "qkv-hip-api-stats.csv", "qkv-kernel-stats.csv")
    if summary.get("status") != "pass" or len(raw) != 60 or \
            summary.get("raw_processes") != 60 or len(comparisons) != 10 or \
            summary.get("record_type") != "bf16_qkv_arena_model_summary" or \
            summary.get("comparison_mode") != "qkv" or \
            summary.get("correctness_gate") is not True or \
            summary.get("arena_minimum_rows") != 512 or \
            summary.get("eligible_rows") != 2 or \
            summary.get("bypassed_rows") != 8 or \
            summary.get("keep_rows") != 0 or \
            summary.get("regression_rows") != 1 or \
            summary.get("decision") != \
                    "reject rows>=512 selective QKV Arena" or \
            not (0.97 <= min(speedups, default=0.0) <= 0.98) or \
            not (1.00 <= max(speedups, default=0.0) <= 1.01) or \
            any(float(row.get("arena_speedup", 0.0)) >= 1.01 or
                int(row.get("arena_entries", 0)) != 1 or
                int(row.get("arena_eligible_calls", 0)) <= 0 or
                int(row.get("arena_engine_allocation_calls", 0)) >=
                    int(row.get("baseline_engine_allocation_calls", 0))
                for row in eligible) or \
            any(int(row.get("arena_entries", -1)) != 0 or
                int(row.get("arena_capacity_bytes", -1)) != 0 or
                int(row.get("arena_eligible_calls", -1)) != 0 or
                int(row.get("arena_bypassed_calls", 0)) <= 0 or
                int(row.get("arena_engine_allocation_calls", -1)) !=
                    int(row.get("baseline_engine_allocation_calls", 0)) or
                int(row.get("arena_engine_peak_bytes", -1)) !=
                    int(row.get("baseline_engine_peak_bytes", 0))
                for row in bypassed) or \
            any(row.get("maximum_absolute_logit_difference") != 0 or
                row.get("exact_expected_tokens") is not True
                for row in comparisons) or \
            profile.get("status") != "pass" or \
            baseline_profile.get("kernel_calls") != qkv_profile.get("kernel_calls") or \
            qkv_profile.get("hip_malloc_calls", 0) >= \
                    baseline_profile.get("hip_malloc_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 65 or \
            verification.get("formal_processes") != 60 or \
            any(row.get("record_type") !=
                    "bf16_qkv_arena_model_measurement" or
                row.get("status") != "pass" for row in raw) or \
            any(not (data / name).is_file() for name in profile_files):
        errors.append("BF16 QKV Arena discard evidence changed")
    ops = (REPOSITORY / "include/microllm/ops/ops.h").read_text(encoding="utf-8")
    model = (REPOSITORY / "include/microllm/model/model.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/model/model.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    if "struct Bf16QkvWorkspace" not in ops or \
            "bf16_qkv_projection_out_" not in ops or \
            "set_bf16_qkv_arena_enabled" not in model or \
            "class Bf16QkvArenaCache" not in source or \
            "--bf16-qkv-arena" not in cli:
        errors.append("BF16 QKV Arena source/test contract changed")
    return len(raw), min(speedups, default=0.0), max(speedups, default=0.0), \
        len(eligible), len(bypassed)


def validate_allocation_source_attribution(
        errors: list[str]) -> tuple[int, int, int, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-allocation-source-attribution"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    models = {row.get("model"): row for row in summary.get("models", [])}
    qwen = models.get("qwen2.5-0.5b", {})
    deep = models.get("deepseek-r1-distill-qwen-1.5b", {})
    qwen_top = qwen.get("sources", [{}])[0] if qwen.get("sources") else {}
    deep_top = deep.get("sources", [{}])[0] if deep.get("sources") else {}
    expected_tests = {
        "cpu_debug": {"passed": 290, "total": 290},
        "asan_ubsan": {"passed": 288, "total": 288},
        "pytorch_enabled_cpu": {"passed": 264, "total": 264},
        "hip_full_configuration": {
            "passed": 454, "total": 454, "conditional_skips": 3},
        "hip_label": {"passed": 153, "total": 153},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    if summary.get("status") != "pass" or len(raw) != 6 or \
            summary.get("raw_processes") != 6 or summary.get("context") != 512 or \
            summary.get("deterministic_distributions") is not True or \
            summary.get("common_top_source") != "attention.core" or \
            summary.get("decision") != "profile and optimize attention.core" or \
            len(models) != 2 or \
            qwen.get("allocation_calls") != 580 or \
            qwen.get("allocation_bytes") != 1079854592 or \
            qwen_top.get("source") != "attention.core" or \
            qwen_top.get("calls") != 144 or \
            qwen_top.get("total_bytes") != 572522496 or \
            deep.get("allocation_calls") != 676 or \
            deep.get("allocation_bytes") != 1817003520 or \
            deep_top.get("source") != "attention.core" or \
            deep_top.get("calls") != 168 or \
            deep_top.get("total_bytes") != 792723456 or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 66 or \
            verification.get("formal_processes") != 6 or \
            any(row.get("record_type") != "hf_allocation_source_measurement" or
                row.get("status") != "pass" or
                row.get("allocation_source_diagnostics") is not True
                for row in raw):
        errors.append("allocation source attribution evidence changed")
    header = (REPOSITORY / "include/microllm/runtime/diagnostics.h").read_text(
        encoding="utf-8")
    runtime = (REPOSITORY / "src/runtime/runtime.cpp").read_text(encoding="utf-8")
    model = (REPOSITORY / "src/model/model.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "hf_allocation_sources.py").read_text(encoding="utf-8")
    if "enum class AllocationSource" not in header or \
            "class ScopedAllocationSource" not in header or \
            "if (!allocation_source_diagnostics_enabled) return" not in runtime or \
            "AllocationSource::AttentionCore" not in model or \
            "--allocation-source-diagnostics" not in cli or \
            "deterministic_distributions" not in runner:
        errors.append("allocation source attribution source/test contract changed")
    return len(raw), int(qwen.get("allocation_calls", 0)), \
        int(qwen_top.get("total_bytes", 0)), \
        int(deep.get("allocation_calls", 0)), int(deep_top.get("total_bytes", 0))


def validate_attention_core_arena_discard(
        errors: list[str]) -> tuple[int, float, float, int, int]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-attention-core-arena-discard"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    speedups = [float(row.get("arena_speedup", 0.0)) for row in comparisons]
    eligible = [row for row in comparisons
                if int(row.get("flattened_rows", 0)) >= 512]
    bypassed = [row for row in comparisons
                if int(row.get("flattened_rows", 0)) < 512]
    modes = profile.get("modes", {})
    baseline_profile = modes.get("baseline", {})
    core_profile = modes.get("core", {})
    expected_tests = {
        "cpu_debug": {"passed": 290, "total": 290},
        "asan_ubsan": {"passed": 288, "total": 288},
        "pytorch_enabled_cpu": {"passed": 264, "total": 264},
        "hip_full_configuration": {
            "passed": 454, "total": 454, "conditional_skips": 3},
        "hip_label": {"passed": 153, "total": 153},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    profile_files = (
        "baseline-hip-api-stats.csv", "baseline-kernel-stats.csv",
        "core-hip-api-stats.csv", "core-kernel-stats.csv")
    if summary.get("status") != "pass" or len(raw) != 60 or \
            summary.get("raw_processes") != 60 or len(comparisons) != 10 or \
            summary.get("record_type") != "attention_core_arena_model_summary" or \
            summary.get("comparison_mode") != "core" or \
            summary.get("correctness_gate") is not True or \
            summary.get("arena_minimum_rows") != 512 or \
            summary.get("eligible_rows") != 2 or \
            summary.get("bypassed_rows") != 8 or \
            summary.get("keep_rows") != 0 or \
            summary.get("regression_rows") != 0 or \
            summary.get("decision") != \
                    "reject rows>=512 selective Attention core Arena" or \
            not (0.99 <= min(speedups, default=0.0) <= 1.00) or \
            not (1.00 <= max(speedups, default=0.0) <= 1.01) or \
            any(float(row.get("arena_speedup", 0.0)) >= 1.01 or
                int(row.get("arena_entries", 0)) != 1 or
                int(row.get("arena_eligible_calls", 0)) <= 0 or
                int(row.get("arena_engine_allocation_calls", 0)) >=
                    int(row.get("baseline_engine_allocation_calls", 0)) or
                int(row.get("arena_engine_peak_bytes", 0)) <=
                    int(row.get("baseline_engine_peak_bytes", 0))
                for row in eligible) or \
            any(int(row.get("arena_entries", -1)) != 0 or
                int(row.get("arena_capacity_bytes", -1)) != 0 or
                int(row.get("arena_eligible_calls", -1)) != 0 or
                int(row.get("arena_bypassed_calls", 0)) <= 0 or
                int(row.get("arena_engine_allocation_calls", -1)) !=
                    int(row.get("baseline_engine_allocation_calls", 0)) or
                int(row.get("arena_engine_peak_bytes", -1)) !=
                    int(row.get("baseline_engine_peak_bytes", 0))
                for row in bypassed) or \
            any(row.get("maximum_absolute_logit_difference") != 0 or
                row.get("exact_expected_tokens") is not True
                for row in comparisons) or \
            profile.get("status") != "pass" or \
            baseline_profile.get("kernel_calls") != core_profile.get("kernel_calls") or \
            core_profile.get("hip_malloc_calls", 0) >= \
                    baseline_profile.get("hip_malloc_calls", 0) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 66 or \
            verification.get("formal_processes") != 60 or \
            any(row.get("record_type") !=
                    "attention_core_arena_model_measurement" or
                row.get("status") != "pass" for row in raw) or \
            any(not (data / name).is_file() for name in profile_files):
        errors.append("Attention core Arena discard evidence changed")
    ops = (REPOSITORY / "include/microllm/ops/ops.h").read_text(encoding="utf-8")
    model = (REPOSITORY / "include/microllm/model/model.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/model/model.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    if "struct CausalGqaAttentionWorkspace" not in ops or \
            "causal_gqa_attention_out_" not in ops or \
            "set_attention_core_arena_enabled" not in model or \
            "class AttentionCoreArenaCache" not in source or \
            "--attention-core-arena" not in cli:
        errors.append("Attention core Arena source/test contract changed")
    return len(raw), min(speedups, default=0.0), max(speedups, default=0.0), \
        len(eligible), len(bypassed)


def validate_fp32_attention_solutions(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-fp32-attention-solutions"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    by_case = {(row.get("model"), row.get("operation")): row
               for row in comparisons}
    expected = {
        ("qwen", "qk"): (305434, 1.324141474179969),
        ("qwen", "pv"): (294519, 1.197782905917267),
        ("deepseek", "qk"): (305460, 1.2527205636829792),
        ("deepseek", "pv"): (292941, 1.1144110840295476),
    }
    speedups = [float(row.get("recommended_event_speedup", 0.0))
                for row in comparisons]
    expected_tests = {
        "cpu_debug": {"passed": 291, "total": 291},
        "asan_ubsan": {"passed": 289, "total": 289},
        "pytorch_enabled_cpu": {"passed": 265, "total": 265},
        "hip_full_configuration": {
            "passed": 456, "total": 456, "conditional_skips": 3},
        "hip_label": {"passed": 154, "total": 154},
        "rccl_multi_gpu": {"passed": 11, "total": 11},
        "rccl_full_label": {"passed": 13, "total": 13},
    }
    if summary.get("status") != "pass" or len(raw) != 12 or \
            summary.get("raw_processes") != 12 or len(comparisons) != 4 or \
            summary.get("keep_rows") != 4 or \
            summary.get("decision") != \
                    "register exact FP32 Attention candidates" or \
            set(by_case) != set(expected) or \
            any(row.get("common_passing_candidates") != 64 or
                row.get("recommended_index") != expected[case][0] or
                abs(float(row.get("recommended_event_speedup", 0.0)) -
                    expected[case][1]) > 1.0e-9 or
                float(row.get("recommended_maximum_absolute_error", 1.0)) >
                    4.5e-7 or
                float(row.get("recommended_maximum_rms_error", 1.0)) >
                    6.7e-8 or
                row.get("recommended_workspace_bytes") != 0
                for case, row in by_case.items()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 67 or \
            verification.get("formal_processes") != 12 or \
            any(row.get("record_type") != "fp32_attention_algorithm_tune" or
                row.get("status") != "pass" or
                int(row.get("passing_candidates", 0)) < 64
                for row in raw):
        errors.append("FP32 Attention solution evidence changed")
    benchmark = (REPOSITORY / "benchmarks/micro/"
                 "tune_fp32_attention_algorithms.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "fp32_attention_solution_matrix.py").read_text(encoding="utf-8")
    if "hipblasLtMatmulAlgoGetHeuristic" not in benchmark or \
            "complete_output_elements" not in benchmark or \
            "common_passing_candidates" not in runner:
        errors.append("FP32 Attention solution source/test contract changed")
    return len(raw), int(summary.get("keep_rows", 0)), \
        min(speedups, default=0.0), max(speedups, default=0.0)


def validate_fp32_attention_model_gate(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-fp32-attention-model-gate"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = summary.get("comparisons", [])
    speedups = [float(row.get("candidate_speedup", 0.0))
                for row in comparisons]
    by_case = {(row.get("model"), row.get("policy")): row
               for row in comparisons}
    expected_speedups = {
        ("qwen2.5-0.5b", "qk"): 1.0092907663474806,
        ("qwen2.5-0.5b", "pv"): 1.0037318746976513,
        ("qwen2.5-0.5b", "both"): 1.008228237745222,
        ("deepseek-r1-distill-qwen-1.5b", "qk"): 0.9991234974552436,
        ("deepseek-r1-distill-qwen-1.5b", "pv"): 1.0028952717701445,
        ("deepseek-r1-distill-qwen-1.5b", "both"): 1.0042520321010382,
    }
    expected_tests = {
        "cpu_debug": {"passed": 294, "total": 294},
        "asan_ubsan": {"passed": 292, "total": 292},
        "pytorch_enabled_cpu": {"passed": 268, "total": 268},
        "hip_full_configuration": {
            "passed": 460, "total": 460, "conditional_skips": 3},
        "hip_label": {"passed": 156, "total": 156},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {(model, policy): 0 for model, policy in expected_speedups}
    for row in raw:
        key = (row.get("model"), row.get("policy"))
        if key in counts:
            counts[key] += 1
    if summary.get("status") != "pass" or len(raw) != 24 or \
            summary.get("raw_processes") != 24 or len(comparisons) != 6 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not False or \
            summary.get("memory_gate") is not True or \
            summary.get("keep_default") is not False or \
            summary.get("keep_policies") != [] or \
            summary.get("policy_keep") != {"both": False, "pv": False,
                                            "qk": False} or \
            set(by_case) != set(expected_speedups) or \
            any(abs(float(row.get("candidate_speedup", 0.0)) -
                    expected_speedups[key]) > 1.0e-9 or
                row.get("correctness_passed") is not True or
                row.get("performance_passed") is not False or
                row.get("memory_passed") is not True or
                row.get("finite_complete_logits") is not True or
                row.get("maximum_absolute_logit_difference") != 0 or
                row.get("maximum_rms_logit_difference") != 0 or
                row.get("candidate_engine_peak_bytes") !=
                    row.get("baseline_engine_peak_bytes") or
                row.get("candidate_engine_allocation_calls") !=
                    row.get("baseline_engine_allocation_calls") or
                int(row.get("candidate_registered_entries", 0)) !=
                    (2 if key[1] == "both" else 1) or
                int(row.get("candidate_cached_algorithms", 0)) !=
                    (2 if key[1] == "both" else 1) or
                int(row.get("candidate_dispatches", 0)) <= 0
                for key, row in by_case.items()) or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 68 or \
            verification.get("formal_processes") != 24 or \
            any(row.get("record_type") !=
                    "fp32_attention_solution_model_measurement" or
                row.get("status") != "pass" for row in raw):
        errors.append("FP32 Attention complete-model gate evidence changed")
    header = (REPOSITORY / "include/microllm/ops/ops.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_fp32_attention_solutions.py").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(
        encoding="utf-8")
    for token, document in (
            ("struct Fp32MatmulSolutionKey", header),
            ("register_fp32_matmul_solution", source),
            ("registered_fp32_solution", source),
            ("--fp32-attention-qk-solution-index", cli),
            ("POLICIES = (\"baseline\", \"qk\", \"pv\", \"both\")", runner),
            ("ExactFp32AttentionSolutionDispatchesAndCaches", tests)):
        if token not in document:
            errors.append("FP32 Attention exact registry source/test contract changed")
            break
    return len(raw), sum(row.get("correctness_passed") is True
                         for row in comparisons), \
        min(speedups, default=0.0), max(speedups, default=0.0)


def validate_bf16_grouped_qkv(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-grouped-qkv"
    operator = json.loads((data / "operator-summary.json").read_text(
        encoding="utf-8"))
    model = json.loads((data / "model-summary.json").read_text(
        encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    operator_raw = [json.loads(line) for line in
                    (data / "operator-raw.jsonl").read_text(
                        encoding="utf-8").splitlines() if line.strip()]
    model_raw = [json.loads(line) for line in
                 (data / "model-raw.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    qwen_profile = json.loads((data / "qwen-profile-delta.json").read_text(
        encoding="utf-8"))
    deep_profile = json.loads((data / "deepseek-profile-delta.json").read_text(
        encoding="utf-8"))
    operator_rows = {row.get("model"): row
                     for row in operator.get("comparisons", [])}
    model_rows = {row.get("model"): row
                  for row in model.get("comparisons", [])}
    expected_operator = {
        "qwen": (1.880722645889, 0.908366288583, {64699, 64700}),
        "deepseek": (1.224754597791, 0.814856662631, {64701}),
    }
    expected_model = {
        "qwen2.5-0.5b": (1.0316563751060652, 24, 144, 24, 168),
        "deepseek-r1-distill-qwen-1.5b":
            (1.0014549014359095, 28, 168, 28, 196),
    }
    expected_tests = {
        "cpu_debug": {"passed": 297, "total": 297},
        "asan_ubsan": {"passed": 295, "total": 295},
        "pytorch_enabled_cpu": {"passed": 271, "total": 271},
        "hip_full_configuration": {
            "passed": 465, "total": 465, "conditional_skips": 3},
        "hip_label": {"passed": 158, "total": 158},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    operator_counts = {(model_name, mode): 0 for model_name in ("qwen", "deepseek")
                       for mode in ("model", "fp32")}
    for row in operator_raw:
        key = (row.get("model"), row.get("output_dtype"))
        if key in operator_counts:
            operator_counts[key] += 1
    model_counts = {(model_name, policy): 0 for model_name in expected_model
                    for policy in ("baseline", "grouped")}
    for row in model_raw:
        key = (row.get("model"), row.get("policy"))
        if key in model_counts:
            model_counts[key] += 1
    profile_ok = all(
        profile.get("status") == "pass" and
        profile.get("track") == "inference_prefill_kernel_phase_delta" and
        profile.get("derived_steps") == 5 and
        profile.get("categories", [{}])[0].get("category") ==
            "hipBLASLt GEMM" and
        0.53 <= float(profile.get("categories", [{}])[0].get(
            "kernel_share", 0.0)) <= 0.63
        for profile in (qwen_profile, deep_profile))
    if operator.get("status") != "pass" or len(operator_raw) != 12 or \
            operator.get("raw_processes") != 12 or \
            operator.get("direct_fp32_unsupported_rows") != 6 or \
            operator.get("operator_keep") is not True or \
            set(operator_rows) != set(expected_operator) or \
            any(abs(float(row.get("event_speedup_median", 0.0)) -
                    expected_operator[name][0]) > 1.0e-9 or
                abs(float(row.get("reinitialized_event_speedup_median", 0.0)) -
                    expected_operator[name][1]) > 1.0e-9 or
                set(row.get("solution_indices", [])) !=
                    expected_operator[name][2] or
                row.get("passing_candidates") != 16 or
                float(row.get("maximum_absolute_error", 1.0)) > 2.5e-4 or
                row.get("workspace_bytes") != 720
                for name, row in operator_rows.items()) or \
            any(count != 3 for count in operator_counts.values()) or \
            model.get("status") != "pass" or len(model_raw) != 12 or \
            model.get("raw_processes") != 12 or \
            model.get("correctness_gate") is not True or \
            model.get("performance_gate") is not False or \
            model.get("memory_gate") is not True or \
            model.get("keep_default") is not False or \
            set(model_rows) != set(expected_model) or \
            any(abs(float(row.get("grouped_speedup", 0.0)) -
                    expected_model[name][0]) > 1.0e-9 or
                row.get("grouped_plan_entries") != expected_model[name][1] or
                row.get("grouped_plan_hits") != expected_model[name][2] or
                row.get("grouped_plan_misses") != expected_model[name][3] or
                row.get("grouped_dispatches") != expected_model[name][4] or
                row.get("finite_complete_logits") is not True or
                row.get("top_tokens_equal") is not True or
                float(row.get("maximum_absolute_logit_difference", 1.0)) > 0.1 or
                float(row.get("maximum_rms_logit_difference", 1.0)) > 0.021 or
                float(row.get("peak_ratio", 2.0)) > 1.005 or
                row.get("grouped_engine_allocation_calls", 0) >=
                    row.get("baseline_engine_allocation_calls", 0)
                for name, row in model_rows.items()) or \
            any(count != 3 for count in model_counts.values()) or \
            not profile_ok or verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 70 or \
            verification.get("operator_processes") != 12 or \
            verification.get("model_processes") != 12:
        errors.append("BF16 grouped QKV evidence changed")
    header = (REPOSITORY / "include/microllm/ops/ops.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    benchmark = (REPOSITORY / "benchmarks/micro/"
                 "benchmark_bf16_grouped_qkv.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(
        encoding="utf-8")
    for token, document in (
            ("struct Bf16GroupedQkvKey", header),
            ("class Bf16GroupedQkvPlan", source),
            ("try_bf16_grouped_qkv", source),
            ("--bf16-grouped-qkv-algorithm-index", cli),
            ("reinitialized_event_speedup", benchmark),
            ("GroupedQkvCachesExactPointerStablePlan", tests)):
        if token not in document:
            errors.append("BF16 grouped QKV source/test contract changed")
            break
    model_speedups = [float(row.get("grouped_speedup", 0.0))
                      for row in model_rows.values()]
    return len(operator_raw), len(model_raw), \
        min(model_speedups, default=0.0), max(model_speedups, default=0.0)


def validate_bf16_grouped_qkv_expanded(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-grouped-qkv-expanded"
    baseline = REPOSITORY / "benchmarks/results/2026-08-24-bf16-grouped-qkv"
    operator = json.loads((data / "operator-summary.json").read_text(
        encoding="utf-8"))
    model = json.loads((data / "model-summary.json").read_text(
        encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    operator_raw = [json.loads(line) for line in
                    (data / "operator-raw.jsonl").read_text(
                        encoding="utf-8").splitlines() if line.strip()]
    model_raw = [json.loads(line) for line in
                 (data / "model-raw.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    qwen_candidate = json.loads((data / "qwen-candidate-profile-delta.json").read_text(
        encoding="utf-8"))
    deep_candidate = json.loads((data / "deepseek-candidate-profile-delta.json").read_text(
        encoding="utf-8"))
    qwen_baseline = json.loads((baseline / "qwen-profile-delta.json").read_text(
        encoding="utf-8"))
    deep_baseline = json.loads((baseline / "deepseek-profile-delta.json").read_text(
        encoding="utf-8"))
    operator_rows = {row.get("model"): row
                     for row in operator.get("comparisons", [])}
    model_rows = {row.get("model"): row
                  for row in model.get("comparisons", [])}
    expected_operator = {
        "qwen": (2.010077007329, 0.954473419582, {64713}),
        "deepseek": (1.692402362234, 0.958102986479, {64755}),
    }
    expected_model = {
        "qwen2.5-0.5b": (1.0458010919547664, 64713, 207.91916),
        "deepseek-r1-distill-qwen-1.5b":
            (1.0295074344653419, 64755, 203.684388),
    }
    expected_tests = {
        "cpu_debug": {"passed": 297, "total": 297},
        "asan_ubsan": {"passed": 295, "total": 295},
        "pytorch_enabled_cpu": {"passed": 271, "total": 271},
        "hip_full_configuration": {
            "passed": 465, "total": 465, "conditional_skips": 3},
        "hip_label": {"passed": 158, "total": 158},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    profile_pairs = (
        (qwen_baseline, qwen_candidate, 217, 169, 1.018),
        (deep_baseline, deep_candidate, 253, 197, 1.020),
    )
    profile_ok = True
    for before, after, before_calls, after_calls, minimum_speedup in profile_pairs:
        profile_speedup = (
            float(before.get("total_kernel_ns_per_step", 0.0)) /
            float(after.get("total_kernel_ns_per_step", math.inf)))
        if before.get("status") != "pass" or after.get("status") != "pass" or \
                after.get("track") != "inference_prefill_kernel_phase_delta" or \
                profile_speedup < minimum_speedup or \
                int(before.get("categories", [{}])[0].get("calls_per_step", -1)) != \
                    before_calls or \
                int(after.get("categories", [{}])[0].get("calls_per_step", -1)) != \
                    after_calls:
            profile_ok = False
    if operator.get("status") != "pass" or len(operator_raw) != 12 or \
            operator.get("raw_processes") != 12 or \
            operator.get("operator_keep") is not True or \
            operator.get("direct_fp32_unsupported_rows") != 6 or \
            set(operator_rows) != set(expected_operator) or \
            any(abs(float(row.get("event_speedup_median", 0.0)) -
                    expected_operator[name][0]) > 1.0e-9 or
                abs(float(row.get("reinitialized_event_speedup_median", 0.0)) -
                    expected_operator[name][1]) > 1.0e-9 or
                set(row.get("solution_indices", [])) !=
                    expected_operator[name][2] or
                row.get("passing_candidates") != 64
                for name, row in operator_rows.items()) or \
            model.get("status") != "pass" or len(model_raw) != 12 or \
            model.get("raw_processes") != 12 or \
            model.get("correctness_gate") is not True or \
            model.get("performance_gate") is not True or \
            model.get("memory_gate") is not True or \
            model.get("setup_gate") is not False or \
            model.get("keep_steady_policy") is not True or \
            model.get("keep_default") is not False or \
            set(model_rows) != set(expected_model) or \
            any(abs(float(row.get("grouped_speedup", 0.0)) -
                    expected_model[name][0]) > 1.0e-9 or
                row.get("solution_index") != expected_model[name][1] or
                abs(float(row.get("grouped_kernel_setup_ms", 0.0)) -
                    expected_model[name][2]) > 1.0e-6 or
                row.get("grouped_algorithm_entries") != 1 or
                row.get("grouped_kernel_entries") != 1 or
                float(row.get("grouped_argument_setup_ms", math.inf)) > 1.0 or
                row.get("finite_complete_logits") is not True or
                row.get("top_tokens_equal") is not True or
                float(row.get("peak_ratio", 2.0)) > 1.005
                for name, row in model_rows.items()) or \
            not profile_ok or verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 70 or \
            verification.get("steady_policy_kept") is not True or \
            verification.get("default_policy_kept") is not False:
        errors.append("expanded BF16 grouped QKV evidence changed")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(encoding="utf-8")
    header = (REPOSITORY / "include/microllm/ops/ops.h").read_text(
        encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_grouped_qkv_models.py").read_text(encoding="utf-8")
    for token, document in (
            ("hipblaslt_ext::UserArguments", source),
            ("class Bf16GroupedQkvKernel", source),
            ("kernel_setup_ms", header),
            ("maximum-kernel-setup-ms", runner),
            ("default=64713", runner),
            ("default=64755", runner)):
        if token not in document:
            errors.append("expanded BF16 grouped QKV source contract changed")
            break
    speedups = [float(row.get("grouped_speedup", 0.0))
                for row in model_rows.values()]
    return len(operator_raw), len(model_raw), \
        min(speedups, default=0.0), max(speedups, default=0.0)


def validate_bf16_grouped_qkv_prewarm(
        errors: list[str]) -> tuple[int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-grouped-qkv-prewarm"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b":
            (4972.722535, 5744.076396, 915.347308, 4851.945239,
             208.224463, 0.638167),
        "deepseek-r1-distill-qwen-1.5b":
            (4992.869244, 5741.351027, 886.464203, 4794.677073,
             201.361002, 1.151982),
    }
    expected_tests = {
        "cpu_debug": {"passed": 298, "total": 298},
        "asan_ubsan": {"passed": 296, "total": 296},
        "pytorch_enabled_cpu": {"passed": 272, "total": 272},
        "hip_full_configuration": {
            "passed": 467, "total": 467, "conditional_skips": 3},
        "hip_label": {"passed": 159, "total": 159},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {(model, policy): 0 for model in expected
              for policy in ("baseline", "lazy", "prewarm")}
    for row in raw:
        key = (row.get("model"), row.get("policy"))
        if key in counts:
            counts[key] += 1
    if summary.get("status") != "pass" or len(raw) != 18 or \
            summary.get("raw_processes") != 18 or \
            summary.get("correctness_gate") is not True or \
            summary.get("setup_moved_before_request") is not True or \
            set(comparisons) != set(expected) or \
            any(any(abs(float(row.get(field, 0.0)) - value) > 1.0e-6
                    for field, value in zip(
                        ("baseline_first_ms", "lazy_first_ms", "prewarm_ms",
                         "prewarmed_first_ms", "kernel_setup_ms",
                         "argument_setup_ms"), expected[name], strict=True)) or
                row.get("finite_complete_logits") is not True or
                row.get("prewarmed_first_ms", math.inf) >=
                    row.get("lazy_first_ms", 0.0) or
                abs(float(row.get("prewarm_plus_first_ms", 0.0)) -
                    (float(row.get("prewarm_ms", 0.0)) +
                     float(row.get("prewarmed_first_ms", 0.0)))) > 1.0e-6
                for name, row in comparisons.items()) or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 71 or \
            verification.get("formal_processes") != 18 or \
            verification.get("setup_moved_before_request") is not True:
        errors.append("BF16 grouped QKV prewarm evidence changed")
    header = (REPOSITORY / "include/microllm/model/model.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/model/model.cpp").read_text(encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_grouped_qkv_prewarm.py").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(
        encoding="utf-8")
    for token, document in (
            ("Bf16GroupedQkvPrewarmReport", header),
            ("prewarm_bf16_grouped_qkv", source),
            ("--bf16-grouped-qkv-prewarm", cli),
            ("prewarm_plus_first_ms", runner),
            ("ModelPrewarmBuildsPlansBeforeFirstRequest", tests)):
        if token not in document:
            errors.append("BF16 grouped QKV prewarm source/test contract changed")
            break
    savings = [float(row["lazy_first_ms"]) -
               float(row["prewarmed_first_ms"])
               for row in comparisons.values()]
    return len(raw), min(savings, default=0.0), max(savings, default=0.0)


def validate_hipblaslt_preload(
        errors: list[str]) -> tuple[int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-hipblaslt-preload"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b":
            (3582.284506, 5029.966559, 17189.739954,
             6175.936616957188, 19394.636845216155,
             1309500928, 1309500928, 0.10526800155639648,
             0.01596291886159742),
        "deepseek-r1-distill-qwen-1.5b":
            (3563.8234, 4967.874941, 17123.138375,
             6694.004409946501, 19668.497520964593,
             4561625088, 4561625088, 0.04504561424255371,
             0.008749104007888512),
    }
    expected_tests = {
        "cpu_debug": {"passed": 299, "total": 299},
        "asan_ubsan": {"passed": 297, "total": 297},
        "pytorch_enabled_cpu": {"passed": 273, "total": 273},
        "hip_full_configuration": {
            "passed": 468, "total": 468, "conditional_skips": 3},
        "hip_label": {"passed": 159, "total": 159},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    policies = ("fp32", "bf16_lazy", "bf16_preload_all")
    counts = {(model, policy): 0 for model in expected for policy in policies}
    for row in raw:
        key = (row.get("model"), row.get("policy"))
        if key in counts:
            counts[key] += 1
        if row.get("hipblaslt_preload_kernels") != (
                1 if row.get("policy") == "bf16_preload_all" else 0):
            errors.append("hipBLASLt preload environment label changed")
            break
    fields = (
        "fp32_first_forward_ms", "bf16_lazy_first_forward_ms",
        "bf16_preload_first_forward_ms", "bf16_lazy_process_wall_ms",
        "bf16_preload_process_wall_ms", "bf16_lazy_peak_bytes",
        "bf16_preload_peak_bytes", "maximum_absolute_logit_difference",
        "maximum_rms_logit_difference")
    if summary.get("status") != "pass" or len(raw) != 18 or \
            summary.get("raw_processes") != 18 or \
            summary.get("correctness_gate") is not True or \
            summary.get("preload_counterexample_gate") is not True or \
            set(comparisons) != set(expected) or \
            any(any(abs(float(row.get(field, 0.0)) - value) > 1.0e-6
                    for field, value in zip(
                        fields, expected[name], strict=True)) or
                row.get("finite_complete_logits") is not True or
                float(row.get("preload_forward_slowdown", 0.0)) < 1.25 or
                float(row.get("preload_process_slowdown", 0.0)) < 1.25
                for name, row in comparisons.items()) or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 72 or \
            verification.get("formal_processes") != 18 or \
            verification.get("formal_comparisons") != 2 or \
            verification.get("preload_counterexample_gate") is not True:
        errors.append("hipBLASLt preload evidence changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_hipblaslt_preload.py").read_text(encoding="utf-8")
    contract = (REPOSITORY / "python/tests/"
                "test_hipblaslt_preload.py").read_text(encoding="utf-8")
    for token, document in (
            ("HIPBLASLT_PRELOAD_KERNELS", runner),
            ("process_wall_ms", runner),
            ("preload_counterexample_gate", runner),
            ("HIPBLASLT_PRELOAD_KERNELS", contract)):
        if token not in document:
            errors.append("hipBLASLt preload runner/test contract changed")
            break
    slowdowns = [float(row["preload_forward_slowdown"])
                 for row in comparisons.values()]
    return len(raw), min(slowdowns, default=0.0), max(slowdowns, default=0.0)


def validate_bf16_exact_startup(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-exact-startup"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    tuning = [json.loads(line) for line in (data / "tuning-raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    model_raw = [json.loads(line) for line in (data / "model-raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b": {
            "selected_index": 76074,
            "operator_speedup": 1.059120268816216,
            "default_cold_forward_ms": 4888.837971,
            "exact_cold_forward_ms": 4935.873106,
            "cold_forward_speedup": 0.990470756846884,
            "default_cold_process_wall_ms": 5966.294773854315,
            "exact_cold_process_wall_ms": 6099.905170034617,
            "cold_process_speedup": 0.9780963158514899,
            "default_steady_tokens_per_second": 93617.447659253,
            "exact_steady_tokens_per_second": 91088.405033133,
            "steady_speedup": 0.972985349533079,
        },
        "deepseek-r1-distill-qwen-1.5b": {
            "selected_index": 76091,
            "operator_speedup": 1.0322355943946127,
            "default_cold_forward_ms": 4907.306589,
            "exact_cold_forward_ms": 4924.916162,
            "cold_forward_speedup": 0.996424391315354,
            "default_cold_process_wall_ms": 6535.664352122694,
            "exact_cold_process_wall_ms": 6660.592336207628,
            "cold_process_speedup": 0.9812437126040858,
            "default_steady_tokens_per_second": 49954.926411417,
            "exact_steady_tokens_per_second": 50315.568039669,
            "steady_speedup": 1.007219340596798,
        },
    }
    expected_tests = {
        "cpu_debug": {"passed": 300, "total": 300},
        "asan_ubsan": {"passed": 298, "total": 298},
        "pytorch_enabled_cpu": {"passed": 274, "total": 274},
        "hip_full_configuration": {
            "passed": 469, "total": 469, "conditional_skips": 3},
        "hip_label": {"passed": 159, "total": 159},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    tuning_counts = {name: 0 for name in expected}
    model_counts = {(name, phase, policy): 0 for name in expected
                    for phase in ("cold", "steady")
                    for policy in ("default", "exact")}
    for row in tuning:
        name = row.get("model")
        if name in tuning_counts:
            tuning_counts[name] += 1
    for row in model_raw:
        key = (row.get("model"), row.get("phase"), row.get("policy"))
        if key in model_counts:
            model_counts[key] += 1
    evidence_changed = False
    for name, values in expected.items():
        row = comparisons.get(name, {})
        if row.get("selected_index") != values["selected_index"] or \
                row.get("common_passing_candidates") != 64 or \
                row.get("finite_complete_logits") is not True or \
                float(row.get("maximum_absolute_logit_difference", 1.0)) != 0 or \
                float(row.get("maximum_rms_logit_difference", 1.0)) != 0 or \
                float(row.get("peak_ratio", 0.0)) != 1.0:
            evidence_changed = True
            break
        for field, value in values.items():
            if field == "selected_index":
                continue
            if abs(float(row.get(field, 0.0)) - value) > 1.0e-6:
                evidence_changed = True
                break
    if summary.get("status") != "pass" or len(tuning) != 6 or \
            len(model_raw) != 24 or summary.get("tuner_processes") != 6 or \
            summary.get("model_processes") != 24 or \
            summary.get("correctness_gate") is not True or \
            summary.get("memory_gate") is not True or \
            summary.get("cold_performance_gate") is not False or \
            summary.get("steady_performance_gate") is not False or \
            summary.get("performance_gate") is not False or \
            set(comparisons) != set(expected) or evidence_changed or \
            any(count != 3 for count in tuning_counts.values()) or \
            any(count != 3 for count in model_counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 73 or \
            verification.get("tuner_processes") != 6 or \
            verification.get("model_processes") != 24 or \
            verification.get("formal_comparisons") != 2 or \
            verification.get("performance_gate") is not False:
        errors.append("exact BF16 startup evidence changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_exact_startup.py").read_text(encoding="utf-8")
    contract = (REPOSITORY / "python/tests/"
                "test_bf16_exact_startup.py").read_text(encoding="utf-8")
    for token, document in (
            ("common_passing_candidates", runner),
            ("cold_forward_speedup", runner),
            ("steady_speedup", runner),
            ("HIPBLASLT_PRELOAD_KERNELS", runner),
            ("performance_gate", contract)):
        if token not in document:
            errors.append("exact BF16 startup runner/test contract changed")
            break
    cold_ratios = [float(row["cold_forward_speedup"])
                   for row in comparisons.values()]
    operator_ratios = [float(row["operator_speedup"])
                       for row in comparisons.values()]
    return len(tuning), len(model_raw), min(cold_ratios, default=0.0), \
        max(operator_ratios, default=0.0)


def validate_bf16_grouped_gate_up(
        errors: list[str]) -> tuple[int, float, float]:
    data = REPOSITORY / "benchmarks/results/2026-08-24-bf16-grouped-gate-up"
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen": {
            "solution_indices": [65168, 65198],
            "event_speedup_median": 1.202964033465,
            "wall_speedup_median": 1.162324455206,
            "reinitialized_event_speedup_median": 0.823183132833,
            "reinitialized_wall_speedup_median": 0.869636993632,
            "user_arguments_setup_ms_median": 0.053674,
            "user_arguments_event_speedup_median": 1.188396274575,
            "user_arguments_wall_speedup_median": 1.139376797471,
        },
        "deepseek": {
            "solution_indices": [65200],
            "event_speedup_median": 1.138599957149,
            "wall_speedup_median": 1.113179702577,
            "reinitialized_event_speedup_median": 0.939959619441,
            "reinitialized_wall_speedup_median": 0.970368008444,
            "user_arguments_setup_ms_median": 0.052356,
            "user_arguments_event_speedup_median": 1.155151582167,
            "user_arguments_wall_speedup_median": 1.152467741318,
        },
    }
    expected_tests = {
        "cpu_debug": {"passed": 301, "total": 301},
        "asan_ubsan": {"passed": 299, "total": 299},
        "pytorch_enabled_cpu": {"passed": 275, "total": 275},
        "hip_full_configuration": {
            "passed": 471, "total": 471, "conditional_skips": 3},
        "hip_label": {"passed": 160, "total": 160},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {name: 0 for name in expected}
    for row in raw:
        name = row.get("model")
        if name in counts:
            counts[name] += 1
    evidence_changed = False
    for name, values in expected.items():
        row = comparisons.get(name, {})
        if row.get("solution_indices") != values["solution_indices"] or \
                row.get("groups") != 2 or \
                row.get("algorithm_count") != 10227 or \
                row.get("passing_candidates") != 64 or \
                float(row.get("maximum_absolute_error", 1.0)) != 0 or \
                float(row.get("maximum_rms_error", 1.0)) != 0:
            evidence_changed = True
            break
        for field, value in values.items():
            if field == "solution_indices":
                continue
            if abs(float(row.get(field, 0.0)) - value) > 1.0e-6:
                evidence_changed = True
                break
    if summary.get("status") != "pass" or len(raw) != 6 or \
            summary.get("raw_processes") != 6 or \
            summary.get("capability_gate") is not True or \
            summary.get("reinitialization_counterexample_gate") is not True or \
            set(comparisons) != set(expected) or evidence_changed or \
            any(count != 3 for count in counts.values()) or \
            any(row.get("record_type") != "bf16_grouped_gate_up_probe" or
                row.get("projection") != "gate-up" or
                row.get("groups") != 2 or
                row.get("grouped_supported") is not True
                for row in raw) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 74 or \
            verification.get("formal_processes") != 6 or \
            verification.get("formal_comparisons") != 2 or \
            verification.get("capability_gate") is not True or \
            verification.get(
                "reinitialization_counterexample_gate") is not True:
        errors.append("BF16 grouped gate/up evidence changed")
    source = (REPOSITORY / "benchmarks/micro/"
              "benchmark_bf16_grouped_qkv.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "bf16_grouped_gate_up_matrix.py").read_text(encoding="utf-8")
    contract = (REPOSITORY / "python/tests/"
                "test_bf16_grouped_gate_up_matrix.py").read_text(
                    encoding="utf-8")
    for token, document in (
            ("--projection", source),
            ("bf16_grouped_gate_up_probe", source),
            ("user_arguments_event_speedup_median", runner),
            ("reinitialization_counterexample_gate", contract)):
        if token not in document:
            errors.append("BF16 grouped gate/up source/test contract changed")
            break
    user_ratios = [float(row["user_arguments_event_speedup_median"])
                   for row in comparisons.values()]
    return len(raw), min(user_ratios, default=0.0), \
        max(user_ratios, default=0.0)


def validate_bf16_grouped_gate_up_model(
        errors: list[str]) -> tuple[int, float, float, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-bf16-grouped-gate-up-model")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b": {
            "solution_index": 65168,
            "baseline_tokens_per_second": 93470.65145685,
            "grouped_tokens_per_second": 95117.74015706,
            "grouped_speedup": 1.0176214530929033,
            "baseline_peak_bytes": 1310108672,
            "grouped_peak_bytes": 1310118560,
            "peak_ratio": 1.0000075474654975,
            "kernel_setup_ms": 56.9619,
            "argument_setup_ms": 0.575717,
            "plan_entries": 24,
            "plan_hits": 144,
            "dispatches": 168,
            "maximum_absolute_logit_difference": 0.07027947902679443,
            "maximum_rms_logit_difference": 0.01537959767072217,
        },
        "deepseek-r1-distill-qwen-1.5b": {
            "solution_index": 65200,
            "baseline_tokens_per_second": 50156.977625168,
            "grouped_tokens_per_second": 50745.560313547,
            "grouped_speedup": 1.011734811710099,
            "baseline_peak_bytes": 4562232832,
            "grouped_peak_bytes": 4562244288,
            "peak_ratio": 1.0000025110511501,
            "kernel_setup_ms": 56.827738,
            "argument_setup_ms": 0.698019,
            "plan_entries": 28,
            "plan_hits": 168,
            "dispatches": 196,
            "maximum_absolute_logit_difference": 0.06139183044433594,
            "maximum_rms_logit_difference": 0.010285622249765376,
        },
    }
    expected_profiles = {
        ("qwen2.5-0.5b", "baseline"):
            (5733181.6, 217.0, 3139070.2),
        ("qwen2.5-0.5b", "grouped"):
            (5629606.4, 193.0, 3033743.4),
        ("deepseek-r1-distill-qwen-1.5b", "baseline"):
            (10504172.2, 253.0, 6602615.3999999985),
        ("deepseek-r1-distill-qwen-1.5b", "grouped"):
            (10525514.4, 225.0, 6474042.000000001),
    }
    expected_tests = {
        "cpu_debug": {"passed": 303, "total": 303},
        "asan_ubsan": {"passed": 301, "total": 301},
        "pytorch_enabled_cpu": {"passed": 277, "total": 277},
        "hip_full_configuration": {
            "passed": 474, "total": 474, "conditional_skips": 3},
        "hip_label": {"passed": 161, "total": 161},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {(name, policy): 0 for name in expected
              for policy in ("baseline", "grouped")}
    for row in raw:
        key = (row.get("model"), row.get("policy"))
        if key in counts:
            counts[key] += 1
    evidence_changed = False
    for name, values in expected.items():
        row = comparisons.get(name, {})
        if row.get("finite_complete_logits") is not True or \
                row.get("top_tokens_equal") is not True:
            evidence_changed = True
            break
        for field, value in values.items():
            if abs(float(row.get(field, 0.0)) - value) > 1.0e-6:
                evidence_changed = True
                break
    profile_calls = {}
    for (model, policy), expected_profile in expected_profiles.items():
        directory = data / f"profile-{model}-{policy}"
        for filename in (
                "one-step-kernel-stats.csv",
                "three-step-kernel-stats.csv"):
            if not (directory / filename).is_file():
                errors.append(
                    f"grouped gate/up profile file missing: {directory}/{filename}")
        profile = json.loads((directory / "profile-delta.json").read_text(
            encoding="utf-8"))
        categories = {row.get("category"): row
                      for row in profile.get("categories", [])}
        gemm = categories.get("hipBLASLt GEMM", {})
        actual = (
            float(profile.get("total_kernel_ns_per_step", 0.0)),
            float(gemm.get("calls_per_step", 0.0)),
            float(gemm.get("duration_ns_per_step", 0.0)))
        if profile.get("status") != "pass" or \
                profile.get("track") != \
                    "inference_prefill_kernel_phase_delta" or \
                any(abs(value - expected_value) > 1.0e-6
                    for value, expected_value in zip(
                        actual, expected_profile, strict=True)):
            errors.append("grouped gate/up profile evidence changed")
        profile_calls[(model, policy)] = actual[1]
    if summary.get("status") != "pass" or len(raw) != 12 or \
            summary.get("raw_processes") != 12 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not True or \
            summary.get("memory_gate") is not True or \
            summary.get("setup_gate") is not True or \
            set(comparisons) != set(expected) or evidence_changed or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 75 or \
            verification.get("formal_processes") != 12 or \
            verification.get("formal_comparisons") != 2 or \
            verification.get("profile_deltas") != 4 or \
            any(verification.get(gate) is not True for gate in (
                "correctness_gate", "performance_gate",
                "memory_gate", "setup_gate")):
        errors.append("BF16 grouped gate/up model evidence changed")
    header = (REPOSITORY / "include/microllm/ops/ops.h").read_text(
        encoding="utf-8")
    source = (REPOSITORY / "src/ops/optimized.cpp").read_text(
        encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(
        encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_grouped_gate_up_models.py").read_text(
                  encoding="utf-8")
    for token, document in (
            ("Bf16GroupedGateUpKey", header),
            ("Bf16GroupedGateUpKernel", source),
            ("try_bf16_grouped_gate_up", source),
            ("--bf16-grouped-gate-up-algorithm-index", cli),
            ("GroupedGateUpCachesExactPointerStablePlan", tests),
            ("top_tokens_equal", runner)):
        if token not in document:
            errors.append("BF16 grouped gate/up model source/test contract changed")
            break
    speedups = [float(row["grouped_speedup"])
                for row in comparisons.values()]
    calls_saved = sum(
        int(profile_calls[(model, "baseline")] -
            profile_calls[(model, "grouped")])
        for model in expected)
    return len(raw), min(speedups, default=0.0), \
        max(speedups, default=0.0), calls_saved


def validate_bf16_grouped_composition(
        errors: list[str]) -> tuple[int, float, float, float]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-bf16-grouped-composition")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b": {
            "tokens_per_second": {
                "baseline": 93564.905499811,
                "qkv": 97741.029187075,
                "gate_up": 95217.886233595,
                "both": 99689.599298434,
            },
            "speedup_vs_baseline": {
                "baseline": 1.0,
                "qkv": 1.0446334409782783,
                "gate_up": 1.0176666745394976,
                "both": 1.0654593061993247,
            },
            "both_vs_qkv_speedup": 1.0199360506796942,
            "both_vs_gate_up_speedup": 1.0469629524632451,
            "both_peak_ratio": 1.0034204551849573,
            "qkv_kernel_setup_ms": 214.23376,
            "gate_up_kernel_setup_ms": 0.248997,
            "combined_kernel_setup_ms": 214.482757,
            "maximum_absolute_logit_difference": 0.12031126022338867,
            "maximum_rms_logit_difference": 0.029053766956449397,
        },
        "deepseek-r1-distill-qwen-1.5b": {
            "tokens_per_second": {
                "baseline": 50327.806036164,
                "qkv": 51819.339564541,
                "gate_up": 50917.050893164,
                "both": 52710.931440045,
            },
            "speedup_vs_baseline": {
                "baseline": 1.0,
                "qkv": 1.0296363709418452,
                "gate_up": 1.0117081371792083,
                "both": 1.0473520622410712,
            },
            "both_vs_qkv_speedup": 1.01720577458139,
            "both_vs_gate_up_speedup": 1.0352314306389228,
            "both_peak_ratio": 1.001730065143681,
            "qkv_kernel_setup_ms": 205.349409,
            "gate_up_kernel_setup_ms": 0.239435,
            "combined_kernel_setup_ms": 205.588844,
            "maximum_absolute_logit_difference": 0.07199788093566895,
            "maximum_rms_logit_difference": 0.012551317609247993,
        },
    }
    expected_tests = {
        "cpu_debug": {"passed": 304, "total": 304},
        "asan_ubsan": {"passed": 302, "total": 302},
        "pytorch_enabled_cpu": {"passed": 278, "total": 278},
        "hip_full_configuration": {
            "passed": 475, "total": 475, "conditional_skips": 3},
        "hip_label": {"passed": 161, "total": 161},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {(name, policy): 0 for name in expected
              for policy in ("baseline", "qkv", "gate_up", "both")}
    for row in raw:
        key = (row.get("model"), row.get("policy"))
        if key in counts:
            counts[key] += 1
    evidence_changed = False
    for name, values in expected.items():
        row = comparisons.get(name, {})
        if row.get("finite_complete_logits") is not True or \
                row.get("top_tokens_equal") is not True:
            evidence_changed = True
            break
        for mapping in ("tokens_per_second", "speedup_vs_baseline"):
            if any(abs(float(row.get(mapping, {}).get(policy, 0.0)) - value)
                   > 1.0e-6 for policy, value in
                   values[mapping].items()):
                evidence_changed = True
                break
        for field, value in values.items():
            if isinstance(value, dict):
                continue
            if abs(float(row.get(field, 0.0)) - value) > 1.0e-6:
                evidence_changed = True
                break
    if summary.get("status") != "pass" or len(raw) != 24 or \
            summary.get("raw_processes") != 24 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not True or \
            summary.get("memory_gate") is not True or \
            summary.get("setup_gate") is not True or \
            set(comparisons) != set(expected) or evidence_changed or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 76 or \
            verification.get("formal_processes") != 24 or \
            verification.get("formal_comparisons") != 2 or \
            any(verification.get(gate) is not True for gate in (
                "correctness_gate", "performance_gate",
                "memory_gate", "setup_gate")):
        errors.append("BF16 grouped composition evidence changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_grouped_composition.py").read_text(
                  encoding="utf-8")
    contract = (REPOSITORY / "python/tests/"
                "test_bf16_grouped_composition.py").read_text(
                    encoding="utf-8")
    for token, document in (
            ("POLICIES = (\"baseline\", \"qkv\", \"gate_up\", \"both\")",
             runner),
            ("both_vs_qkv_speedup", runner),
            ("bf16_grouped_qkv_dispatches", runner),
            ("bf16_grouped_gate_up_dispatches", runner),
            ("both_vs_qkv_speedup", contract)):
        if token not in document:
            errors.append("BF16 grouped composition runner/test contract changed")
            break
    both_ratios = [
        float(row["speedup_vs_baseline"]["both"])
        for row in comparisons.values()]
    incremental = [
        float(row["both_vs_qkv_speedup"])
        for row in comparisons.values()]
    return len(raw), min(both_ratios, default=0.0), \
        max(both_ratios, default=0.0), min(incremental, default=0.0)


def validate_bf16_grouped_shape_matrix(
        errors: list[str]) -> tuple[int, float, float, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-bf16-grouped-shape-matrix")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {
        (int(row.get("rows", 0)), row.get("model"), row.get("projection")): row
        for row in summary.get("comparisons", [])}
    expected = {
        (256, "qwen", "qkv"):
            ([64713, 64752], 1.694743346128, 0.929548571318,
             0.000224091113, 6.2246841e-05),
        (256, "qwen", "gate-up"):
            ([65197], 1.338726334483, 0.834679252714, 0.0, 0.0),
        (256, "deepseek", "qkv"):
            ([64699, 64713], 1.604234893472, 0.964444863125,
             0.000243678689, 0.000108414782),
        (256, "deepseek", "gate-up"):
            ([65168], 1.236322275422, 0.924286296054,
             7.629395e-06, 4.15344e-07),
        (1024, "qwen", "qkv"):
            ([64713, 64754, 64755], 1.389292250439, 0.782835086489,
             0.000224091113, 6.234508e-05),
        (1024, "qwen", "gate-up"):
            ([65168, 65200], 1.123997295942, 0.846666662161,
             0.0, 0.0),
        (1024, "deepseek", "qkv"):
            ([64754, 64755], 1.397329254401, 0.921041695594,
             0.00024368614, 0.000108335437),
        (1024, "deepseek", "gate-up"):
            ([65183, 65212], 1.224940982167, 0.916102543712,
             7.629395e-06, 2.58169e-07),
    }
    expected_tests = {
        "cpu_debug": {"passed": 305, "total": 305},
        "asan_ubsan": {"passed": 303, "total": 303},
        "pytorch_enabled_cpu": {"passed": 279, "total": 279},
        "hip_full_configuration": {
            "passed": 476, "total": 476, "conditional_skips": 3},
        "hip_label": {"passed": 161, "total": 161},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {key: 0 for key in expected}
    for row in raw:
        key = (int(row.get("rows", 0)),
               row.get("model"), row.get("projection"))
        if key in counts:
            counts[key] += 1
    evidence_changed = False
    for key, values in expected.items():
        row = comparisons.get(key, {})
        if row.get("solution_indices") != values[0] or \
                row.get("algorithm_count") != 10227 or \
                row.get("passing_candidates") != 64 or \
                abs(float(row.get(
                    "user_arguments_event_speedup_median", 0.0)) -
                    values[1]) > 1.0e-6 or \
                abs(float(row.get(
                    "reinitialized_event_speedup_median", 0.0)) -
                    values[2]) > 1.0e-6 or \
                abs(float(row.get("maximum_absolute_error", 0.0)) -
                    values[3]) > 1.0e-9 or \
                abs(float(row.get("maximum_rms_error", 0.0)) -
                    values[4]) > 1.0e-9:
            evidence_changed = True
            break
    if summary.get("status") != "pass" or len(raw) != 24 or \
            summary.get("raw_processes") != 24 or \
            summary.get("capability_gate") is not True or \
            summary.get("reinitialization_faster_cases") != 0 or \
            set(comparisons) != set(expected) or evidence_changed or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 77 or \
            verification.get("formal_processes") != 24 or \
            verification.get("formal_comparisons") != 8 or \
            verification.get("capability_gate") is not True or \
            verification.get("reinitialization_faster_cases") != 0:
        errors.append("BF16 grouped shape matrix evidence changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "bf16_grouped_shape_matrix.py").read_text(encoding="utf-8")
    contract = (REPOSITORY / "python/tests/"
                "test_bf16_grouped_shape_matrix.py").read_text(
                    encoding="utf-8")
    for token, document in (
            ("ROWS = (256, 1024)", runner),
            ("PROJECTIONS = ((\"qkv\", \"model\"), (\"gate-up\", \"bf16\"))",
             runner),
            ("solution_indices", runner),
            ("reinitialization_faster_cases", contract)):
        if token not in document:
            errors.append("BF16 grouped shape matrix runner/test contract changed")
            break
    ratios = [
        float(row["user_arguments_event_speedup_median"])
        for row in comparisons.values()]
    return len(raw), min(ratios, default=0.0), \
        max(ratios, default=0.0), \
        int(summary.get("reinitialization_faster_cases", -1))


def validate_bf16_grouped_shape_models(
        errors: list[str]) -> tuple[int, float, float, float]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-bf16-grouped-shape-models")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {
        (row.get("model"), row.get("case")): row
        for row in summary.get("comparisons", [])}
    expected = {
        ("qwen2.5-0.5b", "b1t256"):
            (60913.764526327, 67459.263827594, 1.1074551762178155,
             1.0017574664585, 212.039605,
             0.10462665557861328, 0.021174594473784277),
        ("qwen2.5-0.5b", "b1t1024"):
            (111465.849726487, 114589.355615711, 1.0280220883516198,
             1.0064369239735418, 208.825208,
             0.15857377648353577, 0.03289814430513995),
        ("qwen2.5-0.5b", "b2t512"):
            (135291.347935973, 139498.928024302, 1.031100141675876,
             1.006573036018949, 208.117006,
             0.12460708618164062, 0.024704278749459706),
        ("deepseek-r1-distill-qwen-1.5b", "b1t256"):
            (34201.605209589, 36785.508670732, 1.0755491868088858,
             1.0008758462610199, 209.429043,
             0.050108909606933594, 0.008701945071299688),
        ("deepseek-r1-distill-qwen-1.5b", "b1t1024"):
            (61699.696490516, 63006.41637901, 1.0211787085321378,
             1.0033806192776318, 208.564852,
             0.03344884514808655, 0.007308861071778848),
        ("deepseek-r1-distill-qwen-1.5b", "b2t512"):
            (66321.754681918, 67803.745184891, 1.022345465829134,
             1.0033985253333988, 212.173384,
             0.06598424911499023, 0.009590439001201432),
    }
    expected_tests = {
        "cpu_debug": {"passed": 307, "total": 307},
        "asan_ubsan": {"passed": 305, "total": 305},
        "pytorch_enabled_cpu": {"passed": 281, "total": 281},
        "hip_full_configuration": {
            "passed": 478, "total": 478, "conditional_skips": 3},
        "hip_label": {"passed": 161, "total": 161},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {(model, case, policy): 0
              for model, case in expected
              for policy in ("baseline", "both")}
    for row in raw:
        key = (row.get("model"), row.get("case"), row.get("policy"))
        if key in counts:
            counts[key] += 1
    fields = (
        "baseline_tokens_per_second", "both_tokens_per_second",
        "speedup", "peak_ratio", "combined_kernel_setup_ms",
        "maximum_absolute_logit_difference",
        "maximum_rms_logit_difference")
    evidence_changed = False
    for key, values in expected.items():
        row = comparisons.get(key, {})
        if row.get("finite_complete_logits") is not True or \
                row.get("top_rows_equal") is not True or \
                any(abs(float(row.get(field, 0.0)) - value) > 1.0e-6
                    for field, value in zip(fields, values, strict=True)):
            evidence_changed = True
            break
    if summary.get("status") != "pass" or len(raw) != 36 or \
            summary.get("raw_processes") != 36 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not True or \
            summary.get("memory_gate") is not True or \
            summary.get("setup_gate") is not True or \
            set(comparisons) != set(expected) or evidence_changed or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 79 or \
            verification.get("formal_processes") != 36 or \
            verification.get("formal_comparisons") != 6 or \
            verification.get("batch_cli_regression") is not True or \
            any(verification.get(gate) is not True for gate in (
                "correctness_gate", "performance_gate",
                "memory_gate", "setup_gate")):
        errors.append("BF16 grouped shape model evidence changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_bf16_grouped_shape_models.py").read_text(
                  encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    cli_test = (REPOSITORY / "python/tests/"
                "test_hf_cli_batch_logits.py").read_text(encoding="utf-8")
    fixture = (REPOSITORY / "tests/io/hf_cli_fixture.cpp").read_text(
        encoding="utf-8")
    for token, document in (
            ("CASES = (", runner),
            ("row_top_indices", runner),
            ("prefill logits shape does not match batch export contract", cli),
            ("batch_last[8:] == batch_one", cli_test),
            ("qwen_style_weight_mapping", fixture)):
        if token not in document:
            errors.append("BF16 grouped shape model/CLI contract changed")
            break
    ratios = [float(row["speedup"]) for row in comparisons.values()]
    peaks = [float(row["peak_ratio"]) for row in comparisons.values()]
    return len(raw), min(ratios, default=0.0), \
        max(ratios, default=0.0), max(peaks, default=0.0)


def validate_bf16_grouped_composed_profile(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-bf16-grouped-composed-profile")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b":
            (5733181.6, 5680368.2, 1.0092975311001846,
             217.0, 145.0, 72.0, 3139070.2, 2656657.6,
             1.1815862909845816),
        "deepseek-r1-distill-qwen-1.5b":
            (10504172.2, 10160056.200000001, 1.0338694976903768,
             253.0, 169.0, 84.0, 6602615.3999999985, 6007167.0,
             1.0991229975793912),
    }
    fields = (
        "baseline_total_kernel_ns", "composed_total_kernel_ns",
        "total_kernel_speedup", "baseline_gemm_calls",
        "composed_gemm_calls", "gemm_calls_saved",
        "baseline_gemm_ns", "composed_gemm_ns", "gemm_time_speedup")
    evidence_changed = False
    for model, values in expected.items():
        row = comparisons.get(model, {})
        if any(abs(float(row.get(field, 0.0)) - value) > 1.0e-6
               for field, value in zip(fields, values, strict=True)):
            evidence_changed = True
            break
        directory = data / model
        for filename in (
                "one-step-kernel-stats.csv",
                "three-step-kernel-stats.csv"):
            if not (directory / filename).is_file():
                errors.append(
                    f"composed profile file missing: {model}/{filename}")
        delta = json.loads((directory / "profile-delta.json").read_text(
            encoding="utf-8"))
        if delta.get("track") != "inference_prefill_kernel_phase_delta" or \
                delta.get("many_step_count") != 6 or \
                delta.get("derived_steps") != 5:
            errors.append("composed phase-delta contract changed")
    expected_tests = {
        "cpu_debug": {"passed": 307, "total": 307},
        "asan_ubsan": {"passed": 305, "total": 305},
        "pytorch_enabled_cpu": {"passed": 281, "total": 281},
        "hip_full_configuration": {
            "passed": 478, "total": 478, "conditional_skips": 3},
        "hip_label": {"passed": 161, "total": 161},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    if summary.get("status") != "pass" or \
            summary.get("profile_processes") != 4 or \
            summary.get("derived_forwards") != 10 or \
            set(comparisons) != set(expected) or evidence_changed or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 79 or \
            verification.get("profile_processes") != 4 or \
            verification.get("derived_forwards") != 10 or \
            verification.get("gemm_calls_saved") != 156:
        errors.append("BF16 grouped composed profile evidence changed")
    profiler = (REPOSITORY / "benchmarks/single_gpu/"
                "profile_step_delta.py").read_text(encoding="utf-8")
    if "inference_prefill_kernel_phase_delta" not in profiler:
        errors.append("inference phase-delta source contract changed")
    totals = [float(row["total_kernel_speedup"])
              for row in comparisons.values()]
    return int(summary.get("profile_processes", 0)), \
        int(sum(row["gemm_calls_saved"] for row in comparisons.values())), \
        min(totals, default=0.0), max(totals, default=0.0)


def validate_hf_strided_copy_sources(
        errors: list[str]) -> tuple[int, int, int, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-hf-strided-copy-sources")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b": (
            96, 100663296,
            {"attention.core": {"bytes": 44040192, "calls": 24},
             "attention.layout": {"bytes": 56623104, "calls": 72}}),
        "deepseek-r1-distill-qwen-1.5b": (
            112, 205520896,
            {"attention.core": {"bytes": 88080384, "calls": 28},
             "attention.layout": {"bytes": 117440512, "calls": 84}}),
    }
    counts = {name: 0 for name in expected}
    for row in raw:
        name = row.get("model")
        if name in counts:
            counts[name] += 1
    if summary.get("status") != "pass" or len(raw) != 6 or \
            summary.get("raw_processes") != 6 or \
            summary.get("attribution_gate") is not True or \
            set(comparisons) != set(expected) or \
            any(row.get("calls") != expected[name][0] or
                row.get("bytes") != expected[name][1] or
                row.get("record_count") != 3 or
                row.get("source_totals") != expected[name][2]
                for name, row in comparisons.items()) or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("registered_test_files") != 80 or \
            verification.get("formal_processes") != 6 or \
            verification.get("formal_comparisons") != 2 or \
            verification.get("attribution_gate") is not True:
        errors.append("HF strided-copy source evidence changed")
    expected_tests = {
        "cpu_debug": {"passed": 308, "total": 308},
        "asan_ubsan": {"passed": 306, "total": 306},
        "pytorch_enabled_cpu": {"passed": 282, "total": 282},
        "hip_full_configuration": {
            "passed": 479, "total": 479, "conditional_skips": 3},
        "hip_label": {"passed": 161, "total": 161},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    if verification.get("tests") != expected_tests:
        errors.append("HF strided-copy verification counts changed")
    header = (REPOSITORY / "include/microllm/runtime/"
              "diagnostics.h").read_text(encoding="utf-8")
    runtime = (REPOSITORY / "src/runtime/runtime.cpp").read_text(
        encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "hf_strided_copy_sources.py").read_text(encoding="utf-8")
    tests = (REPOSITORY / "tests/runtime/runtime_test.cpp").read_text(
        encoding="utf-8")
    for token, document in (
            ("AllocationSource source", header),
            ("record.source == active_allocation_source", runtime),
            ("--strided-copy-diagnostics", cli),
            ("strided_copy_records", cli),
            ("source_totals", runner),
            ("AllocationSource::AttentionLayout", tests)):
        if token not in document:
            errors.append("HF strided-copy source/test contract changed")
            break
    total_calls = sum(row["calls"] for row in comparisons.values())
    total_bytes = sum(row["bytes"] for row in comparisons.values())
    return len(raw), total_calls, total_bytes, \
        sum(row["source_totals"]["attention.layout"]["bytes"]
            for row in comparisons.values())


def validate_inference_bthd_attention(
        errors: list[str]) -> tuple[int, int, float, float, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-inference-bthd-attention")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    performance = [json.loads(line) for line in (
        data / "performance-raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    diagnostics = [json.loads(line) for line in (
        data / "diagnostic-raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b":
            (99791.089692582, 111226.387800099, 1.1145923763609031,
             1314589840, 1310395536, 0.9968094200393334,
             4194304, 96, 0, 100663296, 0, 0.0, 0.0),
        "deepseek-r1-distill-qwen-1.5b":
            (52593.248805984, 57513.450452735, 1.0935519626274006,
             4570125792, 4562785760, 0.9983939102917366,
             7340032, 112, 0, 205520896, 0, 0.0, 0.0),
    }
    fields = (
        "baseline_tokens_per_second", "bthd_tokens_per_second",
        "speedup", "baseline_peak_bytes", "bthd_peak_bytes",
        "peak_ratio", "peak_bytes_saved", "baseline_strided_calls",
        "bthd_strided_calls", "baseline_strided_bytes",
        "bthd_strided_bytes", "maximum_absolute_logit_difference",
        "maximum_rms_logit_difference")
    counts = {(name, policy): 0 for name in expected
              for policy in ("baseline", "bthd")}
    for row in performance:
        key = (row.get("model"), row.get("policy"))
        if key in counts:
            counts[key] += 1
    diagnostic_counts = counts.copy()
    for key in diagnostic_counts:
        diagnostic_counts[key] = 0
    for row in diagnostics:
        key = (row.get("model"), row.get("policy"))
        if key in diagnostic_counts:
            diagnostic_counts[key] += 1
    if summary.get("status") != "pass" or len(performance) != 12 or \
            len(diagnostics) != 12 or \
            summary.get("performance_processes") != 12 or \
            summary.get("diagnostic_processes") != 12 or \
            summary.get("correctness_gate") is not True or \
            summary.get("copy_elimination_gate") is not True or \
            summary.get("performance_gate") is not True or \
            summary.get("memory_gate") is not True or \
            set(comparisons) != set(expected) or \
            any(any(abs(float(row.get(field, 0.0)) - value) > 1.0e-6
                    for field, value in zip(
                        fields, expected[name], strict=True))
                for name, row in comparisons.items()) or \
            any(count != 3 for count in counts.values()) or \
            any(count != 3 for count in diagnostic_counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("registered_test_files") != 81 or \
            verification.get("performance_processes") != 12 or \
            verification.get("diagnostic_processes") != 12 or \
            verification.get("formal_comparisons") != 2 or \
            any(verification.get(gate) is not True for gate in (
                "correctness_gate", "copy_elimination_gate",
                "performance_gate", "memory_gate")):
        errors.append("inference BTHD Attention evidence changed")
    expected_tests = {
        "cpu_debug": {"passed": 309, "total": 309},
        "asan_ubsan": {"passed": 307, "total": 307},
        "pytorch_enabled_cpu": {"passed": 283, "total": 283},
        "hip_full_configuration": {
            "passed": 481, "total": 481, "conditional_skips": 3},
        "hip_label": {"passed": 162, "total": 162},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    if verification.get("tests") != expected_tests:
        errors.append("inference BTHD verification counts changed")
    model_source = (REPOSITORY / "src/model/model.cpp").read_text(
        encoding="utf-8")
    ops_header = (REPOSITORY / "include/microllm/ops/ops.h").read_text(
        encoding="utf-8")
    cli = (REPOSITORY / "apps/hf_infer.cpp").read_text(encoding="utf-8")
    hip_tests = (REPOSITORY / "tests/ops/hip_ops_test.cpp").read_text(
        encoding="utf-8")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_inference_bthd_attention.py").read_text(
                  encoding="utf-8")
    for token, document in (
            ("inference_bthd_attention_enabled", ops_header),
            ("rope_split_half_bias_bthd", model_source),
            ("causal_gqa_attention_bthd", model_source),
            ("--inference-bthd-attention", cli),
            ("RemovesFourLayoutCopiesPerBlockAndMatchesFallback", hip_tests),
            ("copy_elimination_gate", runner)):
        if token not in document:
            errors.append("inference BTHD source/test contract changed")
            break
    ratios = [float(row["speedup"]) for row in comparisons.values()]
    bytes_removed = sum(
        int(row["baseline_strided_bytes"]) for row in comparisons.values())
    return len(performance), len(diagnostics), \
        min(ratios, default=0.0), max(ratios, default=0.0), bytes_removed


def validate_inference_bthd_shape_models(
        errors: list[str]) -> tuple[int, int, float, float, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-inference-bthd-shape-models")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    performance = [json.loads(line) for line in (
        data / "performance-raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    diagnostics = [json.loads(line) for line in (
        data / "diagnostic-raw.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    comparisons = {(row.get("model"), row.get("case")): row
                   for row in summary.get("comparisons", [])}
    expected = {
        ("qwen2.5-0.5b", "b1t256"):
            (67211.591647153, 76761.126630252, 1.1420816670022051,
             0.9983669331269008, 2097152, 0, 0, 0),
        ("qwen2.5-0.5b", "b1t1024"):
            (114527.582223422, 125848.852971277, 1.0988519143429512,
             0.9939971149540556, 8388608, 0, 0, 0),
        ("qwen2.5-0.5b", "b2t512"):
            (139500.349527389, 151381.881522449, 1.0851720589612375,
             0.9938710097773484, 8388608, 0, 1, 7168),
        ("deepseek-r1-distill-qwen-1.5b", "b1t256"):
            (36834.067045887, 40321.673712354, 1.0946842677492612,
             0.9991891645724786, 3670016, 0, 0, 0),
        ("deepseek-r1-distill-qwen-1.5b", "b1t1024"):
            (63042.060135723, 68804.868438977, 1.0914121190019375,
             0.9968611016473216, 14680064, 0, 0, 0),
        ("deepseek-r1-distill-qwen-1.5b", "b2t512"):
            (67916.421680063, 74186.205579307, 1.0923161695529156,
             0.9968445322213607, 14680064, 0, 1, 12288),
    }
    fields = (
        "baseline_tokens_per_second", "bthd_tokens_per_second",
        "speedup", "peak_ratio", "peak_bytes_saved",
        "bthd_attention_strided_calls",
        "bthd_residual_strided_calls", "bthd_residual_strided_bytes")
    if summary.get("status") != "pass" or len(performance) != 36 or \
            len(diagnostics) != 6 or \
            summary.get("performance_processes") != 36 or \
            summary.get("diagnostic_processes") != 6 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not True or \
            summary.get("memory_gate") is not True or \
            summary.get("copy_elimination_gate") is not True or \
            set(comparisons) != set(expected) or \
            any(row.get("finite_complete_logits") is not True or
                row.get("top_rows_equal") is not True or
                float(row.get("maximum_absolute_logit_difference", 1.0)) != 0 or
                float(row.get("maximum_rms_logit_difference", 1.0)) != 0 or
                any(abs(float(row.get(field, 0.0)) - value) > 1.0e-6
                    for field, value in zip(
                        fields, expected[key], strict=True))
                for key, row in comparisons.items()) or \
            verification.get("status") != "pass" or \
            verification.get("registered_test_files") != 82 or \
            verification.get("performance_processes") != 36 or \
            verification.get("diagnostic_processes") != 6 or \
            verification.get("formal_comparisons") != 6 or \
            any(verification.get(gate) is not True for gate in (
                "correctness_gate", "performance_gate",
                "memory_gate", "copy_elimination_gate")):
        errors.append("inference BTHD shape-model evidence changed")
    expected_tests = {
        "cpu_debug": {"passed": 310, "total": 310},
        "asan_ubsan": {"passed": 308, "total": 308},
        "pytorch_enabled_cpu": {"passed": 284, "total": 284},
        "hip_full_configuration": {
            "passed": 482, "total": 482, "conditional_skips": 3},
        "hip_label": {"passed": 162, "total": 162},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    if verification.get("tests") != expected_tests:
        errors.append("inference BTHD shape verification counts changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_inference_bthd_shape_models.py").read_text(
                  encoding="utf-8")
    contract = (REPOSITORY / "python/tests/"
                "test_inference_bthd_shape_models.py").read_text(
                    encoding="utf-8")
    for token, document in (
            ("attention_strided_calls", runner),
            ("residual_strided_calls", runner),
            ("row_top", runner),
            ("diagnostic_processes", contract)):
        if token not in document:
            errors.append("inference BTHD shape runner/test contract changed")
            break
    ratios = [float(row["speedup"]) for row in comparisons.values()]
    residual_bytes = sum(
        int(row["bthd_residual_strided_bytes"])
        for row in comparisons.values())
    return len(performance), len(diagnostics), \
        min(ratios, default=0.0), max(ratios, default=0.0), residual_bytes


def validate_inference_bthd_profile(
        errors: list[str]) -> tuple[int, float, float, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-inference-bthd-profile")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    comparisons = {row.get("model"): row
                   for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b":
            (5680368.2, 4858257.2, 1.169219324164229,
             96.0, 0.0, 486340.6, 0.0),
        "deepseek-r1-distill-qwen-1.5b":
            (10160056.200000001, 9085212.2, 1.1183069780142287,
             112.0, 0.0, 754890.6, 0.0),
    }
    fields = (
        "baseline_total_kernel_ns", "bthd_total_kernel_ns",
        "total_kernel_speedup", "baseline_strided_calls",
        "bthd_strided_calls", "baseline_strided_ns", "bthd_strided_ns")
    for model in expected:
        directory = data / model
        for filename in (
                "one-step-kernel-stats.csv",
                "three-step-kernel-stats.csv"):
            if not (directory / filename).is_file():
                errors.append(f"BTHD profile file missing: {model}/{filename}")
        delta = json.loads((directory / "profile-delta.json").read_text(
            encoding="utf-8"))
        if delta.get("track") != "inference_prefill_kernel_phase_delta":
            errors.append("BTHD profile track changed")
    expected_tests = {
        "cpu_debug": {"passed": 310, "total": 310},
        "asan_ubsan": {"passed": 308, "total": 308},
        "pytorch_enabled_cpu": {"passed": 284, "total": 284},
        "hip_full_configuration": {
            "passed": 482, "total": 482, "conditional_skips": 3},
        "hip_label": {"passed": 162, "total": 162},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    if summary.get("status") != "pass" or \
            summary.get("profile_processes") != 4 or \
            summary.get("derived_forwards") != 10 or \
            set(comparisons) != set(expected) or \
            any(any(abs(float(row.get(field, 0.0)) - value) > 1.0e-6
                    for field, value in zip(fields, expected[model], strict=True))
                for model, row in comparisons.items()) or \
            verification.get("status") != "pass" or \
            verification.get("tests") != expected_tests or \
            verification.get("registered_test_files") != 82 or \
            verification.get("profile_processes") != 4 or \
            verification.get("derived_forwards") != 10 or \
            verification.get("strided_calls") != 0:
        errors.append("inference BTHD profile evidence changed")
    ratios = [float(row["total_kernel_speedup"])
              for row in comparisons.values()]
    return int(summary.get("profile_processes", 0)), \
        min(ratios, default=0.0), max(ratios, default=0.0), \
        int(sum(row["baseline_strided_calls"]
                for row in comparisons.values()))


def validate_inference_bthd_bf16_qk(
        errors: list[str]) -> tuple[int, int, float, float, int]:
    initial_dir = REPOSITORY / (
        "benchmarks/results/2026-08-24-inference-bthd-bf16-qk")
    formal_dir = REPOSITORY / (
        "benchmarks/results/2026-08-24-inference-bthd-bf16-qk-formal5")
    initial = json.loads((initial_dir / "summary.json").read_text(
        encoding="utf-8"))
    formal = json.loads((formal_dir / "summary.json").read_text(
        encoding="utf-8"))
    profile = json.loads((initial_dir / "profile-summary.json").read_text(
        encoding="utf-8"))
    initial_verification = json.loads((initial_dir / "verification.json").read_text(
        encoding="utf-8"))
    formal_verification = json.loads((formal_dir / "verification.json").read_text(
        encoding="utf-8"))
    initial_raw = [json.loads(line) for line in (initial_dir / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    formal_raw = [json.loads(line) for line in (formal_dir / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    initial_rows = {row.get("model"): row
                    for row in initial.get("comparisons", [])}
    formal_rows = {row.get("model"): row
                   for row in formal.get("comparisons", [])}
    profile_rows = {row.get("model"): row
                    for row in profile.get("comparisons", [])}
    expected_initial = {
        "qwen2.5-0.5b": (111056.002461973, 113579.568810252,
                          1.0227233674212532),
        "deepseek-r1-distill-qwen-1.5b":
            (57495.162601822, 57886.64758774, 1.0068090073703975),
    }
    expected_formal = {
        "qwen2.5-0.5b": (110960.70253035, 113440.757890613,
                          1.0223507539489907),
        "deepseek-r1-distill-qwen-1.5b":
            (56979.167948791, 58333.105089869, 1.0237619675719876),
    }
    expected_profile = {
        "qwen2.5-0.5b":
            (5128389.0, 4754117.4, 1.0787257798892387,
             144.0, 96.0, 48.0, 639242.0, 334101.0),
        "deepseek-r1-distill-qwen-1.5b":
            (9463387.6, 8927681.0, 1.0600051233909455,
             168.0, 112.0, 56.0, 926838.0, 513776.4),
    }
    tests = {
        "cpu_debug": {"passed": 311, "total": 311},
        "asan_ubsan": {"passed": 309, "total": 309},
        "pytorch_enabled_cpu": {"passed": 285, "total": 285},
        "hip_full_configuration": {
            "passed": 484, "total": 484, "conditional_skips": 3},
        "hip_label": {"passed": 163, "total": 163},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    def values_match(rows: dict, expected: dict, fields: tuple[str, ...]) -> bool:
        return set(rows) == set(expected) and all(
            all(abs(float(rows[name].get(field, 0.0)) - value) <= 1.0e-6
                for field, value in zip(fields, expected[name], strict=True))
            for name in expected)
    initial_ok = values_match(
        initial_rows, expected_initial,
        ("fp32_boundary_tokens_per_second", "bf16_qk_tokens_per_second",
         "speedup"))
    formal_ok = values_match(
        formal_rows, expected_formal,
        ("fp32_boundary_tokens_per_second", "bf16_qk_tokens_per_second",
         "speedup"))
    profile_ok = values_match(
        profile_rows, expected_profile,
        ("baseline_total_kernel_ns", "bf16_qk_total_kernel_ns",
         "total_kernel_speedup", "baseline_cast_calls",
         "bf16_qk_cast_calls", "cast_calls_removed",
         "baseline_cast_ns", "bf16_qk_cast_ns"))
    for model in expected_profile:
        for policy in ("fp32-boundary", "bf16-qk"):
            directory = initial_dir / "profile" / model / policy
            for filename in ("one-step-kernel-stats.csv",
                             "three-step-kernel-stats.csv",
                             "profile-delta.json"):
                if not (directory / filename).is_file():
                    errors.append(
                        f"BTHD BF16 Q/K profile file missing: {model}/{policy}/{filename}")
    if initial.get("status") != "pass" or len(initial_raw) != 12 or \
            initial.get("processes") != 12 or \
            initial.get("correctness_gate") is not True or \
            initial.get("routing_gate") is not True or \
            initial.get("performance_gate") is not False or \
            initial.get("memory_gate") is not True or not initial_ok or \
            formal.get("status") != "pass" or len(formal_raw) != 20 or \
            formal.get("processes") != 20 or \
            any(formal.get(gate) is not True for gate in (
                "correctness_gate", "routing_gate", "performance_gate",
                "memory_gate")) or not formal_ok or \
            profile.get("status") != "pass" or \
            profile.get("profile_processes") != 8 or \
            profile.get("derived_forwards") != 20 or \
            profile.get("cast_elimination_gate") is not True or \
            profile.get("kernel_performance_gate") is not True or \
            not profile_ok or \
            initial_verification.get("status") != "pass" or \
            formal_verification.get("status") != "pass" or \
            initial_verification.get("tests") != tests or \
            formal_verification.get("tests") != tests or \
            initial_verification.get("registered_test_files") != 83 or \
            formal_verification.get("registered_test_files") != 83:
        errors.append("inference BTHD BF16 Q/K evidence changed")
    for raw, runs in ((initial_raw, 3), (formal_raw, 5)):
        counts = {(model, policy): 0 for model in expected_formal
                  for policy in ("fp32-boundary", "bf16-qk")}
        for row in raw:
            key = (row.get("model"), row.get("policy"))
            if key in counts:
                counts[key] += 1
            blocks = 24 if row.get("model") == "qwen2.5-0.5b" else 28
            expected_retained = blocks * 7 if row.get("policy") == "bf16-qk" else 0
            if int(row.get(
                    "bf16_grouped_qkv_retained_query_key_dispatches", -1)) != \
                    expected_retained:
                errors.append("BTHD BF16 Q/K retained dispatch count changed")
                break
        if any(count != runs for count in counts.values()):
            errors.append("BTHD BF16 Q/K process matrix changed")
    sources = (
        ("--inference-bthd-bf16-qk", REPOSITORY / "apps/hf_infer.cpp"),
        ("retain_query_key_bf16", REPOSITORY / "src/ops/optimized.cpp"),
        ("hip_bfloat16", REPOSITORY / "src/ops/hip/basic_kernels.hip"),
        ("RetainsGroupedQueryKeyInBf16", REPOSITORY / "tests/ops/hip_ops_test.cpp"),
        ("cast_elimination_gate", REPOSITORY / "benchmarks/single_gpu/"
         "summarize_inference_bthd_bf16_qk_profile.py"),
    )
    if any(token not in path.read_text(encoding="utf-8")
           for token, path in sources):
        errors.append("inference BTHD BF16 Q/K source/test contract changed")
    ratios = [float(row["speedup"]) for row in formal_rows.values()]
    casts = int(sum(row["cast_calls_removed"] for row in profile_rows.values()))
    return len(initial_raw), len(formal_raw), min(ratios), max(ratios), casts


def validate_inference_bthd_bf16_qk_shapes(
        errors: list[str]) -> tuple[int, int, float, float, int]:
    pilot_dir = REPOSITORY / (
        "benchmarks/results/2026-08-24-inference-bthd-bf16-qk-shapes-pilot3")
    formal_dir = REPOSITORY / (
        "benchmarks/results/2026-08-24-inference-bthd-bf16-qk-shapes-formal5")
    pilot = json.loads((pilot_dir / "summary.json").read_text(encoding="utf-8"))
    formal = json.loads((formal_dir / "summary.json").read_text(encoding="utf-8"))
    pilot_raw = [json.loads(line) for line in (pilot_dir / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    formal_raw = [json.loads(line) for line in (formal_dir / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    pilot_verification = json.loads((pilot_dir / "verification.json").read_text(
        encoding="utf-8"))
    formal_verification = json.loads((formal_dir / "verification.json").read_text(
        encoding="utf-8"))
    rows = {(row.get("model"), row.get("case")): row
            for row in formal.get("comparisons", [])}
    expected = {
        ("qwen2.5-0.5b", "b1t256"):
            (76738.935759259, 78609.853524049, 1.02438029334495),
        ("qwen2.5-0.5b", "b1t1024"):
            (125896.16492496, 127632.634986718, 1.0137928749679788),
        ("qwen2.5-0.5b", "b2t512"):
            (152932.15902543, 154883.210496629, 1.0127576271964784),
        ("deepseek-r1-distill-qwen-1.5b", "b1t256"):
            (40106.206247419, 40861.120012238, 1.0188228664701384),
        ("deepseek-r1-distill-qwen-1.5b", "b1t1024"):
            (68935.173322616, 69983.221659293, 1.0152033901731434),
        ("deepseek-r1-distill-qwen-1.5b", "b2t512"):
            (74101.328182365, 75542.246152786, 1.019445238105245),
    }
    fields = ("fp32_boundary_tokens_per_second",
              "bf16_qk_tokens_per_second", "speedup")
    values_ok = set(rows) == set(expected) and all(
        all(abs(float(rows[key].get(field, 0.0)) - value) <= 1.0e-6
            for field, value in zip(fields, expected[key], strict=True)) and
        rows[key].get("finite_complete_logits") is True and
        rows[key].get("top_rows_equal") is True and
        float(rows[key].get("maximum_absolute_logit_difference", 1.0)) == 0 and
        float(rows[key].get("maximum_rms_logit_difference", 1.0)) == 0 and
        float(rows[key].get("peak_ratio", 0.0)) == 1.0
        for key in expected)
    tests = {
        "cpu_debug": {"passed": 312, "total": 312},
        "asan_ubsan": {"passed": 310, "total": 310},
        "pytorch_enabled_cpu": {"passed": 286, "total": 286},
        "hip_full_configuration": {
            "passed": 485, "total": 485, "conditional_skips": 3},
        "hip_label": {"passed": 163, "total": 163},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    if pilot.get("status") != "pass" or len(pilot_raw) != 36 or \
            pilot.get("processes") != 36 or \
            pilot.get("correctness_gate") is not True or \
            pilot.get("routing_gate") is not True or \
            pilot.get("performance_gate") is not False or \
            pilot.get("memory_gate") is not True or \
            formal.get("status") != "pass" or len(formal_raw) != 60 or \
            formal.get("processes") != 60 or \
            any(formal.get(gate) is not True for gate in (
                "correctness_gate", "routing_gate", "performance_gate",
                "memory_gate")) or not values_ok or \
            pilot_verification.get("status") != "pass" or \
            pilot_verification.get("registered_test_files") != 84 or \
            formal_verification.get("status") != "pass" or \
            formal_verification.get("registered_test_files") != 84 or \
            formal_verification.get("tests") != tests:
        errors.append("inference BTHD BF16 Q/K shape evidence changed")
    cases = ("b1t256", "b1t1024", "b2t512")
    models = ("qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b")
    for raw, runs in ((pilot_raw, 3), (formal_raw, 5)):
        counts = {(model, case, policy): 0 for model in models for case in cases
                  for policy in ("fp32-boundary", "bf16-qk")}
        for row in raw:
            key = (row.get("model"), row.get("case"), row.get("policy"))
            if key in counts:
                counts[key] += 1
            blocks = 24 if row.get("model") == "qwen2.5-0.5b" else 28
            expected_dispatches = blocks * 7
            expected_retained = (expected_dispatches
                                 if row.get("policy") == "bf16-qk" else 0)
            if int(row.get("bf16_grouped_qkv_dispatches", -1)) != \
                    expected_dispatches or int(row.get(
                        "bf16_grouped_qkv_retained_query_key_dispatches", -1)) != \
                    expected_retained:
                errors.append("BTHD BF16 Q/K shape dispatch changed")
                break
        if any(count != runs for count in counts.values()):
            errors.append("BTHD BF16 Q/K shape process matrix changed")
    runner = (REPOSITORY / "benchmarks/single_gpu/"
              "compare_inference_bthd_bf16_qk_shapes.py").read_text(
                  encoding="utf-8")
    contract = (REPOSITORY / "python/tests/"
                "test_inference_bthd_bf16_qk_shapes.py").read_text(
                    encoding="utf-8")
    for token, document in (("b2t512", runner), ("row_top", runner),
                            ("retained_query_key_dispatches", runner),
                            ("processes", contract)):
        if token not in document:
            errors.append("BTHD BF16 Q/K shape runner/test contract changed")
            break
    ratios = [float(row["speedup"]) for row in rows.values()]
    return len(pilot_raw), len(formal_raw), min(ratios), max(ratios), len(rows)


def validate_causal_softmax_128_discard(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-causal-softmax-128-operator")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = {(row.get("family"), int(row.get("sequence", 0))): row
            for row in summary.get("comparisons", [])}
    expected = {
        ("qwen", 256): 1.0168225833456122,
        ("qwen", 512): 1.025461463341972,
        ("qwen", 1024): 1.012717855463594,
        ("deepseek", 256): 1.0062595234885816,
        ("deepseek", 512): 1.0070511094441879,
        ("deepseek", 1024): 1.0214072958092015,
    }
    tests = {
        "cpu_debug": {"passed": 313, "total": 313},
        "asan_ubsan": {"passed": 311, "total": 311},
        "pytorch_enabled_cpu": {"passed": 287, "total": 287},
        "hip_full_configuration": {
            "passed": 487, "total": 487, "conditional_skips": 3},
        "hip_label": {"passed": 164, "total": 164},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {(family, sequence, policy): 0
              for family, sequence in expected
              for policy in ("threads256", "threads128")}
    for row in raw:
        key = (row.get("family"), int(row.get("sequence", 0)),
               row.get("policy"))
        if key in counts:
            counts[key] += 1
    if summary.get("status") != "pass" or len(raw) != 36 or \
            summary.get("processes") != 36 or \
            summary.get("correctness_gate") is not True or \
            summary.get("universal_performance_gate") is not False or \
            summary.get("t512_performance_gate") is not False or \
            set(rows) != set(expected) or \
            any(abs(float(rows[key].get("event_speedup", 0.0)) - value) > 1.0e-9
                for key, value in expected.items()) or \
            any(float(row.get("maximum_absolute_error", 1.0)) > 2.0e-6 or
                float(row.get("maximum_rms_error", 1.0)) > 1.0e-7
                for row in rows.values()) or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("registered_test_files") != 85 or \
            verification.get("tests") != tests or \
            verification.get("model_gate_executed") is not False:
        errors.append("causal-softmax 128-thread rejection evidence changed")
    sources = (
        ("CausalSoftmaxImplementation::Rows128",
         REPOSITORY / "benchmarks/micro/benchmark_causal_softmax.cpp"),
        ("blockDim.x / 2", REPOSITORY / "src/ops/hip/basic_kernels.hip"),
        ("Optional128ThreadRowsMatchCpu",
         REPOSITORY / "tests/ops/hip_ops_test.cpp"),
        ("universal_performance_gate",
         REPOSITORY / "benchmarks/single_gpu/compare_causal_softmax_threads.py"),
    )
    if any(token not in path.read_text(encoding="utf-8")
           for token, path in sources):
        errors.append("causal-softmax thread source/test contract changed")
    ratios = list(expected.values())
    return len(raw), sum(value >= 1.01 for value in ratios), min(ratios), max(ratios)


def validate_bf16_repeat_fusion_discard(
        errors: list[str]) -> tuple[int, int, float, float]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-bf16-repeat-operator")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = {(row.get("family"), row.get("case")): row
            for row in summary.get("comparisons", [])}
    expected = {
        ("qwen", "b1t256"): 1.2531099206101144,
        ("qwen", "b1t512"): 1.291371527991624,
        ("qwen", "b1t1024"): 0.9961762926812381,
        ("qwen", "b2t512"): 1.0040925057230385,
        ("deepseek", "b1t256"): 1.345150643548046,
        ("deepseek", "b1t512"): 1.0269528096119205,
        ("deepseek", "b1t1024"): 1.0110125919592616,
        ("deepseek", "b2t512"): 0.9948489408841099,
    }
    tests = {
        "cpu_debug": {"passed": 315, "total": 315},
        "asan_ubsan": {"passed": 313, "total": 313},
        "pytorch_enabled_cpu": {"passed": 289, "total": 289},
        "hip_full_configuration": {
            "passed": 489, "total": 489, "conditional_skips": 3},
        "hip_label": {"passed": 164, "total": 164},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    counts = {(family, case, policy): 0 for family, case in expected
              for policy in ("composed", "fused")}
    for row in raw:
        key = (row.get("family"), row.get("case"), row.get("policy"))
        if key in counts:
            counts[key] += 1
        if int(row.get("host_to_device_calls", -1)) != 0 or \
                int(row.get("device_to_host_calls", -1)) != 0:
            errors.append("BF16 repeat formal timing transferred payloads")
            break
    if summary.get("status") != "pass" or len(raw) != 48 or \
            summary.get("processes") != 48 or \
            summary.get("correctness_gate") is not True or \
            summary.get("performance_gate") is not False or \
            set(rows) != set(expected) or \
            any(abs(float(rows[key].get("event_speedup", 0.0)) - value) > 1.0e-9
                for key, value in expected.items()) or \
            any(count != 3 for count in counts.values()) or \
            verification.get("status") != "pass" or \
            verification.get("registered_test_files") != 86 or \
            verification.get("tests") != tests or \
            verification.get("model_gate_executed") is not False or \
            verification.get("invalid_host_cast_pilot_rejected") is not True:
        errors.append("BF16 repeat fusion rejection evidence changed")
    sources = (
        ("repeat_interleave_bf16_to_float",
         REPOSITORY / "include/microllm/ops/ops.h"),
        ("repeat_interleave_typed_kernel<hip_bfloat16>",
         REPOSITORY / "src/ops/hip/basic_kernels.hip"),
        ("BF16 repeat complete-output gate failed",
         REPOSITORY / "benchmarks/micro/benchmark_bf16_repeat.cpp"),
        ("performance_gate",
         REPOSITORY / "benchmarks/single_gpu/compare_bf16_repeat.py"),
    )
    if any(token not in path.read_text(encoding="utf-8")
           for token, path in sources):
        errors.append("BF16 repeat fusion source/test contract changed")
    ratios = list(expected.values())
    return len(raw), sum(value >= 1.05 for value in ratios), min(ratios), max(ratios)


def validate_post_bf16_qk_saturation(
        errors: list[str]) -> tuple[int, float, float]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-post-bf16-qk-saturation")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    rows = {row.get("model"): row
            for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b":
            (4754117.4, 0.5741639447103262, 0.10011141079519828,
             1.1112486723313861, 1.0755881810755272,
             1.0460063785011524, 1.2728645159410965),
        "deepseek-r1-distill-qwen-1.5b":
            (8927681.0, 0.6682172895738546, 0.058127480137339135,
             1.0617148063156308, 1.061062779342661,
             1.034622357226516, 1.1752813956944197),
    }
    fields = (
        "total_kernel_ns", "gemm_share", "softmax_share",
        "softmax_perfect_upper_bound", "cast_perfect_upper_bound",
        "repeat_perfect_upper_bound", "three_category_perfect_upper_bound")
    if summary.get("status") != "pass" or set(rows) != set(expected) or \
            any(any(abs(float(rows[name].get(field, 0.0)) - value) > 1.0e-9
                    for field, value in zip(fields, expected[name], strict=True))
                for name in expected) or \
            summary.get("recent_gates", {}).get("direct_bf16_qk") != "keep" or \
            summary.get("recent_gates", {}).get("causal_softmax_128") != \
                "reject_model_policy" or \
            summary.get("recent_gates", {}).get("bf16_v_cast_repeat") != \
                "reject_model_policy" or \
            verification.get("status") != "pass" or \
            verification.get("softmax_processes") != 36 or \
            verification.get("bf16_repeat_processes") != 48 or \
            verification.get("readable_fused_attention_pairs") != 2:
        errors.append("post-BF16-Q/K saturation evidence changed")
    ratios = [float(row["three_category_perfect_upper_bound"])
              for row in rows.values()]
    return len(rows), min(ratios), max(ratios)


def validate_training_add_rms_norm_discard(
        errors: list[str]) -> tuple[int, float, float, int]:
    data = REPOSITORY / (
        "benchmarks/results/2026-08-24-training-add-rms-norm-fusion")
    summary = json.loads((data / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads((data / "profile-summary.json").read_text(
        encoding="utf-8"))
    verification = json.loads((data / "verification.json").read_text(
        encoding="utf-8"))
    raw = [json.loads(line) for line in (data / "training.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = {row.get("model"): row
            for row in summary.get("comparisons", [])}
    expected = {
        "qwen2.5-0.5b": (0.9784857620624696, True),
        "deepseek-r1-distill-qwen-1.5b": (0.9979677958172398, False),
    }
    counts = {(model, policy): 0 for model in expected
              for policy in ("materialized", "fused")}
    for row in raw:
        key = (row.get("model"), row.get("policy"))
        if key in counts:
            counts[key] += 1
    expected_tests = {
        "cpu_debug": {"passed": 316, "total": 316},
        "asan_ubsan": {"passed": 314, "total": 314},
        "pytorch_enabled_cpu": {"passed": 290, "total": 290},
        "hip_full_configuration": {
            "passed": 491, "total": 491, "conditional_skips": 3},
        "hip_label": {"passed": 165, "total": 165},
        "rccl_multi_gpu": {"passed": 12, "total": 12},
        "rccl_full_label": {"passed": 14, "total": 14},
    }
    materialized = profile.get("materialized", {})
    fused = profile.get("fused", {})
    if summary.get("status") != "pass" or len(raw) != 12 or \
            summary.get("decision") != "reject layout fusion" or \
            summary.get("gate_results", {}).get("throughput") is not False or \
            summary.get("gate_results", {}).get("parameter") is not False or \
            set(rows) != set(expected) or \
            any(abs(float(rows[name].get("throughput_speedup", 0.0)) - values[0]) >
                1.0e-12 or
                rows[name].get("observed_parameter_after_equal") is not values[1]
                for name, values in expected.items()) or \
            any(count != 3 for count in counts.values()) or \
            materialized.get("kernel_dispatches") != 6903 or \
            fused.get("kernel_dispatches") != 6831 or \
            materialized.get("fp32_add_calls") != 504 or \
            fused.get("fp32_add_calls") != 432 or \
            materialized.get("rms_norm_forward_calls") != 147 or \
            fused.get("rms_norm_forward_calls") != 75 or \
            fused.get("add_rms_norm_calls") != 72 or \
            verification.get("status") != "pass" or \
            verification.get("model_route_retained") is not False or \
            verification.get("registered_test_files") != 86 or \
            verification.get("performance_processes") != 12 or \
            verification.get("diagnostic_processes") != 4 or \
            verification.get("coverage") != {
                "lines_percent": 80.0,
                "functions_percent": 87.9,
                "branches_percent": 60.7} or \
            verification.get("tests") != expected_tests:
        errors.append("training add plus RMSNorm rejection evidence changed")
    autograd_header = (REPOSITORY / "include/microllm/autograd/autograd.h").read_text(
        encoding="utf-8")
    autograd_source = (REPOSITORY / "src/autograd/autograd.cpp").read_text(
        encoding="utf-8")
    graph_test = (REPOSITORY / "tests/graph/graph_gradient_alignment_test.cpp").read_text(
        encoding="utf-8")
    hip_test = (REPOSITORY / "tests/graph/hip_graph_alignment_test.cpp").read_text(
        encoding="utf-8")
    model_and_cli = (REPOSITORY / "src/model/model.cpp").read_text(
        encoding="utf-8") + (REPOSITORY / "apps/hf_train_step.cpp").read_text(
            encoding="utf-8")
    for token, document in (
            ("std::pair<Value, Value> add_rms_norm", autograd_header),
            ("add_rms_norm_sum", autograd_source),
            ("AddRmsNormMatchesComposedForwardAndAllBranchedGradients", graph_test),
            ("AddRmsNormMatchesBranchedHipGraphAndStaysDeviceNative", hip_test)):
        if token not in document:
            errors.append("training add plus RMSNorm source/test contract changed")
            break
    if "training_add_rms_norm_fusion" in model_and_cli:
        errors.append("rejected training add plus RMSNorm model route returned")
    ratios = [values[0] for values in expected.values()]
    return len(raw), min(ratios), max(ratios), \
        materialized.get("kernel_dispatches", 0) - fused.get("kernel_dispatches", 0)


def validate_links(errors: list[str]) -> int:
    checked = 0
    for document in sorted(ROOT.rglob("*.md")):
        for raw in LINK.findall(document.read_text(encoding="utf-8")):
            target = html.unescape(raw.strip().split(maxsplit=1)[0].strip("<>"))
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            candidate = (document.parent / unquote(target.split("#", 1)[0])).resolve()
            checked += 1
            if not candidate.exists():
                errors.append(f"broken link in {document.relative_to(ROOT)}: {target}")
    return checked


def validate_assets(errors: list[str]) -> None:
    for name in ("progress.svg", "bottleneck-map.svg", "bf16-gemm.svg",
                 "bf16-model-policy.svg", "bf16-ffn-island.svg",
                 "bf16-model-inference.svg", "bf16-prefill-allocator.svg",
                 "bf16-attention.svg", "bf16-plan-cache.svg", "bf16-training.svg",
                 "bf16-training-qkv-discard.svg", "bf16-training-mirrors.svg",
                 "bf16-training-ffn-island-discard.svg",
                 "bf16-training-shape-matrix.svg",
                 "bf16-weight-gradient-routing.svg",
                 "fused-causal-gqa-training.svg",
                 "deepseek-training-shapes.svg",
                 "deepseek-context128-profile.svg",
                 "stable-gradient-buffer-discard.svg",
                 "chunked-adamw-discard.svg",
                 "vectorized-adamw-explicit.svg",
                 "streaming-safetensors-load.svg",
                 "context512-training-profile.svg",
                 "split-kv-backward-discard.svg",
                 "strided-batched-hipblaslt.svg",
                 "batched-attention-backward.svg",
                 "saved-attention-probabilities.svg",
                 "batched-attention-forward.svg",
                 "full-batched-attention-backward.svg",
                 "block-row-causal-softmax.svg",
                 "block-column-rmsnorm-weight-gradient.svg",
                 "inference-context-batch-matrix.svg",
                 "batched-long-prefill-inference.svg",
                 "full-prefill-kv-cache.svg",
                 "device-rowwise-argmax.svg",
                 "batched-kv-cache.svg",
                 "bf16-kv-cache.svg",
                 "fused-prefix-pair-discard.svg",
                 "mixed-layer-kv-policy.svg",
                 "targeted-prefix-pair-discard.svg",
                 "same-binary-kv-policy.svg",
                 "kv-policy-prompt-robustness.svg",
                 "qwen-kv-prompt-failure.svg",
                 "reference-serving-scheduler.svg",
                 "static-batch-generation.svg",
                 "admission-batch-scheduler.svg",
                 "expanded-inference-service-matrix.svg",
                 "serving-last-logit-prefill.svg",
                 "folded-gqa-discard.svg",
                 "register-softmax.svg",
                 "readable-fused-attention-discard.svg",
                 "inplace-causal-softmax.svg",
                 "stop-token-early-completion.svg",
                 "kv-cache-clear-row.svg",
                 "kv-cache-per-row-positions.svg",
                 "steady-inference-shape-memory.svg",
                 "deepseek-steady-profile-d2h-discard.svg",
                 "immediate-default-stream-pool.svg",
                 "bf16x2-key-load-discard.svg",
                 "raw-packed-key-load-discard.svg",
                 "device-token-history.svg",
                 "normalize-cached-probabilities-discard.svg",
                 "bf16-paired-value-load-discard.svg",
                 "divergent-cached-row-reference.svg",
                 "slot-row-prefill.svg",
                 "serving-inference-efficiency.svg",
                 "continuous-slot-scheduler.svg",
                 "active-row-compaction.svg",
                 "positions-aware-decode.svg",
                 "continuous-profile-scatter-discard.svg",
                 "packed-decode-metadata.svg",
                 "batched-slot-prefill.svg",
                 "official-continuous-serving.svg",
                 "continuous-slot-sweep.svg",
                 "continuous-divergence.svg",
                 "prefill-row-audit.svg",
                 "prefill-layer-drift.svg",
                 "block0-drift.svg",
                 "bf16-ffn-drift.svg",
                 "bf16-algorithm-inventory.svg",
                 "bf16-same-algorithm.svg",
                 "qwen-common-algorithm-discard.svg",
                 "qwen-algorithm-search.svg",
                 "request-latency.svg",
                 "length-bucket-tradeoff.svg",
                 "bucket-pareto-sweep.svg",
                 "traffic-skew-tail.svg",
                 "compatible-overflow.svg",
                 "slot-ratio-sweep.svg",
                 "mi300-precision-roofline.svg",
                 "large-precision-roofline.svg",
                 "mi300-int8-probe.svg",
                 "official-fp8-static-scale.svg",
                 "fp8-global-scale-grid.svg",
                 "fp8-scale-boundary.svg",
                 "fp8-scale-turn.svg",
                 "qwen-fp8-scale-closure.svg",
                 "fp8-tensor-amax-weight.svg",
                 "fp8-activation-range.svg",
                 "fp8-device-activation-amax.svg",
                 "fp8-activation-row-range.svg",
                 "fp8-ffn-outer-row.svg",
                 "fp8-device-weight-amax.svg",
                 "fp8-multiblock-amax.svg",
                 "fp8-dynamic-activation-profile.svg",
                 "fp8-shared-activation-quantization.svg",
                 "fp8-shared-activation-profile.svg",
                 "fp8-layer-drift.svg",
                 "fp8-block-detail.svg",
                 "fp8-residual-cancellation.svg",
                 "fp8-selective-block-counterfactual.svg",
                 "fp8-error-source-isolation.svg",
                 "fp8-native-vs-roundtrip.svg",
                 "fp8-output-channel-policy.svg",
                 "fp8-output-column-native-probe.svg",
                 "fp8-weight-reconstruction-audit.svg",
                 "fp8-output-head-only.svg",
                 "fp8-attention-only.svg",
                 "fp8-attention-output-only.svg",
                 "fp8-clipped-pilot-invalid.svg",
                 "fp8-fraction-pilot-workload-invalid.svg",
                 "fp8-clipped-coarse-grid.svg",
                 "fp8-clipped-fine-grid.svg",
                 "fp8-e5-activation-discard.svg",
                 "fp8-layer-leave-one-out.svg",
                 "fp8-qwen-layer9-formal-discard.svg",
                 "block-reduction-determinism.svg",
                 "adamw-correctness-before-timing.svg",
                 "cooperative-bias-gradient.svg",
                 "post-bias-training-profile.svg",
                 "bf16-training-solution-discard.svg",
                 "tied-embedding-sparse-add.svg",
                 "attention-rope-layout-fusion.svg",
                 "attention-interleaved-pv.svg",
                 "attention-context-layout-fusion.svg",
                 "post-layout-training-profile.svg",
                 "attention-layout-plan-cache-discard.svg",
                 "attention-gemm-scale-fusion-discard.svg",
                 "paired-gqa-repeat-discard.svg",
                 "gqa-zero-stride-value-broadcast.svg",
                 "selective-gqa-value-broadcast-discard.svg",
                 "forward-only-gqa-value-broadcast-discard.svg",
                 "unique-gradient-inplace-add-discard.svg",
                 "hip-graph-submission-crossover.svg",
                 "hip-graph-gemm-discard.svg",
                 "scoped-model-stream-discard.svg",
                 "deferred-hip-deallocation.svg",
                 "scoped-deferred-model-stream.svg",
                 "per-device-hipblaslt-handles.svg",
                 "stream-ordered-allocator.svg",
                 "activation-arena.svg",
                 "arena-ffn.svg",
                 "bf16-arena-ffn.svg",
                 "bf16-ffn-arena-model.svg",
                 "bf16-ffn-arena-selective.svg",
                 "bf16-qkv-arena-discard.svg",
                 "allocation-source-attribution.svg",
                 "attention-core-arena-discard.svg",
                 "fp32-attention-solutions.svg",
                 "fp32-attention-model-gate.svg",
                 "bf16-grouped-qkv.svg",
                 "bf16-grouped-qkv-expanded.svg",
                 "bf16-grouped-qkv-prewarm.svg",
                 "hipblaslt-preload.svg",
                 "bf16-exact-startup.svg",
                 "bf16-grouped-gate-up.svg",
                 "bf16-grouped-gate-up-model.svg",
                 "bf16-grouped-composition.svg",
                 "bf16-grouped-shape-matrix.svg",
                 "bf16-grouped-shape-models.svg",
                 "bf16-grouped-composed-profile.svg",
                 "hf-strided-copy-sources.svg",
                 "inference-bthd-attention.svg",
                 "inference-bthd-shape-models.svg",
                 "inference-bthd-profile.svg",
                 "inference-bthd-bf16-qk.svg",
                 "inference-bthd-bf16-qk-shapes.svg",
                 "causal-softmax-128-discard.svg",
                 "bf16-repeat-fusion-discard.svg",
                 "post-bf16-qk-saturation.svg",
                 "training-add-rms-norm-discard.svg"):
        path = ROOT / "assets" / name
        if not path.is_file():
            errors.append(f"missing SVG asset: {name}")
            continue
        try:
            ET.parse(path)
        except ET.ParseError as error:
            errors.append(f"invalid SVG {name}: {error}")
    completed = subprocess.run(
        ["python3", str(ROOT / "scripts" / "render_progress.py"), "--check"],
        cwd=REPOSITORY, capture_output=True, text=True
    )
    if completed.returncode != 0:
        errors.append(completed.stderr.strip() or completed.stdout.strip())


def main() -> int:
    errors: list[str] = []
    result_count = validate_results(errors)
    step_count = validate_steps(errors)
    bf16_policy_count = validate_bf16_policy(errors)
    bf16_ffn_count = validate_bf16_ffn(errors)
    bf16_model_count = validate_bf16_models(errors)
    bf16_prefill_count = validate_bf16_prefill(errors)
    profile_kernel_calls, profile_api_calls = validate_decode_profile(errors)
    bf16_attention_count = validate_bf16_attention(errors)
    post_profile_kernel_calls, post_profile_api_calls = validate_post_attention_profile(errors)
    bf16_plan_count = validate_bf16_plan_cache(errors)
    bf16_training_count = validate_bf16_training(errors)
    training_profile_kernel_calls, training_profile_api_calls = \
        validate_bf16_training_profile(errors)
    bf16_training_qkv_count = validate_bf16_training_qkv(errors)
    bf16_training_mirror_count = validate_bf16_training_mirrors(errors)
    bf16_training_island_count = validate_bf16_training_island(errors)
    bf16_training_shape_count = validate_bf16_training_shapes(errors)
    weight_gradient_candidate_count, weight_gradient_micro_count = \
        validate_weight_gradient_routing(errors)
    fused_causal_gqa_count = validate_fused_causal_gqa(errors)
    deepseek_pilot_count, deepseek_shape_count = validate_deepseek_shapes_and_load(errors)
    optimizer_profile_kernel_calls, optimizer_profile_api_calls = \
        validate_deepseek_optimizer_profile(errors)
    stable_gradient_matched, stable_gradient_mismatched = \
        validate_stable_gradient_discard(errors)
    chunked_adamw_pilot, chunked_adamw_formal = \
        validate_chunked_adamw_discard(errors)
    vectorized_adamw_operator, vectorized_adamw_pilot = \
        validate_vectorized_adamw(errors)
    streaming_load_smoke, streaming_load_formal = validate_streaming_load(errors)
    context512_pilot, context512_formal = validate_context512(errors)
    split_kv_pilot, split_kv_kernel_calls = validate_split_kv_discard(errors)
    batched_gemm_count = validate_batched_gemm(errors)
    batched_backward_records, batched_backward_calls = \
        validate_batched_attention_backward(errors)
    saved_attention_records, saved_attention_calls = validate_saved_attention(errors)
    batched_forward_records, batched_forward_calls = \
        validate_batched_attention_forward(errors)
    full_batched_backward_records, full_batched_backward_calls = \
        validate_full_batched_attention_backward(errors)
    block_softmax_records, block_softmax_calls = \
        validate_block_row_causal_softmax(errors)
    block_rms_records, block_rms_calls = \
        validate_block_column_rmsnorm_weight_gradient(errors)
    inference_core, inference_batch, inference_long, inference_no_warm = \
        validate_inference_shape_matrix(errors)
    prefill_records, prefill_calls = validate_batched_prefill_inference(errors)
    cache_records, cache_long_records, cache_calls = validate_full_prefill_cache(errors)
    row_argmax_records, row_argmax_host_records, row_argmax_calls = \
        validate_device_row_argmax(errors)
    batched_kv_records, batched_kv_pilot, batched_kv_calls = \
        validate_batched_kv_cache(errors)
    bf16_kv_baseline, bf16_kv_formal, bf16_kv_precision, bf16_kv_calls = \
        validate_bf16_kv_cache(errors)
    prefix_pair_formal, prefix_pair_precision, prefix_pair_calls = \
        validate_fused_prefix_pair_discard(errors)
    mixed_kv_formal, mixed_kv_precision, mixed_kv_search, mixed_kv_calls = \
        validate_mixed_layer_kv_policy(errors)
    targeted_prefix_rows, targeted_prefix_precision = \
        validate_targeted_prefix_pair_discard(errors)
    same_binary_policy_raw, same_binary_policy_rows = \
        validate_same_binary_kv_policy(errors)
    prompt_policy_layer1, prompt_policy_first4, prompt_policy_performance = \
        validate_kv_policy_prompt_robustness(errors)
    qwen_prompt_uniform, qwen_prompt_first2, qwen_prompt_search = \
        validate_qwen_kv_prompt_failure(errors)
    serving_reference_raw, serving_reference_rows = \
        validate_reference_serving_scheduler(errors)
    static_batch_raw, static_batch_rows = validate_static_batch_generation(errors)
    admission_batch_raw, admission_batch_rows = validate_admission_batch_scheduler(errors)
    cancellation_cpu, cancellation_hip, cancellation_sanitizer = \
        validate_request_cancellation(errors)
    expanded_prefill, expanded_fp32, expanded_bf16 = \
        validate_expanded_inference_service_matrix(errors)
    full_prefill, last_prefill, last_shape_survey, last_precision = \
        validate_serving_last_logit_prefill(errors)
    folded_gqa_raw, folded_gqa_precision, folded_gqa_profile = \
        validate_folded_gqa_discard(errors)
    register_softmax_paired, register_softmax_survey, register_softmax_recheck, \
        register_softmax_precision = validate_register_softmax(errors)
    readable_fused_raw, readable_fused_pairs = \
        validate_readable_fused_attention_discard(errors)
    inplace_softmax_paired, inplace_softmax_survey, inplace_softmax_precision = \
        validate_inplace_causal_softmax(errors)
    stop_token_cpu, stop_token_hip = validate_stop_token_early_completion(errors)
    clear_row_cpu, clear_row_hip, clear_row_bytes = validate_kv_cache_clear_row(errors)
    row_position_cpu, row_position_hip, row_position_transitions = \
        validate_kv_cache_per_row_positions(errors)
    inference_matrix_raw, inference_matrix_matched, inference_matrix_invalid, \
        inference_matrix_release = \
        validate_inference_shape_memory_matrix(errors)
    steady_profile_raw, steady_profile_b1, steady_profile_b8 = \
        validate_deepseek_steady_profile_d2h_discard(errors)
    immediate_pool_raw, immediate_pool_micro, immediate_pool_allocations = \
        validate_immediate_default_stream_pool(errors)
    bf16x2_rows, bf16x2_b1_values, bf16x2_b8_values = \
        validate_bf16x2_key_load_discard(errors)
    packed_rows, packed_b1_values, packed_b8_values = \
        validate_raw_packed_key_load_discard(errors)
    token_history_raw, token_history_micro, token_history_d2h = \
        validate_device_token_history(errors)
    normalize_rows, normalize_b1_values, normalize_b8_values = \
        validate_normalize_cached_probabilities_discard(errors)
    paired_value_rows, paired_value_b1, paired_value_b8 = \
        validate_bf16_paired_value_load_discard(errors)
    divergent_transitions, divergent_dtypes, divergent_steps = \
        validate_divergent_row_cache_reference(errors)
    row_prefill_transitions, row_prefill_dtypes, row_prefill_steps = \
        validate_slot_row_prefill(errors)
    serving_pilot_raw, serving_long_raw, serving_rechecks = \
        validate_serving_inference_efficiency(errors)
    continuous_transitions, continuous_divergent, continuous_uniform = \
        validate_continuous_slot_scheduler(errors)
    active_matrix, active_pairs, active_speedups = \
        validate_active_row_compaction(errors)
    positions_matrix, positions_pairs, positions_speedups = \
        validate_positions_aware_decode(errors)
    profile_shapes, scatter_pairs, scatter_ratios = \
        validate_continuous_profile_scatter_discard(errors)
    packed_profiles, packed_pairs, packed_ratios = \
        validate_packed_decode_metadata(errors)
    prefill_profiles, prefill_pairs, prefill_ratios = \
        validate_batched_slot_prefill(errors)
    official_continuous_raw, official_continuous_pytorch, \
        official_continuous_qwen, official_continuous_deepseek = \
        validate_official_continuous_serving(errors)
    slot_sweep_before, slot_sweep_after, slot_sweep_exact, \
        slot_sweep_mismatched = validate_fixed_request_slot_sweep(errors)
    divergence_raw, divergence_cases, divergence_default, \
        divergence_counterfactual = validate_deepseek_prefill_divergence(errors)
    row_audit_raw, row_audit_targets, row_audit_b2, row_audit_argmax = \
        validate_b2_prefill_row_audit(errors)
    layer_drift_raw, layer_drift_stages, layer_drift_exact, layer_drift_logits = \
        validate_prefill_layer_drift(errors)
    block0_raw, block0_stages, block0_exact, block0_duplicates = \
        validate_block0_drift(errors)
    ffn_drift_raw, ffn_drift_stages, ffn_drift_internal, ffn_drift_exact = \
        validate_bf16_ffn_drift(errors)
    algo_m32, algo_m64, algo_common = validate_bf16_algorithm_inventory(errors)
    same_precision, same_performance, same_exact = validate_bf16_same_algorithm(errors)
    qwen_common, qwen_precision, qwen_performance = validate_qwen_common_discard(errors)
    qwen_search_tested, qwen_search_exact = validate_qwen_algorithm_search(errors)
    latency_raw, latency_aggregates = validate_request_latency(errors)
    length_bucket_raw, length_bucket_comparisons, length_bucket_samples = \
        validate_length_bucket_tradeoff(errors)
    bucket_pareto_raw, bucket_pareto_rejected, bucket_pareto_samples = \
        validate_bucket_pareto(errors)
    traffic_skew_raw, traffic_skew_comparisons, traffic_skew_rejected = \
        validate_traffic_skew(errors)
    overflow_raw, overflow_comparisons, overflow_rejected = \
        validate_compatible_overflow(errors)
    ratio_raw, ratio_sweeps, ratio_preflight = validate_slot_ratio_sweep(errors)
    roofline_raw, roofline_sizes, roofline_dtypes = \
        validate_mi300_precision_roofline(errors)
    large_roofline_raw, large_roofline_sizes, large_roofline_dtypes = \
        validate_large_precision_roofline(errors)
    int8_raw, int8_samples, int8_paths = validate_int8_executed_probe(errors)
    official_fp8_raw, official_fp8_failures, official_fp8_rejected = \
        validate_official_fp8_static_scale(errors)
    fp8_grid_raw, fp8_grid_candidates, fp8_grid_passed = \
        validate_fp8_global_scale_grid(errors)
    fp8_boundary_raw, fp8_boundary_candidates, fp8_boundary_passed = \
        validate_fp8_scale_boundary(errors)
    fp8_turn_raw, fp8_turn_candidates, fp8_turn_passed = \
        validate_fp8_scale_turn(errors)
    fp8_closure_raw, fp8_closure_candidates, fp8_closure_passed = \
        validate_qwen_fp8_scale_closure(errors)
    fp8_amax_raw, fp8_amax_failures, fp8_amax_rejected = \
        validate_fp8_tensor_amax_weight(errors)
    activation_range_raw, activation_range_saturation, activation_range_rejected = \
        validate_fp8_activation_range(errors)
    device_amax_raw, device_amax_failures, device_amax_pilot = \
        validate_fp8_device_activation_amax(errors)
    row_range_raw, row_range_quarter, row_range_traces = \
        validate_fp8_activation_row_range(errors)
    ffn_row_raw, ffn_row_failures, ffn_row_pilot = \
        validate_fp8_ffn_outer_row(errors)
    device_weight_raw, device_weight_failures, device_weight_pilot = \
        validate_fp8_device_weight_amax(errors)
    multiblock_raw, multiblock_suites, multiblock_failures = \
        validate_fp8_multiblock_amax(errors)
    dynamic_profile_calls, dynamic_profile_qwen, dynamic_profile_deep = \
        validate_fp8_dynamic_profile(errors)
    shared_raw, shared_qwen_calls, shared_deep_calls = \
        validate_fp8_shared_activation(errors)
    shared_profile_calls, shared_profile_qwen, shared_profile_deep = \
        validate_fp8_shared_profile(errors)
    layer_drift_stages, layer_drift_qwen, layer_drift_deep = \
        validate_fp8_layer_drift(errors)
    block_detail_stages, block_detail_qwen, block_detail_deep = \
        validate_fp8_block_detail(errors)
    cancellation_rows, cancellation_qwen, cancellation_deep = \
        validate_fp8_residual_cancellation(errors)
    selective_workers, selective_failures, selective_models = \
        validate_fp8_selective_block_counterfactual(errors)
    source_workers, source_failures, source_logits = \
        validate_fp8_error_source_isolation(errors)
    native_workers, native_pairs, native_logits = \
        validate_fp8_native_roundtrip(errors)
    column_workers, column_fp8, column_passed = \
        validate_fp8_output_channel_policy(errors)
    native_column_workers, native_column_fp8, native_column_status = \
        validate_fp8_output_column_native_probe(errors)
    weight_audit_tensors, weight_audit_groups, weight_audit_invalid = \
        validate_fp8_weight_reconstruction_audit(errors)
    head_workers, head_fp8, head_precision = \
        validate_fp8_output_head_only(errors)
    attention_workers, attention_fp8, attention_precision = \
        validate_fp8_attention_only(errors)
    output_workers, output_fp8, output_precision = \
        validate_fp8_attention_output_only(errors)
    clipped_valid, clipped_required, clipped_excluded = \
        validate_fp8_clipped_pilot_invalid(errors)
    mismatch_workers, mismatch_comparisons, mismatch_cases = \
        validate_fp8_fraction_workload_invalid(errors)
    coarse_workers, coarse_comparisons, coarse_selected = \
        validate_fp8_clipped_coarse_grid(errors)
    fine_workers, fine_comparisons, fine_selected = \
        validate_fp8_clipped_fine_grid(errors)
    e5_workers, e5_fp8, e5_precision = \
        validate_fp8_e5_activation_discard(errors)
    layer_rows, layer_candidates, layer_deep_nonworse = \
        validate_fp8_layer_leave_one_out(errors)
    qwen9_workers, qwen9_fp8, qwen9_precision = \
        validate_fp8_qwen_layer9_formal(errors)
    reduction_before, reduction_after, reduction_full = \
        validate_block_reduction_determinism(errors)
    registry_fields, registry_passed = validate_matmul_exact_registry(errors)
    cache_contracts, cache_passed = validate_matmul_persistent_cache(errors)
    tune_candidates, tune_rejected, tune_cache_entries = \
        validate_matmul_correctness_before_timing(errors)
    adamw_rows, adamw_state_cases, adamw_kept_candidates = \
        validate_adamw_correctness_before_timing(errors)
    bias_rows, bias_models, bias_profile_speedup = \
        validate_cooperative_bias_gradient(errors)
    phase_categories, phase_gemm_share = \
        validate_post_bias_training_profile(errors)
    solution_rows, solution_candidates, solution_models = \
        validate_bf16_training_solution_discard(errors)
    tied_rows, tied_sparse_calls, tied_bytes_saved = \
        validate_tied_embedding_sparse_add(errors)
    interleaved_rows, interleaved_shapes, interleaved_qwen, interleaved_deep = \
        validate_attention_interleaved_pv(errors)
    context_rows, context_qwen_copies, context_deep_copies = \
        validate_attention_context_layout_fusion(errors)
    post_layout_categories, post_layout_gemm, post_layout_total = \
        validate_post_layout_training_profile(errors)
    plan_operator_rows, plan_model_rows, plan_route_rows = \
        validate_attention_layout_plan_cache_discard(errors)
    scale_model_rows, scale_qwen_allocations, scale_deep_allocations = \
        validate_attention_gemm_scale_fusion_discard(errors)
    paired_model_rows, paired_separate_calls, paired_fused_calls = \
        validate_paired_gqa_repeat_discard(errors)
    broadcast_rows, broadcast_shapes, broadcast_deep = \
        validate_gqa_zero_stride_value_broadcast(errors)
    selective_rows, selective_qwen_allocations, selective_deep_allocations = \
        validate_selective_gqa_value_broadcast_discard(errors)
    forward_rows, forward_qwen_allocations, forward_deep_allocations = \
        validate_forward_only_gqa_value_broadcast_discard(errors)
    inplace_rows, inplace_qwen_allocations, inplace_deep_allocations = \
        validate_unique_gradient_inplace_add_discard(errors)
    graph_rows, graph_cases, graph_minimum, graph_maximum, graph_api_saved = \
        validate_hip_graph_runtime(errors)
    graph_gemm_rows, graph_gemm_cases, graph_gemm_qwen, graph_gemm_deep = \
        validate_hip_graph_gemm_discard(errors)
    scoped_stream_runs, scoped_stream_max, scoped_stream_rms = \
        validate_scoped_model_stream_discard(errors)
    deferred_rows, deferred_cases, deferred_minimum, deferred_maximum, \
        deferred_max_bytes = validate_deferred_hip_deallocation(errors)
    scoped_deferred_rows, scoped_deferred_pairs, scoped_deferred_minimum, \
        scoped_deferred_maximum, scoped_deferred_bytes = \
        validate_scoped_deferred_model_stream(errors)
    device_handle_rows, device_handle_minimum, device_handle_maximum = \
        validate_per_device_hipblaslt_handles(errors)
    stream_ordered_rows, stream_ordered_async_minimum, \
        stream_ordered_async_maximum, stream_ordered_graph_minimum, \
        stream_ordered_graph_maximum, stream_ordered_pool = \
        validate_stream_ordered_allocator(errors)
    arena_rows, arena_eager_minimum, arena_eager_maximum, \
        arena_graph_minimum, arena_graph_maximum, arena_break_minimum, \
        arena_break_maximum = validate_activation_arena(errors)
    arena_ffn_rows, arena_ffn_eager_minimum, arena_ffn_eager_maximum, \
        arena_ffn_graph_minimum, arena_ffn_graph_maximum, \
        arena_ffn_break_minimum, arena_ffn_break_maximum = \
        validate_arena_ffn(errors)
    bf16_arena_rows, bf16_arena_eager_minimum, bf16_arena_eager_maximum, \
        bf16_arena_graph_minimum, bf16_arena_graph_maximum, \
        bf16_arena_node_minimum, bf16_arena_node_maximum = \
        validate_bf16_arena_ffn(errors)
    bf16_arena_model_rows, bf16_arena_model_minimum, \
        bf16_arena_model_maximum, bf16_arena_model_keep, \
        bf16_arena_model_regressions = validate_bf16_ffn_arena_model(errors)
    bf16_selective_rows, bf16_selective_minimum, bf16_selective_maximum, \
        bf16_selective_eligible, bf16_selective_bypassed = \
        validate_bf16_ffn_arena_selective(errors)
    bf16_qkv_rows, bf16_qkv_minimum, bf16_qkv_maximum, \
        bf16_qkv_eligible, bf16_qkv_bypassed = \
        validate_bf16_qkv_arena_discard(errors)
    allocation_source_rows, allocation_qwen_calls, allocation_qwen_core, \
        allocation_deep_calls, allocation_deep_core = \
        validate_allocation_source_attribution(errors)
    attention_core_rows, attention_core_minimum, attention_core_maximum, \
        attention_core_eligible, attention_core_bypassed = \
        validate_attention_core_arena_discard(errors)
    fp32_attention_rows, fp32_attention_keep, fp32_attention_minimum, \
        fp32_attention_maximum = validate_fp32_attention_solutions(errors)
    fp32_model_rows, fp32_model_exact, fp32_model_minimum, \
        fp32_model_maximum = validate_fp32_attention_model_gate(errors)
    grouped_qkv_operator, grouped_qkv_model, grouped_qkv_minimum, \
        grouped_qkv_maximum = validate_bf16_grouped_qkv(errors)
    grouped_expanded_operator, grouped_expanded_model, \
        grouped_expanded_minimum, grouped_expanded_maximum = \
        validate_bf16_grouped_qkv_expanded(errors)
    grouped_prewarm_rows, grouped_prewarm_minimum, grouped_prewarm_maximum = \
        validate_bf16_grouped_qkv_prewarm(errors)
    hipblaslt_preload_rows, hipblaslt_preload_minimum, \
        hipblaslt_preload_maximum = validate_hipblaslt_preload(errors)
    bf16_exact_tuning, bf16_exact_models, bf16_exact_cold, \
        bf16_exact_operator = validate_bf16_exact_startup(errors)
    grouped_gate_up_rows, grouped_gate_up_minimum, \
        grouped_gate_up_maximum = validate_bf16_grouped_gate_up(errors)
    grouped_gate_up_model_rows, grouped_gate_up_model_minimum, \
        grouped_gate_up_model_maximum, grouped_gate_up_calls_saved = \
        validate_bf16_grouped_gate_up_model(errors)
    grouped_composition_rows, grouped_composition_minimum, \
        grouped_composition_maximum, grouped_composition_incremental = \
        validate_bf16_grouped_composition(errors)
    grouped_shape_rows, grouped_shape_minimum, grouped_shape_maximum, \
        grouped_shape_reinit = validate_bf16_grouped_shape_matrix(errors)
    grouped_shape_model_rows, grouped_shape_model_minimum, \
        grouped_shape_model_maximum, grouped_shape_model_peak = \
        validate_bf16_grouped_shape_models(errors)
    composed_profile_processes, composed_profile_calls_saved, \
        composed_profile_minimum, composed_profile_maximum = \
        validate_bf16_grouped_composed_profile(errors)
    strided_source_rows, strided_source_calls, strided_source_bytes, \
        strided_layout_bytes = validate_hf_strided_copy_sources(errors)
    bthd_performance_rows, bthd_diagnostic_rows, bthd_minimum, \
        bthd_maximum, bthd_bytes_removed = \
        validate_inference_bthd_attention(errors)
    bthd_shape_performance, bthd_shape_diagnostics, \
        bthd_shape_minimum, bthd_shape_maximum, bthd_shape_residual = \
        validate_inference_bthd_shape_models(errors)
    bthd_profile_processes, bthd_profile_minimum, \
        bthd_profile_maximum, bthd_profile_strided_removed = \
        validate_inference_bthd_profile(errors)
    bthd_bf16_initial, bthd_bf16_formal, bthd_bf16_minimum, \
        bthd_bf16_maximum, bthd_bf16_casts = \
        validate_inference_bthd_bf16_qk(errors)
    bthd_bf16_shape_pilot, bthd_bf16_shape_formal, \
        bthd_bf16_shape_minimum, bthd_bf16_shape_maximum, \
        bthd_bf16_shape_cases = \
        validate_inference_bthd_bf16_qk_shapes(errors)
    softmax_thread_rows, softmax_thread_passed, softmax_thread_minimum, \
        softmax_thread_maximum = validate_causal_softmax_128_discard(errors)
    bf16_repeat_rows, bf16_repeat_passed, bf16_repeat_minimum, \
        bf16_repeat_maximum = validate_bf16_repeat_fusion_discard(errors)
    saturation_rows, saturation_minimum, saturation_maximum = \
        validate_post_bf16_qk_saturation(errors)
    training_norm_rows, training_norm_minimum, training_norm_maximum, \
        training_norm_dispatches = validate_training_add_rms_norm_discard(errors)
    link_count = validate_links(errors)
    validate_assets(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"optimization log valid: results={result_count} steps={step_count} "
          f"bf16_policy={bf16_policy_count} bf16_ffn={bf16_ffn_count} "
          f"bf16_models={bf16_model_count} bf16_prefill={bf16_prefill_count} "
          f"bf16_attention={bf16_attention_count} "
          f"bf16_plan={bf16_plan_count} bf16_training={bf16_training_count} "
          f"bf16_training_qkv={bf16_training_qkv_count} "
          f"bf16_training_mirrors={bf16_training_mirror_count} "
          f"bf16_training_island={bf16_training_island_count} "
          f"bf16_training_shapes={bf16_training_shape_count} "
          f"weight_gradient={weight_gradient_candidate_count}/{weight_gradient_micro_count} "
          f"fused_causal_gqa={fused_causal_gqa_count} "
          f"deepseek_shapes={deepseek_pilot_count}/{deepseek_shape_count} "
          f"optimizer_profile={optimizer_profile_kernel_calls}/{optimizer_profile_api_calls} "
          f"stable_gradient={stable_gradient_matched}/{stable_gradient_mismatched} "
          f"chunked_adamw={chunked_adamw_pilot}/{chunked_adamw_formal} "
          f"vectorized_adamw={vectorized_adamw_operator}/{vectorized_adamw_pilot} "
          f"streaming_load={streaming_load_smoke}/{streaming_load_formal} "
          f"context512={context512_pilot}/{context512_formal} "
          f"split_kv={split_kv_pilot}/{split_kv_kernel_calls} "
          f"batched_gemm={batched_gemm_count} "
          f"batched_backward={batched_backward_records}/{batched_backward_calls} "
          f"saved_attention={saved_attention_records}/{saved_attention_calls} "
          f"batched_forward={batched_forward_records}/{batched_forward_calls} "
          f"full_batched_backward={full_batched_backward_records}/"
          f"{full_batched_backward_calls} "
          f"block_softmax={block_softmax_records}/{block_softmax_calls} "
          f"block_rms={block_rms_records}/{block_rms_calls} "
          f"inference={inference_core}/{inference_batch}/{inference_long}/"
          f"{inference_no_warm} "
          f"prefill={prefill_records}/{prefill_calls} "
          f"cache_prefill={cache_records}/{cache_long_records}/{cache_calls} "
          f"row_argmax={row_argmax_records}/{row_argmax_host_records}/"
          f"{row_argmax_calls} "
          f"batched_kv={batched_kv_records}/{batched_kv_pilot}/{batched_kv_calls} "
          f"bf16_kv={bf16_kv_baseline}/{bf16_kv_formal}/"
          f"{bf16_kv_precision}/{bf16_kv_calls} "
          f"prefix_pair={prefix_pair_formal}/{prefix_pair_precision}/"
          f"{prefix_pair_calls} "
          f"mixed_kv={mixed_kv_formal}/{mixed_kv_precision}/"
          f"{mixed_kv_search}/{mixed_kv_calls} "
          f"targeted_prefix={targeted_prefix_rows}/{targeted_prefix_precision} "
          f"same_binary_policy={same_binary_policy_raw}/{same_binary_policy_rows} "
          f"prompt_policy={prompt_policy_layer1}/{prompt_policy_first4}/"
          f"{prompt_policy_performance} "
          f"qwen_prompt={qwen_prompt_uniform}/{qwen_prompt_first2}/"
          f"{qwen_prompt_search} "
          f"serving_reference={serving_reference_raw}/{serving_reference_rows} "
          f"static_batch={static_batch_raw}/{static_batch_rows} "
          f"admission_batch={admission_batch_raw}/{admission_batch_rows} "
          f"cancellation={cancellation_cpu}/{cancellation_hip}/"
          f"{cancellation_sanitizer} "
          f"expanded_inference={expanded_prefill}/{expanded_fp32}/"
          f"{expanded_bf16} "
          f"last_prefill={full_prefill}/{last_prefill}/"
          f"{last_shape_survey}/{last_precision} "
          f"folded_gqa={folded_gqa_raw}/{folded_gqa_precision}/"
          f"{folded_gqa_profile} "
          f"register_softmax={register_softmax_paired}/"
          f"{register_softmax_survey}/{register_softmax_recheck}/"
          f"{register_softmax_precision} "
          f"readable_fused={readable_fused_raw}/{readable_fused_pairs} "
          f"inplace_softmax={inplace_softmax_paired}/"
          f"{inplace_softmax_survey}/{inplace_softmax_precision} "
          f"stop_token={stop_token_cpu}/{stop_token_hip} "
          f"clear_row={clear_row_cpu}/{clear_row_hip}/{clear_row_bytes} "
          f"row_positions={row_position_cpu}/{row_position_hip}/"
          f"{row_position_transitions} "
          f"steady_inference={inference_matrix_raw}/{inference_matrix_matched}/"
          f"{inference_matrix_invalid}/{inference_matrix_release} "
          f"steady_profile={steady_profile_raw}/{steady_profile_b1}/"
          f"{steady_profile_b8} "
          f"immediate_pool={immediate_pool_raw}/{immediate_pool_micro}/"
          f"{immediate_pool_allocations} "
          f"bf16x2_discard={bf16x2_rows}/{bf16x2_b1_values}/"
          f"{bf16x2_b8_values} "
          f"packed_discard={packed_rows}/{packed_b1_values}/"
          f"{packed_b8_values} "
          f"token_history={token_history_raw}/{token_history_micro}/"
          f"{token_history_d2h} "
          f"normalize_discard={normalize_rows}/{normalize_b1_values}/"
          f"{normalize_b8_values} "
          f"paired_value_discard={paired_value_rows}/{paired_value_b1}/"
          f"{paired_value_b8} "
          f"divergent_rows={divergent_transitions}/{divergent_dtypes}/"
          f"{divergent_steps} "
          f"row_prefill={row_prefill_transitions}/{row_prefill_dtypes}/"
          f"{row_prefill_steps} "
          f"serving_efficiency={serving_pilot_raw}/{serving_long_raw}/"
          f"{serving_rechecks} "
          f"continuous_slots={continuous_transitions}/{continuous_divergent}/"
          f"{continuous_uniform} "
          f"active_compaction={active_matrix}/{active_pairs}/"
          f"{active_speedups} "
          f"positions_aware={positions_matrix}/{positions_pairs}/"
          f"{positions_speedups} "
          f"continuous_profile={profile_shapes}/{scatter_pairs}/"
          f"{scatter_ratios} "
          f"packed_metadata={packed_profiles}/{packed_pairs}/"
          f"{packed_ratios} "
          f"batched_prefill={prefill_profiles}/{prefill_pairs}/"
          f"{prefill_ratios} "
          f"official_continuous={official_continuous_raw}/"
          f"{official_continuous_pytorch}/{official_continuous_qwen}/"
          f"{official_continuous_deepseek} "
          f"slot_sweep={slot_sweep_before}/{slot_sweep_after}/"
          f"{slot_sweep_exact}/{slot_sweep_mismatched} "
          f"prefill_divergence={divergence_raw}/{divergence_cases}/"
          f"{divergence_default}/{divergence_counterfactual} "
          f"prefill_row_audit={row_audit_raw}/{row_audit_targets}/"
          f"{row_audit_b2}/{row_audit_argmax} "
          f"prefill_layer_drift={layer_drift_raw}/{layer_drift_stages}/"
          f"{layer_drift_exact}/{layer_drift_logits} "
          f"block0_drift={block0_raw}/{block0_stages}/"
          f"{block0_exact}/{block0_duplicates} "
          f"bf16_ffn_drift={ffn_drift_raw}/{ffn_drift_stages}/"
          f"{ffn_drift_internal}/{ffn_drift_exact} "
          f"bf16_algorithms={algo_m32}/{algo_m64}/{algo_common} "
          f"same_algorithm={same_precision}/{same_performance}/{same_exact} "
          f"qwen_common_discard={qwen_common}/{qwen_precision}/{qwen_performance} "
          f"qwen_search={qwen_search_tested}/{qwen_search_exact} "
          f"request_latency={latency_raw}/{latency_aggregates} "
          f"length_buckets={length_bucket_raw}/{length_bucket_comparisons}/"
          f"{length_bucket_samples} "
          f"bucket_pareto={bucket_pareto_raw}/{bucket_pareto_rejected}/"
          f"{bucket_pareto_samples} "
          f"traffic_skew={traffic_skew_raw}/{traffic_skew_comparisons}/"
          f"{traffic_skew_rejected} "
          f"compatible_overflow={overflow_raw}/{overflow_comparisons}/"
          f"{overflow_rejected} "
          f"slot_ratios={ratio_raw}/{ratio_sweeps}/{ratio_preflight} "
          f"precision_roofline={roofline_raw}/{roofline_sizes}/"
          f"{roofline_dtypes} "
          f"large_roofline={large_roofline_raw}/{large_roofline_sizes}/"
          f"{large_roofline_dtypes} "
          f"int8_probe={int8_raw}/{int8_samples}/{int8_paths} "
          f"official_fp8={official_fp8_raw}/{official_fp8_failures}/"
          f"{official_fp8_rejected} "
          f"fp8_grid={fp8_grid_raw}/{fp8_grid_candidates}/{fp8_grid_passed} "
          f"fp8_boundary={fp8_boundary_raw}/{fp8_boundary_candidates}/"
          f"{fp8_boundary_passed} "
          f"fp8_turn={fp8_turn_raw}/{fp8_turn_candidates}/{fp8_turn_passed} "
          f"fp8_closure={fp8_closure_raw}/{fp8_closure_candidates}/"
          f"{fp8_closure_passed} "
          f"fp8_amax={fp8_amax_raw}/{fp8_amax_failures}/{fp8_amax_rejected} "
          f"activation_range={activation_range_raw}/{activation_range_saturation}/"
          f"{activation_range_rejected} "
          f"device_amax={device_amax_raw}/{device_amax_failures}/"
          f"{device_amax_pilot} "
          f"row_range={row_range_raw}/{row_range_quarter}/{row_range_traces} "
          f"ffn_row={ffn_row_raw}/{ffn_row_failures}/{ffn_row_pilot} "
          f"device_weight={device_weight_raw}/{device_weight_failures}/"
          f"{device_weight_pilot} "
          f"multiblock={multiblock_raw}/{multiblock_suites}/{multiblock_failures} "
          f"dynamic_profile={dynamic_profile_calls}/{dynamic_profile_qwen}/"
          f"{dynamic_profile_deep} "
          f"shared_activation={shared_raw}/{shared_qwen_calls}/{shared_deep_calls} "
          f"shared_profile={shared_profile_calls}/{shared_profile_qwen}/"
          f"{shared_profile_deep} "
          f"layer_drift={layer_drift_stages}/{layer_drift_qwen}/{layer_drift_deep} "
          f"block_detail={block_detail_stages}/{block_detail_qwen}/{block_detail_deep} "
          f"cancellation={cancellation_rows}/{cancellation_qwen}/{cancellation_deep} "
          f"selective_fp32={selective_workers}/{selective_failures}/"
          f"{selective_models} "
          f"fp8_sources={source_workers}/{source_failures}/{source_logits} "
          f"native_roundtrip={native_workers}/{native_pairs}/{native_logits} "
          f"output_channel={column_workers}/{column_fp8}/{column_passed} "
          f"column_native={native_column_workers}/{native_column_fp8}/"
          f"{native_column_status} "
          f"weight_audit={weight_audit_tensors}/{weight_audit_groups}/"
          f"{weight_audit_invalid} "
          f"head_only={head_workers}/{head_fp8}/{head_precision} "
          f"attention_only={attention_workers}/{attention_fp8}/"
          f"{attention_precision} "
          f"attention_output={output_workers}/{output_fp8}/{output_precision} "
          f"clipped_invalid={clipped_valid}/{clipped_required}/"
          f"{clipped_excluded} "
          f"fraction_mismatch={mismatch_workers}/{mismatch_comparisons}/"
          f"{mismatch_cases} "
          f"clipped_coarse={coarse_workers}/{coarse_comparisons}/"
          f"{coarse_selected} "
          f"clipped_fine={fine_workers}/{fine_comparisons}/{fine_selected} "
          f"e5_activation={e5_workers}/{e5_fp8}/{e5_precision} "
          f"layer_leave_one_out={layer_rows}/{layer_candidates}/"
          f"{layer_deep_nonworse} "
          f"qwen_layer9={qwen9_workers}/{qwen9_fp8}/{qwen9_precision} "
          f"reduction_determinism={reduction_before}/{reduction_after}/"
          f"{reduction_full} "
          f"exact_registry={registry_fields}/{registry_passed} "
          f"persistent_registry={cache_contracts}/{cache_passed} "
          f"matmul_autotune={tune_candidates}/{tune_rejected}/"
          f"{tune_cache_entries} "
          f"adamw_autotune={adamw_rows}/{adamw_state_cases}/"
          f"{adamw_kept_candidates} "
          f"bias_gradient={bias_rows}/{bias_models}/"
          f"{bias_profile_speedup:.2f} "
          f"phase_delta={phase_categories}/{phase_gemm_share:.3f} "
          f"bf16_solutions={solution_rows}/{solution_candidates}/"
          f"{solution_models} "
          f"tied_sparse={tied_rows}/{tied_sparse_calls}/{tied_bytes_saved} "
          f"interleaved_pv={interleaved_rows}/{interleaved_shapes}/"
          f"{interleaved_qwen:.3f}/{interleaved_deep:.3f} "
          f"context_layout={context_rows}/{context_qwen_copies}/"
          f"{context_deep_copies} "
          f"post_layout={post_layout_categories}/{post_layout_gemm:.3f}/"
          f"{post_layout_total} "
          f"attention_plan={plan_operator_rows}/{plan_model_rows}/"
          f"{plan_route_rows} "
          f"attention_scale={scale_model_rows}/{scale_qwen_allocations}/"
          f"{scale_deep_allocations} "
          f"paired_gqa={paired_model_rows}/{paired_separate_calls}/"
          f"{paired_fused_calls} "
          f"gqa_broadcast={broadcast_rows}/{broadcast_shapes}/"
          f"{broadcast_deep:.3f} "
          f"selective_broadcast={selective_rows}/{selective_qwen_allocations}/"
          f"{selective_deep_allocations} "
          f"forward_broadcast={forward_rows}/{forward_qwen_allocations}/"
          f"{forward_deep_allocations} "
          f"gradient_inplace={inplace_rows}/{inplace_qwen_allocations}/"
          f"{inplace_deep_allocations} "
          f"hip_graph={graph_rows}/{graph_cases}/{graph_minimum:.3f}/"
          f"{graph_maximum:.3f}/{graph_api_saved} "
          f"hip_graph_gemm={graph_gemm_rows}/{graph_gemm_cases}/"
          f"{graph_gemm_qwen:.3f}/{graph_gemm_deep:.3f} "
          f"scoped_stream={scoped_stream_runs}/{scoped_stream_max:.3f}/"
          f"{scoped_stream_rms:.3f} "
          f"deferred_release={deferred_rows}/{deferred_cases}/"
          f"{deferred_minimum:.3f}/{deferred_maximum:.3f}/"
          f"{deferred_max_bytes} "
          f"scoped_deferred={scoped_deferred_rows}/{scoped_deferred_pairs}/"
          f"{scoped_deferred_minimum:.3f}/{scoped_deferred_maximum:.3f}/"
          f"{scoped_deferred_bytes} "
          f"device_handles={device_handle_rows}/{device_handle_minimum:.3f}/"
          f"{device_handle_maximum:.3f} "
          f"stream_ordered={stream_ordered_rows}/"
          f"{stream_ordered_async_minimum:.3f}/{stream_ordered_async_maximum:.3f}/"
          f"{stream_ordered_graph_minimum:.3f}/{stream_ordered_graph_maximum:.3f}/"
          f"{stream_ordered_pool} "
          f"activation_arena={arena_rows}/{arena_eager_minimum:.3f}/"
          f"{arena_eager_maximum:.3f}/{arena_graph_minimum:.3f}/"
          f"{arena_graph_maximum:.3f}/{arena_break_minimum}/{arena_break_maximum} "
          f"arena_ffn={arena_ffn_rows}/{arena_ffn_eager_minimum:.3f}/"
          f"{arena_ffn_eager_maximum:.3f}/{arena_ffn_graph_minimum:.3f}/"
          f"{arena_ffn_graph_maximum:.3f}/{arena_ffn_break_minimum}/"
          f"{arena_ffn_break_maximum} "
          f"bf16_arena_ffn={bf16_arena_rows}/"
          f"{bf16_arena_eager_minimum:.3f}/{bf16_arena_eager_maximum:.3f}/"
          f"{bf16_arena_graph_minimum:.3f}/{bf16_arena_graph_maximum:.3f}/"
          f"{bf16_arena_node_minimum}/{bf16_arena_node_maximum} "
          f"bf16_arena_model={bf16_arena_model_rows}/"
          f"{bf16_arena_model_minimum:.3f}/{bf16_arena_model_maximum:.3f}/"
          f"{bf16_arena_model_keep}/{bf16_arena_model_regressions} "
          f"bf16_arena_selective={bf16_selective_rows}/"
          f"{bf16_selective_minimum:.3f}/{bf16_selective_maximum:.3f}/"
          f"{bf16_selective_eligible}/{bf16_selective_bypassed} "
          f"bf16_qkv_arena={bf16_qkv_rows}/{bf16_qkv_minimum:.3f}/"
          f"{bf16_qkv_maximum:.3f}/{bf16_qkv_eligible}/{bf16_qkv_bypassed} "
          f"allocation_sources={allocation_source_rows}/{allocation_qwen_calls}/"
          f"{allocation_qwen_core}/{allocation_deep_calls}/{allocation_deep_core} "
          f"attention_core_arena={attention_core_rows}/"
          f"{attention_core_minimum:.3f}/{attention_core_maximum:.3f}/"
          f"{attention_core_eligible}/{attention_core_bypassed} "
          f"fp32_attention_solutions={fp32_attention_rows}/{fp32_attention_keep}/"
          f"{fp32_attention_minimum:.3f}/{fp32_attention_maximum:.3f} "
          f"fp32_attention_model={fp32_model_rows}/{fp32_model_exact}/"
          f"{fp32_model_minimum:.3f}/{fp32_model_maximum:.3f} "
          f"bf16_grouped_qkv={grouped_qkv_operator}/{grouped_qkv_model}/"
          f"{grouped_qkv_minimum:.3f}/{grouped_qkv_maximum:.3f} "
          f"bf16_grouped_expanded={grouped_expanded_operator}/"
          f"{grouped_expanded_model}/{grouped_expanded_minimum:.3f}/"
          f"{grouped_expanded_maximum:.3f} "
          f"bf16_grouped_prewarm={grouped_prewarm_rows}/"
          f"{grouped_prewarm_minimum:.1f}/{grouped_prewarm_maximum:.1f} "
          f"hipblaslt_preload={hipblaslt_preload_rows}/"
          f"{hipblaslt_preload_minimum:.3f}/{hipblaslt_preload_maximum:.3f} "
          f"bf16_exact_startup={bf16_exact_tuning}/{bf16_exact_models}/"
          f"{bf16_exact_cold:.3f}/{bf16_exact_operator:.3f} "
          f"bf16_grouped_gate_up={grouped_gate_up_rows}/"
          f"{grouped_gate_up_minimum:.3f}/{grouped_gate_up_maximum:.3f} "
          f"bf16_grouped_gate_up_model={grouped_gate_up_model_rows}/"
          f"{grouped_gate_up_model_minimum:.3f}/"
          f"{grouped_gate_up_model_maximum:.3f}/"
          f"{grouped_gate_up_calls_saved} "
          f"bf16_grouped_composition={grouped_composition_rows}/"
          f"{grouped_composition_minimum:.3f}/"
          f"{grouped_composition_maximum:.3f}/"
          f"{grouped_composition_incremental:.3f} "
          f"bf16_grouped_shapes={grouped_shape_rows}/"
          f"{grouped_shape_minimum:.3f}/{grouped_shape_maximum:.3f}/"
          f"{grouped_shape_reinit} "
          f"bf16_grouped_shape_models={grouped_shape_model_rows}/"
          f"{grouped_shape_model_minimum:.3f}/"
          f"{grouped_shape_model_maximum:.3f}/"
          f"{grouped_shape_model_peak:.3f} "
          f"bf16_grouped_composed_profile={composed_profile_processes}/"
          f"{composed_profile_calls_saved}/"
          f"{composed_profile_minimum:.3f}/"
          f"{composed_profile_maximum:.3f} "
          f"strided_sources={strided_source_rows}/{strided_source_calls}/"
          f"{strided_source_bytes}/{strided_layout_bytes} "
          f"inference_bthd={bthd_performance_rows}/"
          f"{bthd_diagnostic_rows}/{bthd_minimum:.3f}/"
          f"{bthd_maximum:.3f}/{bthd_bytes_removed} "
          f"inference_bthd_shapes={bthd_shape_performance}/"
          f"{bthd_shape_diagnostics}/{bthd_shape_minimum:.3f}/"
          f"{bthd_shape_maximum:.3f}/{bthd_shape_residual} "
          f"inference_bthd_profile={bthd_profile_processes}/"
          f"{bthd_profile_minimum:.3f}/{bthd_profile_maximum:.3f}/"
          f"{bthd_profile_strided_removed} "
          f"inference_bthd_bf16_qk={bthd_bf16_initial}/"
          f"{bthd_bf16_formal}/{bthd_bf16_minimum:.3f}/"
          f"{bthd_bf16_maximum:.3f}/{bthd_bf16_casts} "
          f"inference_bthd_bf16_qk_shapes={bthd_bf16_shape_pilot}/"
          f"{bthd_bf16_shape_formal}/{bthd_bf16_shape_minimum:.3f}/"
          f"{bthd_bf16_shape_maximum:.3f}/{bthd_bf16_shape_cases} "
          f"causal_softmax_threads={softmax_thread_rows}/"
          f"{softmax_thread_passed}/{softmax_thread_minimum:.3f}/"
          f"{softmax_thread_maximum:.3f} "
          f"bf16_repeat_fusion={bf16_repeat_rows}/{bf16_repeat_passed}/"
          f"{bf16_repeat_minimum:.3f}/{bf16_repeat_maximum:.3f} "
          f"inference_saturation={saturation_rows}/"
          f"{saturation_minimum:.3f}/{saturation_maximum:.3f} "
          f"training_add_rms_norm={training_norm_rows}/"
          f"{training_norm_minimum:.3f}/{training_norm_maximum:.3f}/"
          f"{training_norm_dispatches} "
          f"profile_calls={profile_kernel_calls}/{profile_api_calls},"
          f"{post_profile_kernel_calls}/{post_profile_api_calls},"
          f"{training_profile_kernel_calls}/{training_profile_api_calls} links={link_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
