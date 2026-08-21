#!/usr/bin/env python3
"""Validate optimization records, local links and generated SVG assets."""

from __future__ import annotations

import csv
import html
import json
import math
import re
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
                 "kv-policy-prompt-robustness.svg"):
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
          f"profile_calls={profile_kernel_calls}/{profile_api_calls},"
          f"{post_profile_kernel_calls}/{post_profile_api_calls},"
          f"{training_profile_kernel_calls}/{training_profile_api_calls} links={link_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
