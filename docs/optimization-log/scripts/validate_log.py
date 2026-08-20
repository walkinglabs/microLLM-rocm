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
                 "bf16-training-qkv-discard.svg", "bf16-training-mirrors.svg"):
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
          f"profile_calls={profile_kernel_calls}/{profile_api_calls},"
          f"{post_profile_kernel_calls}/{post_profile_api_calls},"
          f"{training_profile_kernel_calls}/{training_profile_api_calls} links={link_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
