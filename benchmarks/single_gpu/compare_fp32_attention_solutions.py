#!/usr/bin/env python3
"""Gate exact FP32 Attention solution indices on complete official-model logits."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


POLICIES = ("baseline", "qk", "pv", "both")
GROUPED_T1024 = {
    "qwen2.5-0.5b": (64755, 65200),
    "deepseek-r1-distill-qwen-1.5b": (64755, 65212),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--bthd-policy", action="store_true")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--qwen-qk-index", type=int, default=311017)
    parser.add_argument("--qwen-pv-index", type=int, default=294519)
    parser.add_argument("--deepseek-qk-index", type=int, default=305423)
    parser.add_argument("--deepseek-pv-index", type=int, default=292941)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--rms-tolerance", type=float, default=1.0e-5)
    result = parser.parse_args()
    result.policies = tuple(result.policies.split(","))
    indices = (
        result.qwen_qk_index, result.qwen_pv_index,
        result.deepseek_qk_index, result.deepseek_pv_index,
    )
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.sequence < 256 or result.sequence > 4096 or
            any(index < 0 for index in indices) or
            not result.policies or result.policies[0] != "baseline" or
            any(policy not in POLICIES for policy in result.policies) or
            len(set(result.policies)) != len(result.policies) or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0):
        parser.error("run, shape, index, or tolerance options are invalid")
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    if result.bthd_policy and result.policies != ("baseline", "qk"):
        parser.error("current BTHD path only exposes generic exact QK routing")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or len(document.get("models", [])) != 2:
        raise RuntimeError("formal solution gate requires the two-model manifest")
    result = document["models"]
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if {model.get("name") for model in result} != expected:
        raise RuntimeError("formal solution gate requires pinned Qwen and DeepSeek")
    for model in result:
        if not Path(model["config"]).is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
    return result


def repeated_tokens(seed: list[int], length: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(length)]


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("hf_infer emitted no JSON object")


def floats(path: Path) -> array.array:
    result = array.array("f")
    with path.open("rb") as stream:
        result.fromfile(stream, path.stat().st_size // result.itemsize)
    return result


def errors(reference: array.array, actual: array.array) -> tuple[float, float, bool]:
    if len(reference) != len(actual):
        raise RuntimeError("complete-logit element count changed")
    maximum = 0.0
    squared = 0.0
    finite = True
    for expected, observed in zip(reference, actual, strict=True):
        finite = finite and math.isfinite(observed)
        difference = abs(expected - observed)
        maximum = max(maximum, difference)
        squared += difference * difference
    rms = math.sqrt(squared / len(reference)) if reference else 0.0
    return maximum, rms, finite


def indices(args: argparse.Namespace, model: dict) -> tuple[int, int]:
    if model["name"] == "qwen2.5-0.5b":
        return args.qwen_qk_index, args.qwen_pv_index
    return args.deepseek_qk_index, args.deepseek_pv_index


def command(args: argparse.Namespace, model: dict, policy: str,
            logits: Path) -> list[str]:
    tokens = repeated_tokens(model["inference"]["token_ids"], args.sequence)
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", "1",
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true",
        "--bf16-ffn-arena-minimum-rows", str(args.sequence),
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(args.steps),
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]
    if args.bthd_policy:
        if args.sequence != 1024:
            raise RuntimeError("current BTHD solution gate is pinned to T1024")
        qkv_index, gate_up_index = GROUPED_T1024[model["name"]]
        result.extend([
            "--bf16-qkv-arena", "true",
            "--bf16-qkv-arena-minimum-rows", str(args.sequence),
            "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
            "--bf16-grouped-gate-up-algorithm-index", str(gate_up_index),
            "--inference-bthd-attention", "true",
            "--inference-bthd-bf16-qk", "true",
            "--inference-bthd-online-attention", "false",
        ])
    qk, pv = indices(args, model)
    if policy in ("qk", "both"):
        result.extend([
            "--fp32-attention-qk-solution-index", str(qk),
        ])
    if policy in ("pv", "both"):
        result.extend([
            "--fp32-attention-pv-solution-index", str(pv),
        ])
    return result


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    records: list[dict] = []
    outputs: dict[tuple[str, str, int], array.array] = {}
    with tempfile.TemporaryDirectory(prefix="microllm-fp32-attention-model-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            for process_run in range(1, args.runs + 1):
                order = list(args.policies)
                if process_run % 2 == 0:
                    order.reverse()
                for policy in order:
                    stem = f"{model['name']}-t{args.sequence}-p{process_run}-{policy}"
                    logits_path = temporary / f"{stem}.bin"
                    completed = subprocess.run(
                        command(args, model, policy, logits_path),
                        text=True, capture_output=True, check=False)
                    (logs / f"{stem}.stdout.txt").write_text(
                        completed.stdout, encoding="utf-8")
                    (logs / f"{stem}.stderr.txt").write_text(
                        completed.stderr, encoding="utf-8")
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"hf_infer failed for {stem}: {completed.stderr}")
                    record = last_json(completed.stdout)
                    if record.get("status") != "pass":
                        raise RuntimeError(f"invalid record for {stem}")
                    if args.bthd_policy and (
                            record.get("inference_bthd_attention") is not True or
                            record.get("inference_bthd_bf16_qk") is not True or
                            record.get("inference_bthd_online_attention") is not False or
                            int(record.get("bf16_grouped_qkv_dispatches", 0)) <= 0):
                        raise RuntimeError(
                            f"current BTHD policy did not dispatch for {stem}")
                    registered = int(record["fp32_solution_registered_entries"])
                    dispatches = int(record["fp32_solution_dispatches"])
                    expected_registered = 0 if policy == "baseline" else \
                        2 if policy == "both" else 1
                    if policy != "baseline" and (
                            registered != expected_registered or dispatches <= 0):
                        raise RuntimeError(f"candidate did not dispatch exact solution: {stem}")
                    if policy == "baseline" and (registered != 0 or dispatches != 0):
                        raise RuntimeError(f"baseline unexpectedly selected a solution: {stem}")
                    record.update({
                        "record_type": "fp32_attention_solution_model_measurement",
                        "model": model["name"], "revision": model["revision"],
                        "policy": policy, "sequence": args.sequence,
                        "process_run": process_run, "process_order": order,
                    })
                    records.append(record)
                    outputs[(model["name"], policy, process_run)] = floats(logits_path)

    comparisons: list[dict] = []
    for model in selected_models:
        reference = outputs[(model["name"], "baseline", 1)]
        selected = [row for row in records if row["model"] == model["name"]]
        grouped = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in args.policies
        }
        qk, pv = indices(args, model)
        for policy in args.policies[1:]:
            maximum = 0.0
            rms = 0.0
            finite = True
            for process_run in range(1, args.runs + 1):
                current_maximum, current_rms, current_finite = errors(
                    reference, outputs[(model["name"], policy, process_run)])
                maximum = max(maximum, current_maximum)
                rms = max(rms, current_rms)
                finite = finite and current_finite
            baseline_speed = median(
                grouped["baseline"], "prefill_tokens_per_second")
            candidate_speed = median(
                grouped[policy], "prefill_tokens_per_second")
            row = {
                "model": model["name"], "revision": model["revision"],
                "policy": policy, "sequence": args.sequence, "batch": 1,
                "qk_solution_index": qk if policy in ("qk", "both") else -1,
                "pv_solution_index": pv if policy in ("pv", "both") else -1,
                "baseline_tokens_per_second": baseline_speed,
                "candidate_tokens_per_second": candidate_speed,
                "candidate_speedup": candidate_speed / baseline_speed,
                "baseline_engine_peak_bytes": int(median(
                    grouped["baseline"], "engine_peak_bytes")),
                "candidate_engine_peak_bytes": int(median(
                    grouped[policy], "engine_peak_bytes")),
                "baseline_engine_allocation_calls": int(median(
                    grouped["baseline"], "engine_allocation_calls")),
                "candidate_engine_allocation_calls": int(median(
                    grouped[policy], "engine_allocation_calls")),
                "candidate_registered_entries": int(median(
                    grouped[policy], "fp32_solution_registered_entries")),
                "candidate_cached_algorithms": int(median(
                    grouped[policy], "fp32_solution_cached_algorithms")),
                "candidate_dispatches": int(median(
                    grouped[policy], "fp32_solution_dispatches")),
                "maximum_absolute_logit_difference": maximum,
                "maximum_rms_logit_difference": rms,
                "finite_complete_logits": finite,
            }
            row["correctness_passed"] = (
                finite and maximum <= args.maximum_absolute_tolerance and
                rms <= args.rms_tolerance)
            row["performance_passed"] = row["candidate_speedup"] >= 1.01
            row["memory_passed"] = (
                row["candidate_engine_peak_bytes"] <=
                row["baseline_engine_peak_bytes"])
            comparisons.append(row)

    policy_keep = {}
    for policy in args.policies[1:]:
        rows = [row for row in comparisons if row["policy"] == policy]
        policy_keep[policy] = len(rows) == len(selected_models) and all(
            row["correctness_passed"] and row["performance_passed"] and
            row["memory_passed"] for row in rows)
    keep_policies = [policy for policy, keep in policy_keep.items() if keep]
    correctness = all(row["correctness_passed"] for row in comparisons)
    performance = all(row["performance_passed"] for row in comparisons)
    memory = all(row["memory_passed"] for row in comparisons)
    keep = bool(keep_policies)
    summary = {
        "schema_version": 1,
        "status": "pass" if all(
            row["finite_complete_logits"] for row in comparisons) else "fail",
        "record_type": "fp32_attention_solution_model_summary",
        "bthd_policy": args.bthd_policy,
        "policies": list(args.policies),
        "raw_processes": len(records),
        "maximum_absolute_tolerance": args.maximum_absolute_tolerance,
        "rms_tolerance": args.rms_tolerance,
        "correctness_gate": correctness,
        "performance_gate": performance,
        "memory_gate": memory,
        "policy_keep": policy_keep,
        "keep_policies": keep_policies,
        "keep_default": keep,
        "comparisons": comparisons,
        "decision": (
            "keep exact FP32 Attention policy: " + ",".join(keep_policies)
            if keep else
            "retain explicit FP32 Attention candidates; do not enable by default"
        ),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
