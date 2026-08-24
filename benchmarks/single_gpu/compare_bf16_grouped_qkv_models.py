#!/usr/bin/env python3
"""Gate cached BF16 grouped-QKV plans on complete official-model prefill."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


POLICIES = ("baseline", "grouped")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--qwen-index", type=int, default=64699)
    parser.add_argument("--deepseek-index", type=int, default=64701)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=0.25)
    parser.add_argument("--rms-tolerance", type=float, default=0.05)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.005)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.sequence <= 0 or result.sequence > 4096 or
            result.qwen_index < 0 or result.deepseek_index < 0 or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.maximum_peak_ratio < 1.0):
        parser.error("grouped-QKV model options are invalid")
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("formal grouped-QKV gate requires pinned Qwen and DeepSeek")
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
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


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
    return maximum, math.sqrt(squared / len(reference)), finite


def solution_index(args: argparse.Namespace, model: dict) -> int:
    return args.qwen_index if model["name"] == "qwen2.5-0.5b" \
        else args.deepseek_index


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
        "--bf16-ffn-arena-minimum-rows", "512",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(args.steps),
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]
    if policy == "grouped":
        result.extend([
            "--bf16-qkv-arena", "true",
            "--bf16-qkv-arena-minimum-rows", "512",
            "--bf16-grouped-qkv-algorithm-index",
            str(solution_index(args, model)),
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
    with tempfile.TemporaryDirectory(prefix="microllm-bf16-grouped-qkv-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            for process_run in range(1, args.runs + 1):
                order = list(POLICIES)
                if process_run % 2 == 0:
                    order.reverse()
                for policy in order:
                    stem = f"{model['name']}-p{process_run}-{policy}"
                    logits = temporary / f"{stem}.bin"
                    completed = subprocess.run(
                        command(args, model, policy, logits),
                        text=True, capture_output=True, check=False)
                    (logs / f"{stem}.stdout.txt").write_text(
                        completed.stdout, encoding="utf-8")
                    (logs / f"{stem}.stderr.txt").write_text(
                        completed.stderr, encoding="utf-8")
                    if completed.returncode != 0:
                        raise RuntimeError(f"hf_infer failed for {stem}: {completed.stderr}")
                    record = last_json(completed.stdout)
                    if record.get("status") != "pass":
                        raise RuntimeError(f"invalid result for {stem}")
                    entries = int(record["bf16_grouped_qkv_registered_entries"])
                    dispatches = int(record["bf16_grouped_qkv_dispatches"])
                    if policy == "grouped" and (entries != 1 or dispatches <= 0):
                        raise RuntimeError(f"grouped plan did not dispatch for {stem}")
                    if policy == "baseline" and (entries != 0 or dispatches != 0):
                        raise RuntimeError(f"baseline unexpectedly dispatched grouped QKV")
                    record.update({
                        "record_type": "bf16_grouped_qkv_model_measurement",
                        "model": model["name"], "revision": model["revision"],
                        "policy": policy, "sequence": args.sequence,
                        "process_run": process_run, "process_order": order,
                    })
                    records.append(record)
                    outputs[(model["name"], policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        reference = outputs[(model["name"], "baseline", 1)]
        selected = [row for row in records if row["model"] == model["name"]]
        grouped_rows = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in POLICIES
        }
        maximum = 0.0
        rms = 0.0
        finite = True
        for policy in POLICIES:
            for process_run in range(1, args.runs + 1):
                current = errors(
                    reference, outputs[(model["name"], policy, process_run)])
                maximum = max(maximum, current[0])
                rms = max(rms, current[1])
                finite = finite and current[2]
        baseline_speed = median(
            grouped_rows["baseline"], "prefill_tokens_per_second")
        grouped_speed = median(
            grouped_rows["grouped"], "prefill_tokens_per_second")
        baseline_peak = int(median(grouped_rows["baseline"], "engine_peak_bytes"))
        grouped_peak = int(median(grouped_rows["grouped"], "engine_peak_bytes"))
        baseline_top = grouped_rows["baseline"][0]["top_logits"][0]["token"]
        top_tokens_equal = all(
            row["top_logits"][0]["token"] == baseline_top for row in selected)
        comparisons.append({
            "model": model["name"], "revision": model["revision"],
            "sequence": args.sequence, "batch": 1,
            "solution_index": solution_index(args, model),
            "baseline_tokens_per_second": baseline_speed,
            "grouped_tokens_per_second": grouped_speed,
            "grouped_speedup": grouped_speed / baseline_speed,
            "baseline_engine_peak_bytes": baseline_peak,
            "grouped_engine_peak_bytes": grouped_peak,
            "peak_ratio": grouped_peak / baseline_peak,
            "baseline_engine_allocation_calls": int(median(
                grouped_rows["baseline"], "engine_allocation_calls")),
            "grouped_engine_allocation_calls": int(median(
                grouped_rows["grouped"], "engine_allocation_calls")),
            "grouped_plan_entries": int(median(
                grouped_rows["grouped"], "bf16_grouped_qkv_plan_entries")),
            "grouped_plan_hits": int(median(
                grouped_rows["grouped"], "bf16_grouped_qkv_plan_hits")),
            "grouped_plan_misses": int(median(
                grouped_rows["grouped"], "bf16_grouped_qkv_plan_misses")),
            "grouped_dispatches": int(median(
                grouped_rows["grouped"], "bf16_grouped_qkv_dispatches")),
            "maximum_absolute_logit_difference": maximum,
            "maximum_rms_logit_difference": rms,
            "finite_complete_logits": finite,
            "top_tokens_equal": top_tokens_equal,
        })

    correctness = all(
        row["finite_complete_logits"] and row["top_tokens_equal"] and
        row["maximum_absolute_logit_difference"] <=
            args.maximum_absolute_tolerance and
        row["maximum_rms_logit_difference"] <= args.rms_tolerance
        for row in comparisons)
    performance = all(row["grouped_speedup"] >= 1.01 for row in comparisons)
    memory = all(row["peak_ratio"] <= args.maximum_peak_ratio
                 for row in comparisons)
    keep = correctness and performance and memory
    summary = {
        "schema_version": 1, "status": "pass" if correctness else "fail",
        "record_type": "bf16_grouped_qkv_model_summary",
        "raw_processes": len(records),
        "maximum_absolute_tolerance": args.maximum_absolute_tolerance,
        "rms_tolerance": args.rms_tolerance,
        "maximum_peak_ratio": args.maximum_peak_ratio,
        "correctness_gate": correctness, "performance_gate": performance,
        "memory_gate": memory, "keep_default": keep,
        "comparisons": comparisons,
        "decision": ("keep selective BF16 grouped QKV plans" if keep else
                     "retain grouped-QKV probe; reject model policy"),
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
