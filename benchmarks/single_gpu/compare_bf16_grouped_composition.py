#!/usr/bin/env python3
"""Gate independent and composed BF16 grouped QKV and gate/up policies."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


POLICIES = ("baseline", "qkv", "gate_up", "both")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--qwen-qkv-index", type=int, default=64713)
    parser.add_argument("--deepseek-qkv-index", type=int, default=64755)
    parser.add_argument("--qwen-gate-up-index", type=int, default=65168)
    parser.add_argument("--deepseek-gate-up-index", type=int, default=65200)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=0.25)
    parser.add_argument("--rms-tolerance", type=float, default=0.05)
    parser.add_argument("--minimum-combined-speedup", type=float, default=1.03)
    parser.add_argument("--minimum-incremental-speedup", type=float, default=1.005)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.005)
    parser.add_argument("--maximum-combined-setup-ms", type=float, default=250.0)
    result = parser.parse_args()
    indices = (
        result.qwen_qkv_index, result.deepseek_qkv_index,
        result.qwen_gate_up_index, result.deepseek_gate_up_index)
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.sequence <= 0 or any(index < 0 for index in indices) or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.minimum_combined_speedup <= 1 or
            result.minimum_incremental_speedup <= 1 or
            result.maximum_peak_ratio < 1 or
            result.maximum_combined_setup_ms < 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("grouped composition options are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("grouped composition requires pinned official models")
    for model in result:
        if not Path(model["config"]).is_file() or \
                not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise RuntimeError("grouped composition token seed cannot be empty")
    return [seed[index % len(seed)] for index in range(length)]


def indices(args: argparse.Namespace, model: dict) -> tuple[int, int]:
    if model["name"] == "qwen2.5-0.5b":
        return args.qwen_qkv_index, args.qwen_gate_up_index
    return args.deepseek_qkv_index, args.deepseek_gate_up_index


def command(args: argparse.Namespace, model: dict, policy: str,
            logits: Path) -> list[str]:
    tokens = repeated(model["inference"]["token_ids"], args.sequence)
    qkv_index, gate_up_index = indices(args, model)
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
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
    if policy in ("qkv", "both"):
        result.extend([
            "--bf16-qkv-arena", "true",
            "--bf16-qkv-arena-minimum-rows", "512",
            "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
        ])
    if policy in ("gate_up", "both"):
        result.extend([
            "--bf16-grouped-gate-up-algorithm-index",
            str(gate_up_index),
        ])
    return result


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("hf_infer emitted no JSON")


def floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def error(reference: array.array, actual: array.array) -> tuple[float, float, bool]:
    if len(reference) != len(actual) or not reference:
        raise RuntimeError("grouped composition complete-logit size changed")
    maximum = 0.0
    squared = 0.0
    finite = True
    for expected, observed in zip(reference, actual, strict=True):
        difference = abs(expected - observed)
        maximum = max(maximum, difference)
        squared += difference * difference
        finite = finite and math.isfinite(observed)
    return maximum, math.sqrt(squared / len(reference)), finite


def top_index(values: array.array) -> int:
    return max(range(len(values)), key=values.__getitem__)


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    outputs = {}
    with tempfile.TemporaryDirectory(
            prefix="microllm-grouped-composition-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            blocks = 24 if model["name"] == "qwen2.5-0.5b" else 28
            forwards = args.warmup + args.steps
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
                    if completed.returncode != 0:
                        raise RuntimeError(completed.stdout + completed.stderr)
                    record = last_json(completed.stdout)
                    if record.get("status") != "pass":
                        raise RuntimeError(f"invalid composition record: {stem}")
                    expected_dispatches = blocks * forwards
                    qkv_enabled = policy in ("qkv", "both")
                    gate_enabled = policy in ("gate_up", "both")
                    if int(record["bf16_grouped_qkv_dispatches"]) != (
                            expected_dispatches if qkv_enabled else 0) or \
                            int(record["bf16_grouped_gate_up_dispatches"]) != (
                                expected_dispatches if gate_enabled else 0):
                        raise RuntimeError("grouped composition dispatch mismatch")
                    record.update({
                        "record_type": "bf16_grouped_composition_measurement",
                        "model": model["name"],
                        "revision": model["revision"],
                        "policy": policy,
                        "process_run": process_run,
                        "process_order": order,
                    })
                    records.append(record)
                    outputs[(model["name"], policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        name = model["name"]
        qkv_index, gate_up_index = indices(args, model)
        reference = outputs[(name, "baseline", 1)]
        selected = [row for row in records if row["model"] == name]
        grouped = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in POLICIES
        }
        maximum = 0.0
        rms = 0.0
        finite = True
        top_tokens_equal = True
        for policy in POLICIES:
            for process_run in range(1, args.runs + 1):
                actual = outputs[(name, policy, process_run)]
                current = error(reference, actual)
                maximum = max(maximum, current[0])
                rms = max(rms, current[1])
                finite = finite and current[2]
                top_tokens_equal = (
                    top_tokens_equal and
                    top_index(reference) == top_index(actual))
        tps = {
            policy: median(grouped[policy], "prefill_tokens_per_second")
            for policy in POLICIES
        }
        peak = {
            policy: int(median(grouped[policy], "engine_peak_bytes"))
            for policy in POLICIES
        }
        comparisons.append({
            "model": name,
            "revision": model["revision"],
            "qkv_solution_index": qkv_index,
            "gate_up_solution_index": gate_up_index,
            "tokens_per_second": tps,
            "speedup_vs_baseline": {
                policy: tps[policy] / tps["baseline"]
                for policy in POLICIES
            },
            "both_vs_qkv_speedup": tps["both"] / tps["qkv"],
            "both_vs_gate_up_speedup": tps["both"] / tps["gate_up"],
            "peak_bytes": peak,
            "both_peak_ratio": peak["both"] / peak["baseline"],
            "qkv_kernel_setup_ms": median(
                grouped["both"], "bf16_grouped_qkv_kernel_setup_ms"),
            "gate_up_kernel_setup_ms": median(
                grouped["both"],
                "bf16_grouped_gate_up_kernel_setup_ms"),
            "combined_kernel_setup_ms": median(
                grouped["both"], "bf16_grouped_qkv_kernel_setup_ms") +
                median(grouped["both"],
                       "bf16_grouped_gate_up_kernel_setup_ms"),
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
    performance = all(
        row["speedup_vs_baseline"]["both"] >=
            args.minimum_combined_speedup and
        row["both_vs_qkv_speedup"] >= args.minimum_incremental_speedup
        for row in comparisons)
    memory = all(
        row["both_peak_ratio"] <= args.maximum_peak_ratio
        for row in comparisons)
    setup = all(
        row["combined_kernel_setup_ms"] <=
            args.maximum_combined_setup_ms
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and memory else "fail",
        "record_type": "bf16_grouped_composition_summary",
        "raw_processes": len(records),
        "correctness_gate": correctness,
        "performance_gate": performance,
        "memory_gate": memory,
        "setup_gate": setup,
        "comparisons": comparisons,
        "decision": (
            "keep explicit composed grouped policy"
            if correctness and performance and memory and setup else
            "reject grouped policy composition"),
    }
    with (args.output_directory / "raw.jsonl").open(
            "w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
