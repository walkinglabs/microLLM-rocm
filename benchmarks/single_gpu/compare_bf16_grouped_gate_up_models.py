#!/usr/bin/env python3
"""Gate pointer-stable BF16 grouped gate/up on complete official models."""

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
    parser.add_argument("--qwen-index", type=int, default=65168)
    parser.add_argument("--deepseek-index", type=int, default=65200)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=0.25)
    parser.add_argument("--rms-tolerance", type=float, default=0.05)
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.001)
    parser.add_argument("--maximum-kernel-setup-ms", type=float, default=100.0)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.sequence <= 0 or result.qwen_index < 0 or
            result.deepseek_index < 0 or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.minimum_speedup <= 1 or result.maximum_peak_ratio < 1 or
            result.maximum_kernel_setup_ms < 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("grouped gate/up model options are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("grouped gate/up model gate requires pinned models")
    for model in result:
        if not Path(model["config"]).is_file() or \
                not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise RuntimeError("grouped gate/up token seed cannot be empty")
    return [seed[index % len(seed)] for index in range(length)]


def index_for(args: argparse.Namespace, model: dict) -> int:
    return args.qwen_index if model["name"] == "qwen2.5-0.5b" \
        else args.deepseek_index


def command(args: argparse.Namespace, model: dict, policy: str,
            logits: Path) -> list[str]:
    tokens = repeated(model["inference"]["token_ids"], args.sequence)
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
    if policy == "grouped":
        result.extend([
            "--bf16-grouped-gate-up-algorithm-index",
            str(index_for(args, model)),
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
        raise RuntimeError("grouped gate/up complete-logit size changed")
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
            prefix="microllm-grouped-gate-up-") as temp:
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
                        raise RuntimeError(f"invalid grouped record: {stem}")
                    prefix = "bf16_grouped_gate_up_"
                    if policy == "grouped":
                        expected_dispatches = blocks * forwards
                        expected_hits = blocks * (forwards - 1)
                        if (int(record[prefix + "registered_entries"]) != 1 or
                                int(record[prefix + "kernel_entries"]) != 1 or
                                int(record[prefix + "kernel_misses"]) != 1 or
                                int(record[prefix + "kernel_hits"]) != blocks - 1 or
                                int(record[prefix + "plan_entries"]) != blocks or
                                int(record[prefix + "plan_misses"]) != blocks or
                                int(record[prefix + "plan_hits"]) != expected_hits or
                                int(record[prefix + "dispatches"]) !=
                                    expected_dispatches):
                            raise RuntimeError(
                                "grouped gate/up did not use stable per-block plans")
                    elif any(int(record[prefix + field]) != 0 for field in (
                            "registered_entries", "kernel_entries",
                            "plan_entries", "dispatches")):
                        raise RuntimeError("baseline unexpectedly used grouped gate/up")
                    record.update({
                        "record_type": "bf16_grouped_gate_up_model_measurement",
                        "model": model["name"],
                        "revision": model["revision"],
                        "policy": policy,
                        "process_run": process_run,
                        "process_order": order,
                        "solution_index":
                            index_for(args, model) if policy == "grouped" else -1,
                    })
                    records.append(record)
                    outputs[(model["name"], policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        name = model["name"]
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
        baseline_tps = median(
            grouped["baseline"], "prefill_tokens_per_second")
        grouped_tps = median(
            grouped["grouped"], "prefill_tokens_per_second")
        baseline_peak = int(median(
            grouped["baseline"], "engine_peak_bytes"))
        grouped_peak = int(median(
            grouped["grouped"], "engine_peak_bytes"))
        comparisons.append({
            "model": name,
            "revision": model["revision"],
            "solution_index": index_for(args, model),
            "baseline_tokens_per_second": baseline_tps,
            "grouped_tokens_per_second": grouped_tps,
            "grouped_speedup": grouped_tps / baseline_tps,
            "baseline_peak_bytes": baseline_peak,
            "grouped_peak_bytes": grouped_peak,
            "peak_ratio": grouped_peak / baseline_peak,
            "kernel_setup_ms": median(
                grouped["grouped"],
                "bf16_grouped_gate_up_kernel_setup_ms"),
            "argument_setup_ms": median(
                grouped["grouped"],
                "bf16_grouped_gate_up_argument_setup_ms"),
            "plan_entries": int(median(
                grouped["grouped"],
                "bf16_grouped_gate_up_plan_entries")),
            "plan_hits": int(median(
                grouped["grouped"],
                "bf16_grouped_gate_up_plan_hits")),
            "dispatches": int(median(
                grouped["grouped"],
                "bf16_grouped_gate_up_dispatches")),
            "maximum_absolute_logit_difference": maximum,
            "maximum_rms_logit_difference": rms,
            "finite_complete_logits": finite,
            "top_tokens_equal": top_tokens_equal,
        })

    correctness = all(
        row["finite_complete_logits"] and
        row["top_tokens_equal"] and
        row["maximum_absolute_logit_difference"] <=
            args.maximum_absolute_tolerance and
        row["maximum_rms_logit_difference"] <= args.rms_tolerance
        for row in comparisons)
    performance = all(
        row["grouped_speedup"] >= args.minimum_speedup
        for row in comparisons)
    memory = all(
        row["peak_ratio"] <= args.maximum_peak_ratio
        for row in comparisons)
    setup = all(
        row["kernel_setup_ms"] <= args.maximum_kernel_setup_ms
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and memory else "fail",
        "record_type": "bf16_grouped_gate_up_model_summary",
        "raw_processes": len(records),
        "correctness_gate": correctness,
        "performance_gate": performance,
        "memory_gate": memory,
        "setup_gate": setup,
        "comparisons": comparisons,
        "decision": (
            "keep explicit pointer-stable grouped gate/up policy"
            if correctness and performance and memory and setup else
            "reject grouped gate/up model policy"),
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
