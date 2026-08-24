#!/usr/bin/env python3
"""Run alternating legacy/deferred official-model process pairs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import subprocess
from pathlib import Path


MODELS = {
    "qwen": "qwen2.5-0.5b",
    "deepseek": "deepseek-r1-distill-qwen-1.5b",
}


def csv_values(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--qwen-config", type=Path, required=True)
    parser.add_argument("--qwen-weights", type=Path, required=True)
    parser.add_argument("--deepseek-config", type=Path, required=True)
    parser.add_argument("--deepseek-weights", type=Path, required=True)
    parser.add_argument("--models", type=csv_values, default=["qwen", "deepseek"])
    parser.add_argument("--modes", type=csv_values, default=["inference", "training"])
    parser.add_argument("--contexts", type=csv_values, default=["32", "512"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--maximum-blocks", type=int, default=8192)
    result = parser.parse_args()
    if result.repetitions <= 0 or result.warmup < 0 or result.steps <= 0:
        parser.error("repetitions/steps must be positive and warmup nonnegative")
    if result.batch <= 0 or result.maximum_blocks <= 0:
        parser.error("batch and maximum-blocks must be positive")
    if any(model not in MODELS for model in result.models):
        parser.error("models must contain qwen and/or deepseek")
    if any(mode not in ("inference", "training") for mode in result.modes):
        parser.error("modes must contain inference and/or training")
    if any(int(context) <= 0 for context in result.contexts):
        parser.error("contexts must be positive")
    return result


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("benchmark emitted no JSON object")


def read_floats(path: Path) -> tuple[float, ...]:
    payload = path.read_bytes()
    if len(payload) % 4:
        raise RuntimeError(f"unaligned float output: {path}")
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def inference_error(left: Path, right: Path) -> tuple[float, float]:
    first = read_floats(left)
    second = read_floats(right)
    if len(first) != len(second) or not first:
        raise RuntimeError("paired complete-logit sizes differ or are empty")
    differences = [abs(a - b) for a, b in zip(first, second)]
    rms = math.sqrt(sum(value * value for value in differences) / len(differences))
    return max(differences), rms


def model_paths(args: argparse.Namespace, key: str) -> tuple[Path, Path]:
    if key == "qwen":
        return args.qwen_config, args.qwen_weights
    return args.deepseek_config, args.deepseek_weights


def run_one(args: argparse.Namespace, model_key: str, mode: str,
            context: int, repetition: int, policy: str,
            pair_order: list[str]) -> tuple[dict, Path | None]:
    model = MODELS[model_key]
    config, weights = model_paths(args, model_key)
    stem = f"{model_key}-{mode}-t{context}-r{repetition}-{policy}"
    logits_path = args.output_directory / "logits" / f"{stem}.bin"
    command = [
        str(args.benchmark),
        "--config", str(config),
        "--weights", str(weights),
        "--model", model,
        "--mode", mode,
        "--policy", policy,
        "--precision", args.precision,
        "--context", str(context),
        "--batch", str(args.batch),
        "--warmup", str(args.warmup),
        "--steps", str(args.steps),
        "--maximum-blocks", str(args.maximum_blocks),
    ]
    if mode == "inference":
        command += ["--logits-output", str(logits_path)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    logs = args.output_directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{stem}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{stem}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({completed.returncode}) for {stem}: {completed.stderr}")
    row = last_json(completed.stdout)
    row["process_run"] = repetition
    row["pair_order"] = pair_order
    row["model_key"] = model_key
    if row.get("status") != "pass" or row.get("policy") != policy:
        raise RuntimeError(f"invalid benchmark record for {stem}")
    return row, logits_path if mode == "inference" else None


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    pair_checks: list[dict] = []
    for model_key in args.models:
        for mode in args.modes:
            for context_text in args.contexts:
                context = int(context_text)
                for repetition in range(1, args.repetitions + 1):
                    order = (["legacy", "deferred"] if repetition % 2
                             else ["deferred", "legacy"])
                    paired: dict[str, dict] = {}
                    logits: dict[str, Path] = {}
                    for policy in order:
                        row, path = run_one(
                            args, model_key, mode, context, repetition, policy, order)
                        raw_rows.append(row)
                        paired[policy] = row
                        if path is not None:
                            logits[policy] = path
                    check = {
                        "model": MODELS[model_key],
                        "model_key": model_key,
                        "mode": mode,
                        "context": context,
                        "process_run": repetition,
                    }
                    if mode == "inference":
                        maximum, rms = inference_error(
                            logits["legacy"], logits["deferred"])
                        check.update(maximum_absolute_error=maximum, rms_error=rms)
                    else:
                        check.update(
                            loss_absolute_difference=abs(
                                float(paired["legacy"]["loss"]) -
                                float(paired["deferred"]["loss"])),
                            parameter_absolute_difference=abs(
                                float(paired["legacy"]["observed_parameter_after"]) -
                                float(paired["deferred"]["observed_parameter_after"])),
                        )
                    pair_checks.append(check)

    comparisons: list[dict] = []
    for model_key in args.models:
        for mode in args.modes:
            for context_text in args.contexts:
                context = int(context_text)
                selected = [row for row in raw_rows
                            if row["model_key"] == model_key and
                            row["mode"] == mode and row["context"] == context]
                policies = {
                    policy: [row for row in selected if row["policy"] == policy]
                    for policy in ("legacy", "deferred")
                }
                legacy_rate = median(policies["legacy"], "tokens_per_second")
                deferred_rate = median(policies["deferred"], "tokens_per_second")
                comparisons.append({
                    "model": MODELS[model_key],
                    "model_key": model_key,
                    "mode": mode,
                    "context": context,
                    "legacy_tokens_per_second": legacy_rate,
                    "deferred_tokens_per_second": deferred_rate,
                    "deferred_speedup": deferred_rate / legacy_rate,
                    "legacy_engine_peak_bytes": int(median(
                        policies["legacy"], "engine_peak_bytes")),
                    "deferred_engine_peak_bytes": int(median(
                        policies["deferred"], "engine_peak_bytes")),
                    "maximum_deferred_bytes": int(max(
                        int(row["maximum_deferred_bytes"])
                        for row in policies["deferred"])),
                    "deferred_backend_allocation_calls": int(median(
                        policies["deferred"], "engine_backend_allocation_calls")),
                    "legacy_backend_allocation_calls": int(median(
                        policies["legacy"], "engine_backend_allocation_calls")),
                    "deferred_overflow_flushes": int(max(
                        int(row["deferred_overflow_flushes"])
                        for row in policies["deferred"])),
                })

    correctness = all(
        (row.get("maximum_absolute_error", 0.0) <= 1.0e-5 and
         row.get("rms_error", 0.0) <= 1.0e-6 and
         row.get("loss_absolute_difference", 0.0) <= 1.0e-5 and
         row.get("parameter_absolute_difference", 0.0) <= 1.0e-5)
        for row in pair_checks)
    performance_keep = all(row["deferred_speedup"] >= 1.05 for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness else "fail",
        "record_type": "scoped_deferred_model_matrix_summary",
        "raw_processes": len(raw_rows),
        "paired_checks": pair_checks,
        "comparisons": comparisons,
        "correctness_gate": correctness,
        "performance_gate": performance_keep,
        "decision": ("enable candidate" if correctness and performance_keep
                     else "keep safe infrastructure; default off"),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in raw_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.output_directory / "pairs.jsonl").open("w", encoding="utf-8") as output:
        for row in pair_checks:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if correctness else 2


if __name__ == "__main__":
    raise SystemExit(main())
