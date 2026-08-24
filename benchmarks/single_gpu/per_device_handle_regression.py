#!/usr/bin/env python3
"""Check per-device hipBLASLt handles against the previous single-GPU raw rows."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


MODELS = {
    "qwen": ("qwen2.5-0.5b", "qwen_config", "qwen_weights"),
    "deepseek": ("deepseek-r1-distill-qwen-1.5b",
                 "deepseek_config", "deepseek_weights"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--qwen-config", type=Path, required=True)
    parser.add_argument("--qwen-weights", type=Path, required=True)
    parser.add_argument("--deepseek-config", type=Path, required=True)
    parser.add_argument("--deepseek-weights", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    result = parser.parse_args()
    if result.context <= 0 or result.repetitions <= 0 or result.warmup < 0 or result.steps <= 0:
        parser.error("context/repetitions/steps must be positive and warmup nonnegative")
    return result


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            return row
    raise RuntimeError("benchmark emitted no JSON record")


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    baseline_all = json_lines(args.baseline)
    current_rows: list[dict] = []
    comparisons: list[dict] = []
    for key, (model, config_field, weights_field) in MODELS.items():
        config = getattr(args, config_field)
        weights = getattr(args, weights_field)
        for mode in ("inference", "training"):
            baseline = [row for row in baseline_all
                        if row.get("model_key") == key and row.get("mode") == mode and
                        row.get("policy") == "legacy" and
                        int(row.get("context", -1)) == args.context]
            if len(baseline) < args.repetitions:
                raise RuntimeError(f"baseline lacks {key}/{mode} repetitions")
            rows: list[dict] = []
            for repetition in range(1, args.repetitions + 1):
                command = [
                    str(args.benchmark), "--config", str(config),
                    "--weights", str(weights), "--model", model,
                    "--mode", mode, "--policy", "legacy",
                    "--precision", "bf16", "--context", str(args.context),
                    "--batch", "1", "--warmup", str(args.warmup),
                    "--steps", str(args.steps), "--maximum-blocks", "8192",
                ]
                completed = subprocess.run(
                    command, text=True, capture_output=True, check=False)
                stem = f"{key}-{mode}-r{repetition}"
                (logs / f"{stem}.stdout.txt").write_text(
                    completed.stdout, encoding="utf-8")
                (logs / f"{stem}.stderr.txt").write_text(
                    completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(f"benchmark failed for {stem}: {completed.stderr}")
                row = last_json(completed.stdout)
                row["model_key"] = key
                row["process_run"] = repetition
                rows.append(row)
                current_rows.append(row)
            reference = baseline[0]
            if mode == "inference":
                exact = all(
                    row["top_index"] == reference["top_index"] and
                    row["top_value"] == reference["top_value"] and
                    row["logits_sum"] == reference["logits_sum"] and
                    row["logits_square_sum"] == reference["logits_square_sum"]
                    for row in rows)
            else:
                exact = all(
                    row["loss"] == reference["loss"] and
                    row["observed_parameter_after"] ==
                    reference["observed_parameter_after"]
                    for row in rows)
            baseline_rate = median(baseline, "tokens_per_second")
            current_rate = median(rows, "tokens_per_second")
            comparisons.append({
                "model": model,
                "model_key": key,
                "mode": mode,
                "context": args.context,
                "baseline_tokens_per_second": baseline_rate,
                "current_tokens_per_second": current_rate,
                "throughput_ratio": current_rate / baseline_rate,
                "output_contract_exact": exact,
            })
    correctness = all(row["output_contract_exact"] for row in comparisons)
    performance = all(row["throughput_ratio"] >= 0.95 for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and performance else "fail",
        "record_type": "per_device_hipblaslt_handle_regression",
        "raw_processes": len(current_rows),
        "correctness_gate": correctness,
        "performance_gate": performance,
        "minimum_throughput_ratio": min(
            row["throughput_ratio"] for row in comparisons),
        "comparisons": comparisons,
        "decision": "keep per-device handles" if correctness and performance
                    else "reject per-device handles",
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in current_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
