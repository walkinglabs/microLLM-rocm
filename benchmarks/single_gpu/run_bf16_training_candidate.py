#!/usr/bin/env python3
"""Measure a BF16 training candidate against the retained Experiment 037 summary."""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    result = parser.parse_args()
    if result.runs <= 0:
        parser.error("runs must be positive")
    for path in (result.manifest, result.binary, result.baseline_summary):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def median(records, field):
    return statistics.median(float(record[field]) for record in records)


def main():
    args = options()
    models = json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
    baseline = {row["model"]: row for row in
                json.loads(args.baseline_summary.read_text(encoding="utf-8"))["rows"]}
    records = []
    for model in models:
        training = model["training"]
        for process_run in range(1, args.runs + 1):
            command = [
                str(args.binary), "--config", model["config"],
                "--weights", model["weights"], "--tokens", training["tokens"],
                "--device", "hip", "--learning-rate", str(training["learning_rate"]),
                "--warmup", str(training.get("warmup", 2)),
                "--steps", str(training.get("steps", 5)),
                "--linear-precision", "bf16",
            ]
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            record = json.loads(completed.stdout)
            if not record.get("parameter_changed") or not math.isfinite(record["final_loss"]):
                raise RuntimeError(f"{model['name']} candidate did not make a finite update")
            record.update({
                "record_type": "bf16_training_shared_qkv_candidate",
                "model": model["name"], "revision": model["revision"],
                "process_run": process_run,
            })
            records.append(record)
            print(json.dumps(record, sort_keys=True))

    rows = []
    for model in models:
        group = [row for row in records if row["model"] == model["name"]]
        old = baseline[model["name"]]
        throughput = median(group, "tokens_per_second")
        rows.append({
            "model": model["name"], "revision": model["revision"],
            "candidate_tokens_per_second": throughput,
            "speedup_vs_bf16_independent_linears": throughput /
                old["microllm_bf16_tokens_per_second"],
            "ratio_vs_microllm_fp32": throughput /
                old["microllm_fp32_tokens_per_second"],
            "ratio_vs_pytorch_bf16_amp": throughput /
                old["pytorch_bf16_amp_tokens_per_second"],
            "peak_bytes": median(group, "engine_peak_bytes"),
            "peak_ratio_vs_baseline": median(group, "engine_peak_bytes") /
                old["microllm_bf16_peak_bytes"],
            "first_loss": median(group, "first_loss"),
            "final_loss": median(group, "final_loss"),
            "all_updates_finite": all(row["parameter_changed"] for row in group),
        })
    summary = {"schema_version": 1, "track": "bf16_training_shared_qkv",
               "aggregation": "median of three independent candidate processes by default",
               "rows": rows}
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
