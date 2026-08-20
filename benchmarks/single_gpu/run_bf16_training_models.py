#!/usr/bin/env python3
"""Run the official FP32-master BF16 training comparison matrix."""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--micro-binary", required=True, type=Path)
    parser.add_argument("--pytorch-python", required=True, type=Path)
    parser.add_argument("--pytorch-runner", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    result = parser.parse_args()
    if result.runs <= 0:
        parser.error("runs must be positive")
    for path in (result.manifest, result.micro_binary, result.pytorch_python,
                 result.pytorch_runner):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def run(command):
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    record = json.loads(completed.stdout)
    if not record.get("parameter_changed"):
        raise RuntimeError("training command did not update its observed parameter")
    for key in ("first_loss", "final_loss", "tokens_per_second"):
        if not math.isfinite(float(record[key])):
            raise RuntimeError(f"training command emitted non-finite {key}")
    return record


def median(records, field):
    return statistics.median(float(record[field]) for record in records)


def main():
    args = options()
    models = json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
    records = []
    for model in models:
        training = model["training"]
        for process_run in range(1, args.runs + 1):
            for policy, precision in (("microllm_fp32", "fp32"),
                                      ("microllm_bf16_fp32_master", "bf16")):
                command = [
                    str(args.micro_binary), "--config", model["config"],
                    "--weights", model["weights"], "--tokens", training["tokens"],
                    "--device", "hip", "--learning-rate", str(training["learning_rate"]),
                    "--warmup", str(training.get("warmup", 2)),
                    "--steps", str(training.get("steps", 5)),
                    "--linear-precision", precision,
                ]
                if precision == "bf16":
                    command.extend(("--bf16-weight-mirrors", "false"))
                record = run(command)
                record.update({
                    "record_type": "bf16_training_official_model_measurement",
                    "model": model["name"], "revision": model["revision"],
                    "policy": policy, "process_run": process_run,
                })
                records.append(record)
                print(json.dumps(record, sort_keys=True))

            pytorch_command = [
                str(args.pytorch_python), str(args.pytorch_runner),
                "--manifest", str(args.manifest), "--device", "cuda",
                "--dtype", "bf16_amp", "--modes", "train", "--allow-amdsmi-fallback",
                "--worker-model", model["name"], "--worker-mode", "train",
            ]
            record = run(pytorch_command)
            record.update({
                "record_type": "bf16_training_official_model_measurement",
                "policy": "pytorch_bf16", "process_run": process_run,
            })
            records.append(record)
            print(json.dumps(record, sort_keys=True))

    rows = []
    for model in models:
        groups = {
            policy: [record for record in records
                     if record["model"] == model["name"] and record["policy"] == policy]
            for policy in ("microllm_fp32", "microllm_bf16_fp32_master", "pytorch_bf16")
        }
        fp32 = groups["microllm_fp32"]
        bf16 = groups["microllm_bf16_fp32_master"]
        pytorch = groups["pytorch_bf16"]
        bf16_tps = median(bf16, "tokens_per_second")
        rows.append({
            "model": model["name"], "revision": model["revision"],
            "microllm_fp32_tokens_per_second": median(fp32, "tokens_per_second"),
            "microllm_bf16_tokens_per_second": bf16_tps,
            "pytorch_bf16_amp_tokens_per_second": median(pytorch, "tokens_per_second"),
            "bf16_speedup_vs_microllm_fp32": bf16_tps / median(fp32, "tokens_per_second"),
            "microllm_bf16_ratio_vs_pytorch_bf16_amp": bf16_tps /
                median(pytorch, "tokens_per_second"),
            "microllm_fp32_peak_bytes": median(fp32, "engine_peak_bytes"),
            "microllm_bf16_peak_bytes": median(bf16, "engine_peak_bytes"),
            "pytorch_bf16_amp_peak_allocated_bytes": median(pytorch,
                                                          "device_peak_allocated_bytes"),
            "bf16_peak_ratio_vs_microllm_fp32": median(bf16, "engine_peak_bytes") /
                median(fp32, "engine_peak_bytes"),
            "microllm_bf16_first_loss": median(bf16, "first_loss"),
            "microllm_bf16_final_loss": median(bf16, "final_loss"),
            "pytorch_bf16_amp_first_loss": median(pytorch, "first_loss"),
            "pytorch_bf16_amp_final_loss": median(pytorch, "final_loss"),
            "all_updates_finite": all(record["parameter_changed"] for record in
                                       fp32 + bf16 + pytorch),
        })
    summary = {"schema_version": 1, "track": "bf16_fp32_master_training",
               "aggregation": "median of three independent processes by default",
               "rows": rows}
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8")
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
