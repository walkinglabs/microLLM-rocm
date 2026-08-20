#!/usr/bin/env python3
"""Measure shared-cast BF16 Attention+FFN against retained experiment summaries."""

import argparse
import array
import json
import statistics
import subprocess
import tempfile
from pathlib import Path


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--pytorch-summary", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    result = parser.parse_args()
    if result.runs <= 0:
        parser.error("runs must be positive")
    for path in (result.manifest, result.binary, result.baseline_summary,
                 result.pytorch_summary):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    return result


def load_models(path):
    return json.loads(path.read_text(encoding="utf-8"))["models"]


def read_floats(path):
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def run_json(command):
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def common_command(binary, model):
    inference = model["inference"]
    return [
        str(binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", ",".join(str(token) for token in inference["token_ids"]),
        "--device", "hip", "--top-k", "10",
    ]


def median(records, field):
    return statistics.median(float(record[field]) for record in records)


def main():
    args = options()
    models = load_models(args.manifest)
    baseline = {row["model"]: row for row in
                json.loads(args.baseline_summary.read_text(encoding="utf-8"))["rows"]}
    pytorch = {row["model"]: row for row in
               json.loads(args.pytorch_summary.read_text(encoding="utf-8"))["rows"]}
    records = []
    with tempfile.TemporaryDirectory(prefix="microllm-bf16-attention-") as directory:
        directory = Path(directory)
        for model in models:
            reference_path = directory / f"{model['name']}-fp32.bin"
            reference_command = common_command(args.binary, model) + [
                "--new-tokens", "0", "--prefill-warmup", "0", "--prefill-steps", "1",
                "--bf16-ffn", "false", "--bf16-attention", "false",
                "--workload", "prefill", "--logits-output", str(reference_path),
            ]
            reference_record = run_json(reference_command)
            reference_record.update({
                "record_type": "bf16_attention_reference",
                "model": model["name"], "revision": model["revision"],
                "policy": "fp32_reference_only",
            })
            records.append(reference_record)
            reference = read_floats(reference_path)
            for process_run in range(1, args.runs + 1):
                logits_path = directory / f"{model['name']}-candidate-{process_run}.bin"
                inference = model["inference"]
                command = common_command(args.binary, model) + [
                    "--new-tokens", str(inference["new_tokens"]),
                    "--warmup", str(inference.get("warmup", 2)),
                    "--steps", str(inference.get("steps", 5)),
                    "--prefill-warmup", "2", "--prefill-steps", "5",
                    "--bf16-ffn", "true", "--bf16-attention", "true",
                    "--workload", "both", "--logits-output", str(logits_path),
                ]
                record = run_json(command)
                actual = read_floats(logits_path)
                if len(actual) != len(reference):
                    raise RuntimeError("candidate logit count changed")
                difference = max(abs(left - right) for left, right in zip(actual, reference,
                                                                           strict=True))
                if record.get("generated_tokens") != inference["expected_generated_tokens"]:
                    raise RuntimeError(f"{model['name']} candidate token mismatch")
                record.update({
                    "record_type": "bf16_attention_candidate",
                    "model": model["name"], "revision": model["revision"],
                    "policy": "bf16_ffn_attention_shared_cast",
                    "process_run": process_run,
                    "max_abs_logit_difference_vs_fp32": difference,
                    "exact_expected_tokens": True,
                })
                records.append(record)
                print(json.dumps(record, sort_keys=True))

    rows = []
    for model in models:
        candidate = [row for row in records
                     if row["model"] == model["name"] and
                     row["policy"] == "bf16_ffn_attention_shared_cast"]
        old = baseline[model["name"]]
        torch = pytorch[model["name"]]
        decode = median(candidate, "decode_tokens_per_second")
        prefill = median(candidate, "prefill_tokens_per_second")
        rows.append({
            "model": model["name"], "revision": model["revision"],
            "candidate_decode_tokens_per_second": decode,
            "candidate_prefill_tokens_per_second": prefill,
            "decode_speedup_vs_bf16_ffn": decode / old["bf16_ffn_decode_tokens_per_second"],
            "prefill_speedup_vs_bf16_ffn": prefill / old["bf16_ffn_prefill_tokens_per_second"],
            "decode_ratio_vs_pytorch_bf16": decode /
                torch["pytorch_bf16_decode_tokens_per_second"],
            "prefill_ratio_vs_pytorch_bf16": prefill /
                torch["pytorch_bf16_prefill_tokens_per_second"],
            "engine_current_bytes": median(candidate, "engine_current_bytes"),
            "resident_weight_bytes": median(candidate, "resident_weight_bytes"),
            "max_abs_logit_difference": max(
                row["max_abs_logit_difference_vs_fp32"] for row in candidate),
            "all_exact_expected_tokens": all(row["exact_expected_tokens"] for row in candidate),
        })
    summary = {"schema_version": 1, "track": "bf16_attention_official_models",
               "aggregation": "median of three independent candidate processes by default",
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
