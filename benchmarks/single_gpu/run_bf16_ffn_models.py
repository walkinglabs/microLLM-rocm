#!/usr/bin/env python3
"""Measure single-representation BF16 FFN inference on official HF checkpoints."""

from __future__ import annotations

import argparse
import array
import json
import statistics
import subprocess
import tempfile
from pathlib import Path


POLICIES = (("fp32", "false"), ("bf16_ffn", "true"))


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--pytorch-python", type=Path)
    parser.add_argument("--pytorch-runner", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prefill-warmup", type=int, default=2)
    parser.add_argument("--prefill-steps", type=int, default=5)
    result = parser.parse_args()
    if result.runs <= 0 or result.prefill_warmup < 0 or result.prefill_steps <= 0:
        parser.error("runs/prefill-steps must be positive and warmup nonnegative")
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    if (result.pytorch_python is None) != (result.pytorch_runner is None):
        parser.error("provide both --pytorch-python and --pytorch-runner, or neither")
    if result.pytorch_python is not None and (
        not result.pytorch_python.is_file() or not result.pytorch_runner.is_file()
    ):
        parser.error("PyTorch interpreter and runner must exist")
    return result


def models(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("manifest schema_version must be 1")
    for model in document.get("models", []):
        inference = model.get("inference", {})
        required = ("name", "revision", "config", "weights")
        if any(not model.get(key) for key in required):
            raise RuntimeError("model entry lacks identity or checkpoint path")
        if not inference.get("token_ids") or not inference.get("expected_generated_tokens"):
            raise RuntimeError(f"{model['name']} lacks fixed token evidence")
        if not Path(model["config"]).is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"{model['name']} checkpoint is unavailable")
    return document["models"]


def floats(path: Path):
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def maximum_difference(left, right):
    if len(left) != len(right):
        raise RuntimeError("logit vector length changed")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def median(records, field):
    return statistics.median(float(record[field]) for record in records)


def main():
    args = options()
    records = []
    with tempfile.TemporaryDirectory(prefix="microllm-bf16-model-") as directory:
        temporary = Path(directory)
        for model in models(args.manifest):
            reference_logits = None
            inference = model["inference"]
            for process_run in range(1, args.runs + 1):
                for policy, enabled in POLICIES:
                    logits_path = temporary / f"{model['name']}-{policy}-{process_run}.bin"
                    command = [
                        str(args.binary),
                        "--config", model["config"],
                        "--weights", model["weights"],
                        "--tokens", ",".join(str(token) for token in inference["token_ids"]),
                        "--device", "hip",
                        "--top-k", "10",
                        "--new-tokens", str(inference["new_tokens"]),
                        "--warmup", str(inference.get("warmup", 2)),
                        "--steps", str(inference.get("steps", 5)),
                        "--prefill-warmup", str(args.prefill_warmup),
                        "--prefill-steps", str(args.prefill_steps),
                        "--bf16-ffn", enabled,
                        "--logits-output", str(logits_path),
                    ]
                    completed = subprocess.run(
                        command, check=True, text=True, capture_output=True
                    )
                    record = json.loads(completed.stdout)
                    current_logits = floats(logits_path)
                    if reference_logits is None:
                        if policy != "fp32":
                            raise RuntimeError("first result must establish FP32 logits")
                        reference_logits = current_logits
                    difference = maximum_difference(current_logits, reference_logits)
                    expected = inference["expected_generated_tokens"]
                    if record.get("generated_tokens") != expected:
                        raise RuntimeError(
                            f"{model['name']}/{policy} generated token mismatch: "
                            f"{record.get('generated_tokens')}"
                        )
                    record.update({
                        "record_type": "bf16_ffn_official_model_measurement",
                        "model": model["name"],
                        "revision": model["revision"],
                        "policy": policy,
                        "process_run": process_run,
                        "max_abs_logit_difference_vs_fp32_run1": difference,
                        "exact_expected_tokens": True,
                    })
                    records.append(record)
                    print(json.dumps(record, sort_keys=True))

        if args.pytorch_python is not None:
            for model in models(args.manifest):
                for process_run in range(1, args.runs + 1):
                    command = [
                        str(args.pytorch_python), str(args.pytorch_runner),
                        "--manifest", str(args.manifest),
                        "--device", "cuda", "--dtype", "bf16", "--modes", "infer",
                        "--allow-amdsmi-fallback",
                        "--worker-model", model["name"], "--worker-mode", "infer",
                    ]
                    completed = subprocess.run(
                        command, check=True, text=True, capture_output=True
                    )
                    record = json.loads(completed.stdout)
                    if record.get("generated_tokens") != model["inference"][
                        "expected_generated_tokens"
                    ]:
                        raise RuntimeError(
                            f"{model['name']}/pytorch_bf16 generated token mismatch"
                        )
                    record.update({
                        "record_type": "bf16_ffn_official_model_measurement",
                        "policy": "pytorch_bf16",
                        "process_run": process_run,
                        "exact_expected_tokens": True,
                    })
                    records.append(record)
                    print(json.dumps(record, sort_keys=True))

    rows = []
    for model in models(args.manifest):
        grouped = {
            policy: [record for record in records
                     if record["model"] == model["name"] and record["policy"] == policy]
            for policy, _ in POLICIES
        }
        fp32 = grouped["fp32"]
        bf16 = grouped["bf16_ffn"]
        fp32_decode = median(fp32, "decode_tokens_per_second")
        bf16_decode = median(bf16, "decode_tokens_per_second")
        fp32_prefill = median(fp32, "prefill_tokens_per_second")
        bf16_prefill = median(bf16, "prefill_tokens_per_second")
        row = {
            "model": model["name"],
            "revision": model["revision"],
            "fp32_decode_tokens_per_second": fp32_decode,
            "bf16_ffn_decode_tokens_per_second": bf16_decode,
            "decode_speedup": bf16_decode / fp32_decode,
            "fp32_prefill_tokens_per_second": fp32_prefill,
            "bf16_ffn_prefill_tokens_per_second": bf16_prefill,
            "prefill_speedup": bf16_prefill / fp32_prefill,
            "fp32_engine_current_bytes": median(fp32, "engine_current_bytes"),
            "bf16_ffn_engine_current_bytes": median(bf16, "engine_current_bytes"),
            "current_memory_ratio": (
                median(bf16, "engine_current_bytes") /
                median(fp32, "engine_current_bytes")
            ),
            "bf16_resident_weight_bytes": median(bf16, "resident_weight_bytes"),
            "max_abs_logit_difference": max(
                record["max_abs_logit_difference_vs_fp32_run1"] for record in bf16
            ),
            "all_exact_expected_tokens": all(
                record["exact_expected_tokens"] for record in bf16
            ),
        }
        pytorch = [record for record in records
                   if record["model"] == model["name"] and
                   record["policy"] == "pytorch_bf16"]
        if pytorch:
            pytorch_decode = median(pytorch, "decode_tokens_per_second")
            pytorch_prefill = median(pytorch, "prefill_tokens_per_second")
            row.update({
                "pytorch_bf16_decode_tokens_per_second": pytorch_decode,
                "pytorch_bf16_prefill_tokens_per_second": pytorch_prefill,
                "microllm_bf16_ffn_decode_ratio_vs_pytorch_bf16": (
                    bf16_decode / pytorch_decode
                ),
                "microllm_bf16_ffn_prefill_ratio_vs_pytorch_bf16": (
                    bf16_prefill / pytorch_prefill
                ),
                "pytorch_bf16_current_allocated_bytes": median(
                    pytorch, "device_current_allocated_bytes"
                ),
                "all_pytorch_exact_expected_tokens": all(
                    record["exact_expected_tokens"] for record in pytorch
                ),
            })
        rows.append(row)
    summary = {
        "schema_version": 1,
        "track": "bf16_ffn_official_models",
        "aggregation": "median of three independent process results by default",
        "rows": rows,
    }
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
