#!/usr/bin/env python3
"""Isolate DeepSeek cross-batch drift across BF16 weight islands."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


COMMON_SPEC = importlib.util.spec_from_file_location(
    "audit_cached_cross_batch_logits_common",
    Path(__file__).with_name("audit_cached_cross_batch_logits.py"))
COMMON = importlib.util.module_from_spec(COMMON_SPEC)
assert COMMON_SPEC.loader is not None
COMMON_SPEC.loader.exec_module(COMMON)

POLICIES = {
    "fp32-linear": (False, False),
    "bf16-ffn": (True, False),
    "bf16-attention": (False, True),
    "bf16-both": (True, True),
}
BATCHES = (1, 2, 4, 8)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context <= 0 or args.runs < 2 or args.warmup < 0 or
            args.timeout_seconds <= 0):
        parser.error("precision-isolation inputs are invalid")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def command(args: argparse.Namespace, model: dict, policy: str, batch: int,
            output: Path) -> list[str]:
    bf16_ffn, bf16_attention = POLICIES[policy]
    return [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", COMMON.expanded(model["inference"]["token_ids"], args.context),
        "--device", "hip", "--top-k", "1", "--batch", str(batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", "bf16",
        "--cache-capacity", str(args.context + 1), "--new-tokens", "1",
        "--warmup", str(args.warmup), "--steps", "1",
        "--prefill-warmup", str(args.warmup), "--prefill-steps", "1",
        "--bf16-ffn", str(bf16_ffn).lower(),
        "--bf16-attention", str(bf16_attention).lower(),
        "--workload", "decode", "--cache-logits-output", str(output),
        "--cache-logits-step", "0",
    ]


def run_one(args: argparse.Namespace, model: dict, vocabulary: int, policy: str,
            batch: int, run: int, temporary: Path) -> tuple[dict, list[float]]:
    output = temporary / f"{policy}-b{batch}-r{run}.bin"
    completed = subprocess.run(
        command(args, model, policy, batch, output), text=True,
        capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    record = COMMON.last_json(completed.stdout)
    bf16_ffn, bf16_attention = POLICIES[policy]
    required = {
        "status": "pass", "batch": batch, "token_count": args.context,
        "decode_tokens": 1, "cache_logits_step": 0,
        "cached_attention_materialized_policy": "auto-enabled",
        "cached_attention_materialized_scores": True,
        "cached_attention_pv_splits": 0, "kv_cache_dtype": "bf16",
    }
    for name, wanted in required.items():
        if record.get(name) != wanted:
            raise ValueError(f"{policy} {name} expected {wanted!r}")
    if ((int(record.get("bf16_ffn_converted_tensors", 0)) > 0) != bf16_ffn or
            (int(record.get("bf16_attention_converted_tensors", 0)) > 0) !=
                bf16_attention):
        raise ValueError(f"{policy} did not execute its requested weight island")
    logits = COMMON.read_logits(output, batch, vocabulary)
    rows = [logits[index * vocabulary:(index + 1) * vocabulary]
            for index in range(batch)]
    within = [COMMON.error(rows[0], row) for row in rows[1:]]
    host_tokens = [COMMON.argmax(row) for row in rows]
    device_token = int(record["generated_tokens"][0])
    record.update({
        "schema_version": 1,
        "record_type": "cross_batch_precision_measurement",
        "model": args.model, "revision": model["revision"],
        "precision_island": policy, "context": args.context,
        "decode_step": 0, "process_run": run,
        "complete_logit_elements": len(logits),
        "within_batch_bitwise_equal": all(item[2] for item in within),
        "within_batch_maximum_error": max((item[0] for item in within), default=0.0),
        "within_batch_rms_error": max((item[1] for item in within), default=0.0),
        "host_argmax_tokens": host_tokens,
        "device_argmax_token": device_token,
        "host_device_argmax_equal": all(token == device_token for token in host_tokens),
    })
    return record, logits


def summarize(measurements: list[tuple[dict, list[float]]],
              vocabulary: int) -> dict:
    by_key = {(row["precision_island"], row["batch"], row["process_run"]):
              (row, logits) for row, logits in measurements}
    cases = []
    policies = []
    for policy in POLICIES:
        reference = by_key[(policy, 1, 1)][1]
        policy_cases = []
        for batch in BATCHES:
            samples = [by_key[(policy, batch, run)] for run in (1, 2)]
            first_rows = [logits[:vocabulary] for _, logits in samples]
            cross = [COMMON.error(reference, values) for values in first_rows]
            repeated = COMMON.error(first_rows[0], first_rows[1])
            case = {
                "precision_island": policy, "batch": batch, "runs": 2,
                "complete_values_compared_per_run": vocabulary,
                "cross_batch_maximum_error": max(item[0] for item in cross),
                "cross_batch_maximum_rms_error": max(item[1] for item in cross),
                "cross_batch_bitwise_equal": all(item[2] for item in cross),
                "repeat_bitwise_equal": repeated[2],
                "within_batch_bitwise_equal": all(
                    row["within_batch_bitwise_equal"] for row, _ in samples),
                "host_device_argmax_equal": all(
                    row["host_device_argmax_equal"] for row, _ in samples),
                "device_argmax_tokens": [row["device_argmax_token"]
                                         for row, _ in samples],
                "peak_bytes": max(int(row["engine_peak_bytes"])
                                  for row, _ in samples),
            }
            cases.append(case)
            policy_cases.append(case)
        policies.append({
            "precision_island": policy,
            "maximum_cross_batch_error": max(
                case["cross_batch_maximum_error"] for case in policy_cases),
            "maximum_cross_batch_rms_error": max(
                case["cross_batch_maximum_rms_error"] for case in policy_cases),
            "cross_batch_bitwise_case_count": sum(
                case["cross_batch_bitwise_equal"] for case in policy_cases),
            "all_argmax_tokens_equal": len({
                token for case in policy_cases
                for token in case["device_argmax_tokens"]}) == 1,
        })
    return {
        "schema_version": 1,
        "record_type": "cross_batch_precision_isolation",
        "status": "pass", "process_rows": len(measurements),
        "case_rows": len(cases), "vocabulary_size": vocabulary,
        "policies": list(POLICIES), "batches": list(BATCHES),
        "runs_per_case": 2,
        "all_repeat_bitwise_equal": all(case["repeat_bitwise_equal"] for case in cases),
        "all_within_batch_bitwise_equal": all(
            case["within_batch_bitwise_equal"] for case in cases),
        "all_host_device_argmax_equal": all(
            case["host_device_argmax_equal"] for case in cases),
        "policy_summaries": policies, "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1280, 620
    maximum = max(row["maximum_cross_batch_error"]
                  for row in summary["policy_summaries"])
    scale = 750.0 / maximum if maximum else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:13px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">DeepSeek step0 cross-batch precision isolation</text>',
        '<text x="30" y="62" class="muted">B1 reference · FP32 linear and two BF16 islands · complete logits</text>',
    ]
    for policy_index, policy in enumerate(POLICIES):
        y0 = 92 + policy_index * 128
        parts.append(f'<text x="30" y="{y0 + 18}" class="label">{policy}</text>')
        rows = [row for row in summary["cases"] if row["precision_island"] == policy]
        for index, row in enumerate(rows):
            y = y0 + index * 26
            length = max(2.0, row["cross_batch_maximum_error"] * scale)
            color = "#22c55e" if row["cross_batch_bitwise_equal"] else "#ef4444"
            parts.extend((
                f'<text x="190" y="{y + 16}" class="label">B{row["batch"]}</text>',
                f'<rect x="235" y="{y}" width="{length:.2f}" height="20" rx="4" fill="{color}"/>',
                f'<text x="{250 + length:.2f}" y="{y + 15}" class="label">'
                f'Max {row["cross_batch_maximum_error"]:.3e} · '
                f'RMS {row["cross_batch_maximum_rms_error"]:.3e}</text>',
            ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = COMMON.model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    measurements = []
    with tempfile.TemporaryDirectory(prefix="microllm-cross-batch-precision-") as root:
        temporary = Path(root)
        for run in range(1, args.runs + 1):
            policy_order = list(POLICIES) if run % 2 else list(reversed(POLICIES))
            batch_order = list(BATCHES) if run % 2 else list(reversed(BATCHES))
            for policy in policy_order:
                for batch in batch_order:
                    sample = run_one(args, model, vocabulary, policy, batch,
                                     run, temporary)
                    measurements.append(sample)
                    print(json.dumps({
                        "precision_island": policy, "batch": batch,
                        "process_run": run,
                        "within_batch_bitwise_equal": sample[0][
                            "within_batch_bitwise_equal"],
                        "host_device_argmax_equal": sample[0][
                            "host_device_argmax_equal"],
                    }, sort_keys=True), flush=True)
    summary = summarize(measurements, vocabulary)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row, _ in measurements), encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "precision-isolation.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_cross_batch_precision: {error}", file=sys.stderr)
        raise SystemExit(2) from error
