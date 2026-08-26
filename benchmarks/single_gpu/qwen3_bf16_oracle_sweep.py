#!/usr/bin/env python3
"""Run one FP32-oracle audit for every unique Qwen3 BF16 token split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


AUDIT = Path(__file__).with_name("audit_qwen3_bf16_divergence.py")
FULL_POLICIES = ",".join((
    "micro-fp32-fp32", "micro-fp32-bf16",
    "micro-bf16-fp32", "micro-bf16-bf16"))
CURRENT_POLICIES = "micro-fp32-fp32,micro-bf16-bf16"
CASES = (
    {"name": "t32-b1-step1", "context": 32, "batch": 1,
     "decode_tokens": 4, "capture_step": 1,
     "matrix_decode_lengths": [4, 32], "forced_inputs": [],
     "micro_policies": FULL_POLICIES},
    {"name": "t32-b2-step1", "context": 32, "batch": 2,
     "decode_tokens": 4, "capture_step": 1,
     "matrix_decode_lengths": [4, 32], "forced_inputs": [],
     "micro_policies": FULL_POLICIES},
    {"name": "t128-b2-step8", "context": 128, "batch": 2,
     "decode_tokens": 32, "capture_step": 8,
     "matrix_decode_lengths": [32], "forced_inputs": [],
     "micro_policies": FULL_POLICIES},
    {"name": "t512-b1-step2", "context": 512, "batch": 1,
     "decode_tokens": 4, "capture_step": 2,
     "matrix_decode_lengths": [4, 32], "forced_inputs": [],
     "micro_policies": FULL_POLICIES},
    {"name": "t512-b2-step8-forced", "context": 512, "batch": 2,
     "decode_tokens": 9, "capture_step": 8,
     "matrix_decode_lengths": [32],
     "forced_inputs": [14582, 198, 262, 1096, 374, 279, 2038, 374, 264],
     "micro_policies": CURRENT_POLICIES},
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pytorch-python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    for path in (args.manifest, args.binary, args.pytorch_python, AUDIT):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def policy(summary: dict, name: str) -> dict:
    return next(row for row in summary["policy_rows"] if row["policy"] == name)


def aggregate(summaries: list[tuple[dict, dict]]) -> dict:
    if len(summaries) != len(CASES):
        raise RuntimeError("oracle sweep did not produce every declared case")
    case_rows = []
    matrix_rows = []
    micro_wins = 0
    torch_wins = 0
    for case, summary in summaries:
        if (summary.get("status") != "pass_diagnosed_precision_policy" or
                not all(summary.get("gates", {}).values()) or
                summary.get("context") != case["context"] or
                summary.get("batch") != case["batch"] or
                summary.get("capture_step") != case["capture_step"] or
                summary.get("forced_inputs") != case["forced_inputs"]):
            raise RuntimeError(f"oracle case contract failed: {case['name']}")
        winner = summary["oracle_matching_low_precision_policy"]
        if winner == "micro-bf16-bf16":
            micro_wins += 1
        elif winner == "torch-bf16":
            torch_wins += 1
        else:
            raise RuntimeError(f"unknown oracle-matching policy: {winner}")
        oracle = policy(summary, "torch-fp32")
        micro = policy(summary, "micro-bf16-bf16")
        torch = policy(summary, "torch-bf16")
        row = {
            "name": case["name"], "context": case["context"],
            "batch": case["batch"], "capture_step": case["capture_step"],
            "forced_inputs": case["forced_inputs"],
            "matrix_decode_lengths": case["matrix_decode_lengths"],
            "oracle_matching_policy": winner,
            "oracle_argmax": oracle["argmax_token"],
            "micro_mixed_argmax": micro["argmax_token"],
            "torch_bf16_argmax": torch["argmax_token"],
            "oracle_margin": oracle["top1_top2_margin"],
            "micro_mixed_margin": micro["top1_top2_margin"],
            "torch_bf16_margin": torch["top1_top2_margin"],
            "micro_mixed_oracle_maximum_error":
                micro["versus_torch_fp32_maximum_error"],
            "micro_mixed_oracle_rms_error":
                micro["versus_torch_fp32_rms_error"],
            "torch_bf16_oracle_maximum_error":
                torch["versus_torch_fp32_maximum_error"],
            "torch_bf16_oracle_rms_error":
                torch["versus_torch_fp32_rms_error"],
            "micro_mixed_within_batch_maximum_error":
                micro["captured_rows_maximum_error"],
            "torch_bf16_within_batch_maximum_error":
                torch["captured_rows_maximum_error"],
        }
        case_rows.append(row)
        for decode_tokens in case["matrix_decode_lengths"]:
            matrix_rows.append({
                "context": case["context"], "batch": case["batch"],
                "decode_tokens": decode_tokens,
                "first_difference": case["capture_step"],
                "oracle_matching_policy": winner,
            })
    micro_matrix_rows = sum(
        row["oracle_matching_policy"] == "micro-bf16-bf16"
        for row in matrix_rows)
    torch_matrix_rows = len(matrix_rows) - micro_matrix_rows
    return {
        "schema_version": 1, "record_type": "qwen3_bf16_oracle_sweep",
        "status": "pass_all_mismatches_attributed",
        "model": "Qwen/Qwen3-0.6B", "unique_oracle_cases": len(case_rows),
        "matrix_mismatch_rows": len(matrix_rows),
        "micro_oracle_case_wins": micro_wins,
        "torch_oracle_case_wins": torch_wins,
        "micro_oracle_matrix_rows": micro_matrix_rows,
        "torch_oracle_matrix_rows": torch_matrix_rows,
        "case_rows": case_rows, "matrix_rows": matrix_rows,
        "boundary": (
            "argmax attribution at the first shared-input split; cross-framework "
            "precision_mismatch rows remain visible and no performance claim is made"),
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    combined_raw = []
    for case in CASES:
        output = args.output_directory / case["name"]
        command = [
            str(args.pytorch_python), str(AUDIT),
            "--manifest", str(args.manifest), "--binary", str(args.binary),
            "--pytorch-python", str(args.pytorch_python),
            "--output-directory", str(output),
            "--context", str(case["context"]), "--batch", str(case["batch"]),
            "--decode-tokens", str(case["decode_tokens"]),
            "--capture-step", str(case["capture_step"]),
            "--micro-policies", case["micro_policies"],
            "--timeout-seconds", str(args.timeout_seconds),
        ]
        if case["forced_inputs"]:
            command.extend([
                "--forced-inputs",
                ",".join(str(token) for token in case["forced_inputs"]),
            ])
        if args.allow_amdsmi_fallback:
            command.append("--allow-amdsmi-fallback")
        completed = subprocess.run(
            command, capture_output=True, text=True,
            timeout=args.timeout_seconds)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{case['name']} failed: " +
                (completed.stderr.strip() or completed.stdout.strip()))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        summaries.append((case, summary))
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["oracle_case"] = case["name"]
            combined_raw.append(record)
        print(json.dumps({
            "case": case["name"], "status": summary["status"],
            "oracle_matching_policy": summary["oracle_matching_low_precision_policy"],
        }, sort_keys=True), flush=True)
    summary = aggregate(summaries)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_raw),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
