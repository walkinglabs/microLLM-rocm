#!/usr/bin/env python3
"""Preflight the Qwen3 gate-FP32/up-down-BF16 calibrated policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


AUDIT = Path(__file__).with_name("audit_qwen3_bf16_divergence.py")
SWEEP_PATH = Path(__file__).with_name("qwen3_bf16_oracle_sweep.py")
SWEEP_SPEC = importlib.util.spec_from_file_location("qwen3_oracle_sweep", SWEEP_PATH)
SWEEP = importlib.util.module_from_spec(SWEEP_SPEC)
assert SWEEP_SPEC.loader is not None
SWEEP_SPEC.loader.exec_module(SWEEP)
CANDIDATE = "micro-mixed-up-down-bf16"


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pytorch-python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    for path in (args.manifest, args.binary, args.pytorch_python, AUDIT, SWEEP_PATH):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def aggregate(summaries: list[tuple[dict, dict]]) -> dict:
    rows = []
    for case, summary in summaries:
        policies = {row["policy"]: row for row in summary["policy_rows"]}
        required_gates = summary.get("gates", {})
        if (not required_gates.get("shared_inputs_before_capture") or
                not required_gates.get("fp32_implementations_aligned") or
                not required_gates.get("fp32_oracle_argmax_agrees_with_micro_fp32") or
                summary.get("micro_current_policy") != CANDIDATE or
                CANDIDATE not in policies):
            raise RuntimeError(f"candidate oracle case failed: {case['name']}")
        oracle = policies["torch-fp32"]
        candidate = policies[CANDIDATE]
        torch = policies["torch-bf16"]
        rows.append({
            "name": case["name"], "context": case["context"],
            "batch": case["batch"], "capture_step": case["capture_step"],
            "forced_inputs": case["forced_inputs"],
            "oracle_argmax": oracle["argmax_token"],
            "candidate_argmax": candidate["argmax_token"],
            "torch_bf16_argmax": torch["argmax_token"],
            "candidate_matches_oracle":
                candidate["argmax_token"] == oracle["argmax_token"],
            "candidate_margin": candidate["top1_top2_margin"],
            "candidate_oracle_maximum_error":
                candidate["versus_torch_fp32_maximum_error"],
            "candidate_oracle_rms_error":
                candidate["versus_torch_fp32_rms_error"],
            "candidate_within_batch_maximum_error":
                candidate["captured_rows_maximum_error"],
        })
    gates = {
        "all_five_cases_present": len(rows) == len(SWEEP.CASES) == 5,
        "candidate_matches_every_fp32_argmax": all(
            row["candidate_matches_oracle"] for row in rows),
        "positive_top2_margin_everywhere": all(
            row["candidate_margin"] > 0.0 for row in rows),
    }
    return {
        "schema_version": 1, "record_type": "qwen3_bf16_gate_fp32_oracle",
        "status": "pass_oracle_preflight" if all(gates.values()) else "reject_oracle",
        "model": "Qwen/Qwen3-0.6B", "candidate": CANDIDATE,
        "candidate_description": "FFN gate FP32; FFN up/down and all Attention BF16; BF16 Cache",
        "case_count": len(rows), "gates": gates, "rows": rows,
        "boundary": (
            "five first-divergence states only; complete shape, batch invariance, "
            "resident and performance gates remain required"),
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    combined_raw = []
    for case in SWEEP.CASES:
        output = args.output_directory / case["name"]
        command = [
            str(args.pytorch_python), str(AUDIT),
            "--manifest", str(args.manifest), "--binary", str(args.binary),
            "--pytorch-python", str(args.pytorch_python),
            "--output-directory", str(output), "--context", str(case["context"]),
            "--batch", str(case["batch"]),
            "--decode-tokens", str(case["decode_tokens"]),
            "--capture-step", str(case["capture_step"]),
            "--micro-policies", "micro-fp32-fp32," + CANDIDATE,
            "--micro-current-policy", CANDIDATE,
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
            command, capture_output=True, text=True, timeout=args.timeout_seconds)
        if completed.returncode not in (0, 2) or not (output / "summary.json").is_file():
            raise RuntimeError(
                f"{case['name']} failed: " +
                (completed.stderr.strip() or completed.stdout.strip()))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        summaries.append((case, summary))
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["calibration_case"] = case["name"]
            combined_raw.append(record)
        print(json.dumps({
            "case": case["name"], "status": summary["status"],
            "matching": summary["oracle_matching_low_precision_policies"],
        }, sort_keys=True), flush=True)
    summary = aggregate(summaries)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_raw),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"].startswith("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
