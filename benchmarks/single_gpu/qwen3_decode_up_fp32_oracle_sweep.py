#!/usr/bin/env python3
"""Audit every known Qwen3 first-split state for phase-selective decode-up FP32."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


AUDIT = Path(__file__).with_name("audit_qwen3_bf16_divergence.py")
CANDIDATE = "micro-phase-decode-up-fp32"
POLICIES = "micro-fp32-fp32," + CANDIDATE
CASES = (
    {"name": "t32-b1-step1", "context": 32, "batch": 1,
     "decode_tokens": 4, "capture_step": 1, "forced_inputs": []},
    {"name": "t32-b2-step1", "context": 32, "batch": 2,
     "decode_tokens": 4, "capture_step": 1, "forced_inputs": []},
    {"name": "t128-b1-step8", "context": 128, "batch": 1,
     "decode_tokens": 9, "capture_step": 8, "forced_inputs": []},
    {"name": "t128-b2-step8", "context": 128, "batch": 2,
     "decode_tokens": 32, "capture_step": 8, "forced_inputs": []},
    {"name": "t128-b2-step22-forced", "context": 128, "batch": 2,
     "decode_tokens": 23, "capture_step": 22,
     "forced_inputs": [14582, 1, 374, 264, 3491, 429, 374, 537, 264,
                       320, 606, 11, 2265, 11, 323, 2400, 8, 315, 279,
                       279, 3491, 13, 5209]},
    {"name": "t512-b1-step2", "context": 512, "batch": 1,
     "decode_tokens": 4, "capture_step": 2, "forced_inputs": []},
    {"name": "t512-b2-step2-forced", "context": 512, "batch": 2,
     "decode_tokens": 3, "capture_step": 2,
     "forced_inputs": [14582, 198, 262]},
    {"name": "t512-b2-step8-forced", "context": 512, "batch": 2,
     "decode_tokens": 9, "capture_step": 8,
     "forced_inputs": [14582, 198, 262, 1096, 374, 279, 2038, 374, 264]},
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pytorch-python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
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


def aggregate(results: list[tuple[dict, dict]]) -> dict:
    if len(results) != len(CASES):
        raise RuntimeError("phase oracle sweep omitted a declared case")
    rows = []
    for case, summary in results:
        audit_gates = summary.get("gates", {})
        if (summary.get("status") not in {"pass_diagnosed_precision_policy", "fail"} or
                not audit_gates.get("shared_inputs_before_capture", False) or
                not audit_gates.get("fp32_oracle_argmax_agrees_with_micro_fp32", False) or
                not audit_gates.get("at_least_one_low_precision_policy_matches_fp32", False) or
                summary.get("context") != case["context"] or
                summary.get("batch") != case["batch"] or
                summary.get("capture_step") != case["capture_step"] or
                summary.get("forced_inputs") != case["forced_inputs"] or
                summary.get("micro_current_policy") != CANDIDATE):
            raise RuntimeError(f"oracle case contract failed: {case['name']}")
        oracle = policy(summary, "torch-fp32")
        micro_fp32 = policy(summary, "micro-fp32-fp32")
        candidate = policy(summary, CANDIDATE)
        torch_bf16 = policy(summary, "torch-bf16")
        rows.append({
            **case,
            "audit_status": summary["status"],
            "fp32_complete_logit_alignment_gate":
                audit_gates.get("fp32_implementations_aligned", False),
            "oracle_argmax": oracle["argmax_token"],
            "micro_fp32_argmax": micro_fp32["argmax_token"],
            "candidate_argmax": candidate["argmax_token"],
            "torch_bf16_argmax": torch_bf16["argmax_token"],
            "candidate_matches_oracle":
                candidate["argmax_token"] == oracle["argmax_token"],
            "torch_bf16_matches_oracle":
                torch_bf16["argmax_token"] == oracle["argmax_token"],
            "candidate_oracle_maximum_error":
                candidate["versus_torch_fp32_maximum_error"],
            "candidate_oracle_rms_error":
                candidate["versus_torch_fp32_rms_error"],
            "candidate_margin": candidate["top1_top2_margin"],
            "candidate_resident_weight_bytes": candidate["resident_weight_bytes"],
        })
    passed = sum(row["candidate_matches_oracle"] for row in rows)
    strict = sum(row["fp32_complete_logit_alignment_gate"] for row in rows)
    gates = {
        "all_eight_states_present": len(rows) == 8,
        "candidate_matches_every_fp32_argmax": passed == len(rows),
        "all_micro_fp32_argmax_match_torch_fp32": all(
            row["micro_fp32_argmax"] == row["oracle_argmax"] for row in rows),
        "candidate_resident_bytes_exact": all(
            row["candidate_resident_weight_bytes"] == 1_855_717_376
            for row in rows),
    }
    strict_gate = strict == len(rows)
    if all(gates.values()):
        status = ("pass_all_oracles" if strict_gate else
                  "pass_all_argmax_with_recorded_fp32_alignment_limit")
    else:
        status = "reject_oracle"
    return {
        "schema_version": 1,
        "record_type": "qwen3_decode_up_fp32_oracle_sweep",
        "status": status,
        "model": "Qwen/Qwen3-0.6B", "candidate": CANDIDATE,
        "case_count": len(rows), "oracle_cases_passed": passed,
        "strict_complete_logit_cases_passed": strict,
        "strict_complete_logit_case_count": len(rows),
        "strict_complete_logit_gate": strict_gate,
        "gates": gates, "rows": rows,
        "boundary": (
            "eight fixed first-split states with complete 151936-logit FP32 "
            "oracles; performance and unseen prompts remain separate gates"),
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
            "--micro-policies", POLICIES,
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
            command, capture_output=True, text=True,
            timeout=args.timeout_seconds)
        if completed.returncode not in {0, 2} or not (output / "summary.json").is_file():
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
            "candidate_argmax": policy(summary, CANDIDATE)["argmax_token"],
            "oracle_argmax": policy(summary, "torch-fp32")["argmax_token"],
        }, sort_keys=True), flush=True)
    summary = aggregate(summaries)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_raw),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"].startswith("pass_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
