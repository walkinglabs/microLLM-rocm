#!/usr/bin/env python3
"""Split the two minimal Qwen3 BF16 FFN layer sets by gate/up/down scope."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


AUDIT = Path(__file__).with_name("audit_qwen3_bf16_divergence.py")
POLICIES = ",".join((
    "micro-fp32-fp32", "micro-ffn-bf16-fp32",
    "micro-ffn-gate-bf16-fp32", "micro-ffn-up-bf16-fp32",
    "micro-ffn-down-bf16-fp32", "micro-ffn-gate-up-bf16-fp32",
    "micro-ffn-gate-down-bf16-fp32", "micro-ffn-up-down-bf16-fp32"))
SCOPED_POLICIES = (
    "micro-ffn-gate-bf16-fp32", "micro-ffn-up-bf16-fp32",
    "micro-ffn-down-bf16-fp32", "micro-ffn-gate-up-bf16-fp32",
    "micro-ffn-gate-down-bf16-fp32", "micro-ffn-up-down-bf16-fp32")
CASES = (
    {"name": "layers-0-1-2", "active_layers": [0, 1, 2]},
    {"name": "layers-3-4", "active_layers": [3, 4]},
)
FORCED_INPUTS = [14582, 1, 374, 264, 3491, 429, 374, 537, 264]


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


def aggregate(case_summaries: list[tuple[dict, dict]]) -> dict:
    rows = []
    for case, summary in case_summaries:
        if (summary.get("status") != "pass_diagnosed_precision_policy" or
                not all(summary.get("gates", {}).values()) or
                summary.get("forced_inputs") != FORCED_INPUTS):
            raise RuntimeError(f"projection case failed: {case['name']}")
        all_bf16 = policy(summary, "micro-ffn-bf16-fp32")
        scopes = [policy(summary, name) for name in SCOPED_POLICIES]
        rows.append({
            "name": case["name"], "active_layers": case["active_layers"],
            "all_projection_argmax": all_bf16["argmax_token"],
            "all_projection_margin": all_bf16["top1_top2_margin"],
            "scope_rows": [{
                "policy": row["policy"], "argmax_token": row["argmax_token"],
                "top1_top2_margin": row["top1_top2_margin"],
                "versus_oracle_maximum_error":
                    row["versus_torch_fp32_maximum_error"],
                "versus_oracle_rms_error": row["versus_torch_fp32_rms_error"],
                "within_batch_maximum_error": row["captured_rows_maximum_error"],
            } for row in scopes],
        })
    gates = {
        "both_all_projection_cases_flip":
            all(row["all_projection_argmax"] == 25 for row in rows),
        "all_partial_projection_cases_keep_oracle":
            all(scope["argmax_token"] == 320
                for row in rows for scope in row["scope_rows"]),
        "every_single_and_pair_scope_covered":
            all(len(row["scope_rows"]) == len(SCOPED_POLICIES) for row in rows),
    }
    return {
        "schema_version": 1, "record_type": "qwen3_bf16_ffn_projection_search",
        "status": "pass_all_three_projections_required" if all(gates.values()) else "fail",
        "model": "Qwen/Qwen3-0.6B", "context": 128, "batch": 2,
        "capture_step": 8, "forced_inputs": FORCED_INPUTS,
        "case_count": len(rows), "scope_rows": len(rows) * len(SCOPED_POLICIES),
        "gates": gates, "cases": rows,
        "conclusion": (
            "gate, up, or down alone and every two-projection scope keep FP32 argmax; "
            "all three BF16 projections are required in both tested minimal layer sets"),
        "boundary": (
            "fixed two layer sets and one forced T128/B2 state; this does not prove "
            "all contexts need three projections"),
    }


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    combined_raw = []
    all_layers = set(range(28))
    for case in CASES:
        output = args.output_directory / case["name"]
        fp32_layers = sorted(all_layers - set(case["active_layers"]))
        command = [
            str(args.pytorch_python), str(AUDIT),
            "--manifest", str(args.manifest), "--binary", str(args.binary),
            "--pytorch-python", str(args.pytorch_python),
            "--output-directory", str(output), "--context", "128",
            "--batch", "2", "--decode-tokens", "9", "--capture-step", "8",
            "--forced-inputs", ",".join(str(token) for token in FORCED_INPUTS),
            "--micro-policies", POLICIES, "--micro-ffn-fp32-layers",
            ",".join(str(layer) for layer in fp32_layers),
            "--timeout-seconds", str(args.timeout_seconds),
        ]
        if args.allow_amdsmi_fallback:
            command.append("--allow-amdsmi-fallback")
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=args.timeout_seconds)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{case['name']} failed: " +
                (completed.stderr.strip() or completed.stdout.strip()))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        summaries.append((case, summary))
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["projection_case"] = case["name"]
            combined_raw.append(record)
        print(json.dumps({
            "case": case["name"], "status": summary["status"],
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
