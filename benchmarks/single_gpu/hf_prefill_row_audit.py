#!/usr/bin/env python3
"""Audit whether an official B2 prefill difference follows the local row."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import hf_continuous_matrix as matrix


CASES = {
    "single_5": {"slots": 1, "prompts": [32], "outputs": [16],
                 "offsets": [5], "targets": [0]},
    "pair_4_5": {"slots": 2, "prompts": [32, 32], "outputs": [8, 16],
                 "offsets": [4, 5], "targets": [1]},
    "pair_5_4": {"slots": 2, "prompts": [32, 32], "outputs": [16, 8],
                 "offsets": [5, 4], "targets": [0]},
    "duplicate_5": {"slots": 2, "prompts": [32, 32], "outputs": [16, 16],
                    "offsets": [5, 5], "targets": [0, 1]},
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    result = parser.parse_args()
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    if result.runs <= 0 or result.timeout_seconds <= 0:
        parser.error("runs and timeout must be positive")
    return result


def command(binary: Path, model: dict, case: dict) -> list[str]:
    result = matrix.command(binary, model, case, 0, 1)
    result.extend([
        "--continuous-diagnostics", "true",
        "--continuous-prompt-offsets", ",".join(map(str, case["offsets"])),
    ])
    return result


def selection(record: dict, request: int, generated_index: int) -> dict:
    rows = [row for row in record["selection_diagnostics"]
            if int(row["request_id"]) == request + 1 and
            int(row["generated_index"]) == generated_index]
    if len(rows) != 1:
        raise RuntimeError("row audit diagnostic is missing or duplicate")
    return rows[0]


def numeric_signature(row: dict) -> tuple:
    return (row["device_selected_token"], row["top1_token"], row["top1_logit"],
            row["top2_token"], row["top2_logit"], row["top1_top2_margin"],
            row["cache_position"], row["logit_batch_size"], row["logit_source"])


def summarize(records: list[dict], model: str, runs: int) -> dict:
    by_case = {}
    for case_name, case in CASES.items():
        selected = [row for row in records if row.get("case") == case_name]
        if len(selected) != runs or any(row.get("status") != "pass" for row in selected):
            raise RuntimeError(f"incomplete row audit: {case_name}")
        if any(row["generated_tokens"] != selected[0]["generated_tokens"]
               for row in selected[1:]) or any(
                   row.get("prompt_offsets") != case["offsets"] for row in selected):
            raise RuntimeError(f"row audit input/output changed: {case_name}")
        by_case[case_name] = selected

    single = by_case["single_5"][0]["generated_tokens"][0]
    target_rows = []
    for case_name, case in CASES.items():
        for request in case["targets"]:
            generated = by_case[case_name][0]["generated_tokens"][request]
            prefill = [selection(row, request, 0) for row in by_case[case_name]]
            decision = [selection(row, request, 4) for row in by_case[case_name]]
            target_rows.append({
                "case": case_name,
                "request": request,
                "local_prefill_row": request,
                "offset": case["offsets"][request],
                "generated_equal_to_single": generated == single,
                "first_difference_vs_single": matrix.token_difference(
                    [single], [generated])["first_difference"],
                "prefill_signatures_stable": all(
                    numeric_signature(row) == numeric_signature(prefill[0])
                    for row in prefill[1:]),
                "prefill_diagnostic": prefill[0],
                "decision_signatures_stable": all(
                    numeric_signature(row) == numeric_signature(decision[0])
                    for row in decision[1:]),
                "decision_diagnostic": decision[0],
            })

    targets = {(row["case"], row["request"]): row for row in target_rows}
    pair_row0 = targets[("pair_5_4", 0)]
    pair_row1 = targets[("pair_4_5", 1)]
    duplicate_row0 = targets[("duplicate_5", 0)]
    duplicate_row1 = targets[("duplicate_5", 1)]
    b2_targets_equal = (
        by_case["pair_5_4"][0]["generated_tokens"][0] ==
        by_case["pair_4_5"][0]["generated_tokens"][1] ==
        by_case["duplicate_5"][0]["generated_tokens"][0] ==
        by_case["duplicate_5"][0]["generated_tokens"][1])
    duplicate_prefill_equal = (
        numeric_signature(duplicate_row0["prefill_diagnostic"]) ==
        numeric_signature(duplicate_row1["prefill_diagnostic"]))
    swapped_prefill_equal = (
        numeric_signature(pair_row0["prefill_diagnostic"]) ==
        numeric_signature(pair_row1["prefill_diagnostic"]))
    return {
        "schema_version": 1,
        "track": "official_b2_prefill_row_audit",
        "status": "pass" if b2_targets_equal and duplicate_prefill_equal and
        swapped_prefill_equal else "row_dependent_failure",
        "model": model,
        "runs": runs,
        "cases": CASES,
        "target_rows": target_rows,
        "b2_target_outputs_equal_across_row_order_and_duplicates": b2_targets_equal,
        "duplicate_b2_prefill_numeric_signatures_equal": duplicate_prefill_equal,
        "swapped_b2_target_prefill_signatures_equal": swapped_prefill_equal,
        "single_and_b2_outputs_differ": not pair_row0["generated_equal_to_single"],
        "conclusion": (
            "the B1/B2 difference does not follow local row or prompt-copy order"
        ),
        "measurement_boundary": "host-copy diagnostics; no performance claim",
    }


def main() -> int:
    args = options()
    model = matrix.load_models(args.manifest, [args.model])[0]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    records = []
    for case_name, case in CASES.items():
        for process_run in range(1, args.runs + 1):
            completed = subprocess.run(
                command(args.binary, model, case), capture_output=True, text=True,
                timeout=args.timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise RuntimeError("row audit worker must emit one JSON line")
            record = matrix.validate(
                json.loads(lines[0]), model, case_name, case, 0, 1)
            record["process_run"] = process_run
            records.append(record)
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"case": case_name, "process_run": process_run,
                              "status": "pass"}, sort_keys=True), flush=True)
    summary = summarize(records, model["name"], args.runs)
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
