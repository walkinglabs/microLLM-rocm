#!/usr/bin/env python3
"""Locate the first slot-count-dependent token and preserve its top-2 logits."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import hf_continuous_matrix as matrix


CASES = {name: {**case, "batch_equal_length_prefill": True}
         for name, case in matrix.SUITES["slot-sweep"].items()
         if case["group"] == "short"}
for slots in (4, 8):
    CASES[f"short_s{slots}_serial_prefill"] = {
        **CASES[f"short_s{slots}"],
        "batch_equal_length_prefill": False,
    }


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--raw-input", type=Path)
    parser.add_argument("--pytorch-reference", type=Path)
    result = parser.parse_args()
    if result.raw_input is None and (
            result.manifest is None or result.binary is None or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("manifest and binary must exist for device collection")
    if result.raw_input is not None and not result.raw_input.is_file():
        parser.error("raw input must exist")
    if result.pytorch_reference is not None and not result.pytorch_reference.is_file():
        parser.error("PyTorch reference must exist")
    if result.runs <= 0 or result.timeout_seconds <= 0:
        parser.error("runs and timeout must be positive")
    return result


def diagnostic(record: dict, request: int, generated_index: int) -> dict:
    matches = [row for row in record["selection_diagnostics"]
               if int(row["request_id"]) == request + 1 and
               int(row["generated_index"]) == generated_index]
    if len(matches) != 1:
        raise RuntimeError("selection diagnostic is missing or duplicate")
    return matches[0]


def command(binary: Path, model: dict, case: dict) -> list[str]:
    result = matrix.command(binary, model, case, 0, 1)
    result.extend(["--continuous-diagnostics", "true"])
    if not case["batch_equal_length_prefill"]:
        result.extend(["--continuous-prefill-batch", "false"])
    return result


def summarize(records: list[dict], model: str, runs: int) -> dict:
    by_case = {}
    for case_name in CASES:
        selected = [row for row in records if row.get("case") == case_name]
        if len(selected) != runs or any(row.get("status") != "pass" for row in selected):
            raise RuntimeError(f"incomplete diagnostic case: {case_name}")
        if any(row["generated_tokens"] != selected[0]["generated_tokens"]
               for row in selected[1:]):
            raise RuntimeError(f"generated tokens changed across runs: {case_name}")
        if any(int(row.get("selection_diagnostic_count", -1)) !=
               sum(CASES[case_name]["outputs"]) for row in selected):
            raise RuntimeError(f"diagnostic count changed: {case_name}")
        if any(not diagnostic.get("device_argmax_matches_top1")
               for row in selected for diagnostic in row["selection_diagnostics"]):
            raise RuntimeError(f"device/host argmax changed: {case_name}")
        by_case[case_name] = selected

    baseline = by_case["short_s1"][0]["generated_tokens"]
    comparisons = []
    first = None
    for case_name, case in CASES.items():
        difference = matrix.token_difference(
            baseline, by_case[case_name][0]["generated_tokens"])
        comparisons.append({
            "case": case_name,
            "slots": case["slots"],
            "difference_vs_s1": difference,
        })
        if first is None and not difference["exact"]:
            first = difference["first_difference"]
    if first is None:
        raise RuntimeError("slot diagnostic needs a stable cross-slot divergence")

    evidence = []
    for case_name, case in CASES.items():
        selected = by_case[case_name]
        rows = [diagnostic(row, first["request"], first["token"])
                for row in selected]
        margins = sorted(float(row["top1_top2_margin"]) for row in rows)
        evidence.append({
            "case": case_name,
            "slots": case["slots"],
            "request": first["request"],
            "generated_index": first["token"],
            "selected_tokens": sorted({int(row["device_selected_token"])
                                        for row in rows}),
            "top1_tokens": sorted({int(row["top1_token"]) for row in rows}),
            "top2_tokens": sorted({int(row["top2_token"]) for row in rows}),
            "top1_logit_min": min(float(row["top1_logit"]) for row in rows),
            "top1_logit_max": max(float(row["top1_logit"]) for row in rows),
            "top2_logit_min": min(float(row["top2_logit"]) for row in rows),
            "top2_logit_max": max(float(row["top2_logit"]) for row in rows),
            "margin_min": margins[0],
            "margin_p50": statistics.median(margins),
            "margin_max": margins[-1],
            "logit_sources": sorted({row["logit_source"] for row in rows}),
            "logit_batch_sizes": sorted({int(row["logit_batch_size"])
                                          for row in rows}),
            "cache_positions": sorted({int(row["cache_position"]) for row in rows}),
            "scheduler_steps": sorted({int(row["scheduler_step"]) for row in rows}),
            "stable_across_runs": all(row == rows[0] for row in rows[1:]),
        })
    return {
        "schema_version": 1,
        "track": "official_continuous_slot_divergence",
        "status": "complete_with_recorded_accuracy_failure",
        "model": model,
        "runs": runs,
        "cases": CASES,
        "first_difference": first,
        "comparisons": comparisons,
        "diagnostic_evidence": evidence,
        "measurement_boundary": (
            "diagnostic logits are copied to the host and are excluded from "
            "performance claims"
        ),
    }


def compare_to_pytorch(records: list[dict], reference: dict) -> dict:
    expected = reference["generated_tokens"]
    comparisons = []
    for case_name in CASES:
        selected = next(row for row in records if row["case"] == case_name)
        comparisons.append({
            "case": case_name,
            "slots": CASES[case_name]["slots"],
            "batch_equal_length_prefill":
                CASES[case_name]["batch_equal_length_prefill"],
            "difference_vs_pytorch": matrix.token_difference(
                expected, selected["generated_tokens"]),
        })
    original = {row["case"]: row for row in comparisons}
    return {
        "reference_mode": reference.get("serving_mode"),
        "reference_precision": reference.get("precision"),
        "comparisons": comparisons,
        "original_divergence_request": 5,
        "original_divergence_token": 4,
        "default_s4_matches_reference_at_original_divergence":
            5 not in original["short_s4"]["difference_vs_pytorch"][
                "differing_requests"],
        "serial_s4_matches_reference_at_original_divergence":
            5 not in original["short_s4_serial_prefill"][
                "difference_vs_pytorch"]["differing_requests"],
        "boundary": (
            "PyTorch is sequential full-BF16; this comparison is a token "
            "oracle, not a matched scheduler performance result"
        ),
    }


def main() -> int:
    args = options()
    if args.raw_input is not None:
        records = [json.loads(line) for line in
                   args.raw_input.read_text(encoding="utf-8").splitlines()]
        models = {row["model"] for row in records}
        if len(models) != 1:
            raise RuntimeError("raw diagnostic input must contain one model")
        summary = summarize(records, models.pop(), args.runs)
        if args.pytorch_reference is not None:
            summary["pytorch_comparison"] = compare_to_pytorch(
                records,
                json.loads(args.pytorch_reference.read_text(encoding="utf-8")))
        args.output_directory.mkdir(parents=True, exist_ok=True)
        (args.output_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return 0
    models = matrix.load_models(args.manifest, [args.model])
    model = models[0]
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
                raise RuntimeError("diagnostic worker must emit one JSON line")
            record = matrix.validate(
                json.loads(lines[0]), model, case_name, case, 0, 1)
            record["process_run"] = process_run
            records.append(record)
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"model": model["name"], "case": case_name,
                              "process_run": process_run, "status": "pass"},
                             sort_keys=True), flush=True)
    summary = summarize(records, model["name"], args.runs)
    if args.pytorch_reference is not None:
        summary["pytorch_comparison"] = compare_to_pytorch(
            records,
            json.loads(args.pytorch_reference.read_text(encoding="utf-8")))
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
