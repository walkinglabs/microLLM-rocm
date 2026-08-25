#!/usr/bin/env python3
"""Measure a correctness-first rocWMMA QK tile matrix in fresh processes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


DEFAULT_SEQUENCES = (16, 32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_INNERS = (64, 128)


def positive_csv(text: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive values")
    return values


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("rocWMMA QK benchmark emitted no JSON object")


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--sequences", default=",".join(map(str, DEFAULT_SEQUENCES)))
    parser.add_argument("--inners", default=",".join(map(str, DEFAULT_INNERS)))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--minimum-long-scalar-speedup", type=float, default=5.0)
    parser.add_argument("--minimum-t512-blas-speedup", type=float, default=1.05)
    args = parser.parse_args()
    sequences = positive_csv(args.sequences, "sequences")
    inners = positive_csv(args.inners, "inners")
    if not args.binary.is_file() or args.runs <= 0 or args.warmup < 0 or \
            args.repetitions <= 0 or args.minimum_long_scalar_speedup <= 1.0 or \
            args.minimum_t512_blas_speedup <= 1.0:
        parser.error("rocWMMA QK matrix options are invalid")
    if any(sequence % 16 for sequence in sequences) or any(inner % 16 for inner in inners):
        parser.error("sequence and inner dimensions must be multiples of 16")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    base_cases = [(sequence, inner) for sequence in sequences for inner in inners]
    for process_run in range(1, args.runs + 1):
        cases = list(base_cases)
        if process_run % 2 == 0:
            cases.reverse()
        for case_index, (sequence, inner) in enumerate(cases):
            tile = 16 if sequence < 32 else 32
            command = [
                str(args.binary), "--rows", str(sequence), "--columns", str(sequence),
                "--inner", str(inner), "--tile", str(tile), "--waves-per-block", "1",
                "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"rocWMMA QK process failed: {' '.join(command)}\n"
                    f"{completed.stdout}\n{completed.stderr}")
            record = last_json(completed.stdout)
            required = (
                "rocwmma_event_ms_p50", "scalar_event_ms_p50",
                "hipblaslt_event_ms_p50", "rocwmma_max_error",
                "rocwmma_rms_error", "hipblaslt_max_error",
                "hipblaslt_rms_error", "complete_output_elements",
            )
            if record.get("status") != "pass" or not record.get("accuracy_passed") or \
                    any(field not in record for field in required) or \
                    int(record.get("rows", -1)) != sequence or \
                    int(record.get("columns", -1)) != sequence or \
                    int(record.get("inner", -1)) != inner or \
                    list(record.get("tile", []))[:2] != [tile, tile] or \
                    int(record.get("waves_per_block", -1)) != 1:
                raise RuntimeError("rocWMMA QK benchmark violated its record contract")
            numeric = [float(record[field]) for field in required[:-1]]
            if any(not math.isfinite(value) for value in numeric):
                raise RuntimeError("rocWMMA QK benchmark emitted non-finite evidence")
            record.update({
                "record_type": "rocwmma_qk_measurement",
                "process_run": process_run,
                "case_order_index": case_index,
                "case_order": [f"t{s}-d{d}" for s, d in cases],
            })
            records.append(record)

    comparisons = []
    for sequence, inner in base_cases:
        selected = [row for row in records if int(row["rows"]) == sequence and
                    int(row["inner"]) == inner]
        rocwmma_ms = median(selected, "rocwmma_event_ms_p50")
        scalar_ms = median(selected, "scalar_event_ms_p50")
        blas_ms = median(selected, "hipblaslt_event_ms_p50")
        comparisons.append({
            "sequence": sequence,
            "inner": inner,
            "tile": 16 if sequence < 32 else 32,
            "runs": len(selected),
            "complete_output_elements": sequence * sequence,
            "rocwmma_event_ms_p50": rocwmma_ms,
            "scalar_event_ms_p50": scalar_ms,
            "hipblaslt_event_ms_p50": blas_ms,
            "rocwmma_over_scalar": scalar_ms / rocwmma_ms,
            "rocwmma_over_hipblaslt": blas_ms / rocwmma_ms,
            "maximum_rocwmma_error": max(float(row["rocwmma_max_error"])
                                           for row in selected),
            "maximum_rocwmma_rms_error": max(float(row["rocwmma_rms_error"])
                                               for row in selected),
            "maximum_hipblaslt_error": max(float(row["hipblaslt_max_error"])
                                             for row in selected),
        })

    correctness_gate = all(
        row["maximum_rocwmma_error"] <= 2.0e-3 and
        row["maximum_rocwmma_rms_error"] <= 3.0e-4 and
        row["maximum_hipblaslt_error"] <= 2.0e-3
        for row in comparisons)
    long_rows = [row for row in comparisons if row["sequence"] >= 512]
    long_scalar_gate = bool(long_rows) and all(
        row["rocwmma_over_scalar"] >= args.minimum_long_scalar_speedup
        for row in long_rows)
    t512_rows = [row for row in comparisons if row["sequence"] == 512]
    t512_blas_gate = bool(t512_rows) and all(
        row["rocwmma_over_hipblaslt"] >= args.minimum_t512_blas_speedup
        for row in t512_rows)
    long_blas_counterexample = any(
        row["sequence"] >= 2048 and row["rocwmma_over_hipblaslt"] < 1.0
        for row in comparisons)
    online_prototype_admitted = correctness_gate and long_scalar_gate and t512_blas_gate
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness_gate else "fail",
        "record_type": "rocwmma_qk_matrix_summary",
        "architecture": records[0]["architecture"],
        "rocwmma_version": records[0]["rocwmma_version"],
        "processes": len(records),
        "runs_per_case": args.runs,
        "sequences": list(sequences),
        "inners": list(inners),
        "correctness_gate": correctness_gate,
        "long_scalar_gate": long_scalar_gate,
        "t512_blas_gate": t512_blas_gate,
        "long_blas_counterexample": long_blas_counterexample,
        "tail_support": False,
        "online_prototype_admitted": online_prototype_admitted,
        "model_route_accepted": False,
        "comparisons": comparisons,
        "decision": (
            "admit bounded online Attention prototype; keep model route disabled"
            if online_prototype_admitted
            else "reject rocWMMA QK as the online Attention foundation"),
    }
    verification = {
        "schema_version": 1,
        "status": summary["status"],
        "raw_records": len(records),
        "expected_raw_records": len(base_cases) * args.runs,
        "matrix_cases": len(comparisons),
        "expected_matrix_cases": len(base_cases),
        "all_complete_outputs": all(
            int(row["complete_output_elements"]) == int(row["rows"]) * int(row["columns"])
            for row in records),
        "all_accuracy_passed": all(bool(row["accuracy_passed"]) for row in records),
        "selected_layout": "tile32-wave1 except tile16 for T16",
        "correctness_gate": correctness_gate,
        "online_prototype_admitted": online_prototype_admitted,
        "model_route_accepted": False,
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
