#!/usr/bin/env python3
"""Run a fresh-process matrix for staged training HIP Graph capture."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


PRECISIONS = ("fp32", "bf16")
STAGES = ("forward", "backward", "optimizer", "full-step")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--maximum-blocks", type=int, default=65536)
    result = parser.parse_args()
    if not result.binary.is_file():
        parser.error(f"binary does not exist: {result.binary}")
    if result.runs < 3 or result.maximum_blocks <= 0:
        parser.error("runs must be at least three and maximum-blocks positive")
    return result


def execute(binary: pathlib.Path, precision: str, stage: str,
            maximum_blocks: int) -> dict:
    completed = subprocess.run([
        str(binary), "--precision", precision, "--stage", stage,
        "--maximum-blocks", str(maximum_blocks),
    ], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("training graph probe did not emit one JSON object") from error
    if record.get("schema_version") != 1 or record.get("status") != "pass" or \
       record.get("precision") != precision or record.get("stage") != stage or \
       record.get("capture_recovery_failed") is not False or \
       int(record.get("capture_status_after_recovery", -1)) != 0:
        raise RuntimeError(f"invalid training graph row: {precision}/{stage}")
    return record


def summarize(records: list[dict], runs: int) -> dict:
    cases = []
    for precision in PRECISIONS:
        for stage in STAGES:
            selected = [row for row in records
                        if row["precision"] == precision and row["stage"] == stage]
            if len(selected) != runs:
                raise RuntimeError(f"incomplete case: {precision}/{stage}")
            support = {bool(row["capture_supported"]) for row in selected}
            nodes = {int(row["captured_nodes"]) for row in selected}
            host_step = {
                bool(row["optimizer_replay_advances_host_step"])
                for row in selected
            }
            if len(support) != 1 or len(nodes) != 1 or len(host_step) != 1:
                raise RuntimeError(f"unstable case: {precision}/{stage}")
            cases.append({
                "precision": precision,
                "stage": stage,
                "runs": runs,
                "capture_supported": support.pop(),
                "captured_nodes": nodes.pop(),
                "optimizer_replay_advances_host_step": host_step.pop(),
                "capture_error": selected[0]["capture_error"],
                "maximum_deferred_blocks": max(
                    int(row["deferred_blocks"]) for row in selected),
                "maximum_deferred_bytes": max(
                    int(row["deferred_bytes"]) for row in selected),
            })

    dynamic = [row for row in cases if row["stage"] != "optimizer"]
    optimizer = [row for row in cases if row["stage"] == "optimizer"]
    gates = {
        "capture_failure_recovery_clean": all(
            not row["capture_recovery_failed"] and
            int(row["capture_status_after_recovery"]) == 0
            for row in records),
        "dynamic_graph_stages_rejected": all(
            not row["capture_supported"] and row["captured_nodes"] == 0 and
            "forbids dynamic Tensor allocation" in row["capture_error"]
            for row in dynamic),
        "optimizer_device_nodes_captured": all(
            row["capture_supported"] and row["captured_nodes"] > 0
            for row in optimizer),
        "optimizer_host_step_not_replayed": all(
            not row["optimizer_replay_advances_host_step"]
            for row in optimizer),
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "experiment": "staged_training_hip_graph_capture",
        "processes": len(records),
        "runs_per_case": runs,
        "cases": cases,
        "gates": gates,
        "decision": (
            "reject complete training graph; require graph-wide stable workspaces "
            "and device-owned optimizer step"
            if all(gates.values()) else
            "staged training graph evidence is incomplete"),
    }


def main() -> int:
    args = options()
    records = []
    base_cases = tuple((precision, stage) for precision in PRECISIONS
                       for stage in STAGES)
    for process_run in range(1, args.runs + 1):
        cases = base_cases if process_run % 2 else tuple(reversed(base_cases))
        for precision, stage in cases:
            row = execute(args.binary, precision, stage, args.maximum_blocks)
            row["process_run"] = process_run
            records.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = summarize(records, args.runs)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"training_graph_capture_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
