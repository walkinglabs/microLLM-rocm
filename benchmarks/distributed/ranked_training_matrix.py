#!/usr/bin/env python3
"""Repeatable one-process-per-GPU bootstrap evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("per-parameter", "bucket")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", choices=("tiny", "model-s"), default="tiny")
    parser.add_argument("--compare-binary", type=Path)
    parser.add_argument("--bucket-bytes", type=int, default=4096)
    args = parser.parse_args()
    if (not args.launcher.is_file() or not args.binary.is_file() or
            args.runs <= 0 or args.steps <= 0 or args.timeout_seconds <= 0 or
            args.bucket_bytes < 4):
        parser.error("ranked training matrix inputs are invalid")
    if args.compare_binary is not None and not args.compare_binary.is_file():
        parser.error("--compare-binary is not a file")
    if args.model == "model-s" and (
            args.compare_binary is None):
        parser.error("Model-S requires --compare-binary")
    return args


def record(stdout: str, name: str) -> dict:
    lines = [line for line in stdout.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"{name} emitted an unexpected record count")
    value = json.loads(lines[0])
    if value.get("schema_version") != 1 or value.get("status") != "pass":
        raise RuntimeError(f"{name} failed")
    return value


def main() -> int:
    args = options()
    output = args.output_directory.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError("output directory is not empty; pass --overwrite")
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = []
    for process_run in range(1, args.runs + 1):
        order = POLICIES if process_run % 2 else tuple(reversed(POLICIES))
        for policy in order:
            command = [
                sys.executable, str(args.launcher.resolve()),
                "--binary", str(args.binary.resolve()),
                "--output-directory",
                str(output / f"run-{process_run}-{policy}"),
                "--steps", str(args.steps), "--reducer", policy,
                "--model", args.model,
                "--bucket-bytes", str(args.bucket_bytes),
                "--timeout-seconds", str(args.timeout_seconds), "--overwrite",
            ]
            if args.compare_binary is not None:
                command.extend(["--compare-binary",
                                str(args.compare_binary.resolve())])
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False,
                timeout=args.timeout_seconds + 10)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            value = record(completed.stdout, f"run-{process_run}-{policy}")
            parameter_tensors = 57 if args.model == "model-s" else 12
            parameter_values = 15586176 if args.model == "model-s" else 728
            reference_maximum = 1.0e-2 if args.model == "model-s" else 2.0e-5
            reference_rms = 1.0e-5 if args.model == "model-s" else 2.0e-5
            expected_collectives = args.steps * parameter_tensors
            if policy == "bucket":
                expected_collectives = value.get("buckets_per_rank", 0)
                if (expected_collectives <= 0 or
                        expected_collectives % args.steps != 0):
                    raise RuntimeError("ranked bucket count is invalid")
            if (value.get("record_type") != "ranked_training_summary" or
                    value.get("world_size") != 2 or
                    value.get("steps") != args.steps or
                    value.get("model") != args.model or
                    value.get("reducer") != policy or
                    value.get("collectives_per_rank") != expected_collectives or
                    value.get("parameter_tensors") != parameter_tensors or
                    value.get("parameter_values") != parameter_values or
                    value.get("maximum_rank_difference") != 0.0 or
                    value.get("rank_rms_difference") != 0.0 or
                    value.get("maximum_reference_difference", 1.0) >
                    reference_maximum or
                    value.get("reference_rms_difference", 1.0) > reference_rms or
                    value.get("maximum_mean_loss_difference", 1.0) > 1.0e-4 or
                    value.get("maximum_rank_training_ms", 0.0) <= 0.0 or
                    value.get("maximum_rank_reducer_ms", -1.0) < 0.0 or
                    value.get("parameter_files_retained") is not False or
                    value.get("peer_processes_terminated") != 0):
                raise RuntimeError("ranked training result contract changed")
            value["process_run"] = process_run
            raw.append(value)
    failure_command = [
        sys.executable, str(args.launcher.resolve()),
        "--binary", str(args.binary.resolve()),
        "--output-directory", str(output / "peer-failure"),
        "--steps", "1", "--timeout-seconds", "5",
        "--reducer", "bucket", "--model", "tiny",
        "--failure-mode", "peer-failure", "--overwrite",
    ]
    failure_completed = subprocess.run(
        failure_command, cwd=ROOT, text=True, capture_output=True, check=False,
        timeout=15)
    if failure_completed.returncode != 0:
        raise RuntimeError(failure_completed.stdout + failure_completed.stderr)
    failure = record(failure_completed.stdout, "peer-failure")
    if (failure.get("record_type") != "ranked_peer_failure_summary" or
            failure.get("failure_detected") is not True or
            failure.get("peer_processes_terminated", 0) < 1):
        raise RuntimeError("ranked peer-failure contract changed")
    policies = {}
    for policy in POLICIES:
        rows = [row for row in raw if row["reducer"] == policy]
        policies[policy] = {
            "runs": len(rows),
            "collectives_per_rank": rows[0]["collectives_per_rank"],
            "buckets_per_rank": rows[0]["buckets_per_rank"],
            "median_rank_group_ms": statistics.median(
                row["rank_group_ms"] for row in rows),
            "median_maximum_rank_training_ms": statistics.median(
                row["maximum_rank_training_ms"] for row in rows),
            "median_maximum_rank_forward_backward_ms": statistics.median(
                row["maximum_rank_forward_backward_ms"] for row in rows),
            "median_maximum_rank_reducer_ms": statistics.median(
                row["maximum_rank_reducer_ms"] for row in rows),
            "median_maximum_rank_optimizer_ms": statistics.median(
                row["maximum_rank_optimizer_ms"] for row in rows),
            "maximum_rank_difference": max(
                row["maximum_rank_difference"] for row in rows),
            "maximum_reference_difference": max(
                row["maximum_reference_difference"] for row in rows),
            "maximum_reference_rms_difference": max(
                row["reference_rms_difference"] for row in rows),
            "maximum_mean_loss_difference": max(
                row["maximum_mean_loss_difference"] for row in rows),
        }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "ranked_training_matrix_summary",
        "model": args.model,
        "runs_per_policy": args.runs,
        "policy_runs": len(raw),
        "rank_processes": len(raw) * 2,
        "steps_per_rank": args.steps,
        "parameter_tensors": 57 if args.model == "model-s" else 12,
        "parameter_values": 15586176 if args.model == "model-s" else 728,
        "maximum_rank_difference": max(
            row["maximum_rank_difference"] for row in raw),
        "maximum_reference_difference": max(
            row["maximum_reference_difference"] for row in raw),
        "maximum_reference_rms_difference": max(
            row["reference_rms_difference"] for row in raw),
        "maximum_mean_loss_difference": max(
            row["maximum_mean_loss_difference"] for row in raw),
        "policies": policies,
        "collective_reduction": (
            policies["per-parameter"]["collectives_per_rank"] /
            policies["bucket"]["collectives_per_rank"]),
        "bucket_wall_speedup": (
            policies["per-parameter"]["median_rank_group_ms"] /
            policies["bucket"]["median_rank_group_ms"]),
        "bucket_training_speedup": (
            policies["per-parameter"]["median_maximum_rank_training_ms"] /
            policies["bucket"]["median_maximum_rank_training_ms"]),
        "bucket_reducer_speedup": (
            policies["per-parameter"]["median_maximum_rank_reducer_ms"] /
            policies["bucket"]["median_maximum_rank_reducer_ms"]),
        "peer_failure_detected": True,
        "peer_processes_terminated": failure["peer_processes_terminated"],
        "failure_returncodes": failure["returncodes"],
        "decision": ("admit measured ranked Model-S bucket baseline"
                     if args.model == "model-s" else
                     "admit one-process-per-GPU ready-bucket migration"),
    }
    (output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw),
        encoding="utf-8")
    (output / "failure.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError,
            subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"ranked_training_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
