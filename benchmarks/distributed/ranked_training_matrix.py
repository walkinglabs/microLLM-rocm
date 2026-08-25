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
AVAILABLE_POLICIES = (
    "per-parameter", "bucket", "persistent-bucket", "bucket-views",
    "overlap-views")
DEFAULT_POLICIES = ("per-parameter", "bucket")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", choices=("tiny", "model-s"), default="tiny")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--compare-binary", type=Path)
    parser.add_argument("--bucket-bytes", type=int, default=4096)
    parser.add_argument("--steady-skip-steps", type=int, default=0)
    parser.add_argument("--policies", nargs="+", choices=AVAILABLE_POLICIES,
                        default=list(DEFAULT_POLICIES))
    args = parser.parse_args()
    if (not args.launcher.is_file() or not args.binary.is_file() or
            args.runs <= 0 or args.steps <= 0 or args.timeout_seconds <= 0 or
            args.bucket_bytes < 4 or args.steady_skip_steps < 0 or
            args.steady_skip_steps >= args.steps or
            args.world_size <= 0 or args.world_size > 8):
        parser.error("ranked training matrix inputs are invalid")
    if args.compare_binary is not None and not args.compare_binary.is_file():
        parser.error("--compare-binary is not a file")
    if args.model == "model-s" and (
            args.compare_binary is None):
        parser.error("Model-S requires --compare-binary")
    if args.context == 0:
        args.context = 4 if args.model == "tiny" else 32
    if ((args.model == "tiny" and args.context != 4) or
            (args.model == "model-s" and not 1 <= args.context <= 512)):
        parser.error("context exceeds the selected model contract")
    if (len(set(args.policies)) != len(args.policies) or
            "per-parameter" not in args.policies or
            "bucket" not in args.policies):
        parser.error("policies must uniquely include per-parameter and bucket")
    if ("bucket-views" in args.policies and
            "persistent-bucket" not in args.policies):
        parser.error("bucket-views comparison requires persistent-bucket")
    if ("overlap-views" in args.policies and
            "bucket-views" not in args.policies):
        parser.error("overlap-views comparison requires bucket-views")
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
        order = (tuple(args.policies) if process_run % 2 else
                 tuple(reversed(args.policies)))
        for policy in order:
            command = [
                sys.executable, str(args.launcher.resolve()),
                "--binary", str(args.binary.resolve()),
                "--output-directory",
                str(output / f"run-{process_run}-{policy}"),
                "--steps", str(args.steps), "--reducer", policy,
                "--world-size", str(args.world_size),
                "--model", args.model,
                "--context", str(args.context),
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
            if policy != "per-parameter":
                expected_collectives = value.get("buckets_per_rank", 0)
                if (expected_collectives <= 0 or
                        expected_collectives % args.steps != 0):
                    raise RuntimeError("ranked bucket count is invalid")
            if (value.get("record_type") != "ranked_training_summary" or
                    value.get("world_size") != args.world_size or
                    value.get("steps") != args.steps or
                    value.get("model") != args.model or
                    value.get("context") != args.context or
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
            step_fields = (
                "maximum_rank_step_training_ms",
                "maximum_rank_step_forward_backward_ms",
                "maximum_rank_step_reducer_ms",
                "maximum_rank_step_optimizer_ms",
                "maximum_rank_step_collectives",
                "maximum_rank_step_buckets",
                "maximum_rank_step_pack_copies",
                "maximum_rank_step_unpack_copies",
                "maximum_rank_step_gradient_views",
                "maximum_rank_step_reducer_allocation_calls",
                "maximum_rank_step_reducer_backend_allocation_calls",
                "maximum_rank_step_reducer_deallocation_calls",
                "maximum_rank_step_reducer_total_allocated_bytes",
                "maximum_rank_step_plan_reused",
                "maximum_rank_step_reducer_current_bytes_before",
                "maximum_rank_step_reducer_current_bytes_after",
                "maximum_rank_step_reducer_peak_bytes_after",
                "maximum_rank_step_overlap_enabled",
                "maximum_rank_step_overlapped_buckets",
            )
            if any(not isinstance(value.get(field), list) or
                   len(value[field]) != args.steps for field in step_fields):
                raise RuntimeError("ranked per-step result contract changed")
            expected_step_collectives = (
                parameter_tensors if policy == "per-parameter" else
                expected_collectives // args.steps)
            if any(count != expected_step_collectives for count in
                   value["maximum_rank_step_collectives"]):
                raise RuntimeError("ranked per-step collective count changed")
            expected_reuse = ([0] + [1] * (args.steps - 1)
                              if policy in
                              ("persistent-bucket", "bucket-views",
                               "overlap-views") else
                              [0] * args.steps)
            if (value["maximum_rank_step_plan_reused"] != expected_reuse or
                    value.get("persistent_storage") !=
                    (policy in ("persistent-bucket", "bucket-views",
                                "overlap-views"))):
                raise RuntimeError("ranked persistent plan state changed")
            value["process_run"] = process_run
            raw.append(value)
    failure_command = [
        sys.executable, str(args.launcher.resolve()),
        "--binary", str(args.binary.resolve()),
        "--output-directory", str(output / "peer-failure"),
        "--steps", "1", "--timeout-seconds", "5",
        "--world-size", str(args.world_size),
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
    for policy in args.policies:
        rows = [row for row in raw if row["reducer"] == policy]
        cold_reducer = [row["maximum_rank_step_reducer_ms"][0]
                        for row in rows]
        steady_reducer = [value for row in rows
                          for value in row["maximum_rank_step_reducer_ms"]
                          [args.steady_skip_steps:]]
        steady_training = [value for row in rows
                           for value in row["maximum_rank_step_training_ms"]
                           [args.steady_skip_steps:]]
        steady_forward_backward = [
            value for row in rows
            for value in row["maximum_rank_step_forward_backward_ms"]
            [args.steady_skip_steps:]]
        steady_optimizer = [value for row in rows
                            for value in row["maximum_rank_step_optimizer_ms"]
                            [args.steady_skip_steps:]]
        steady_backend_allocations = [
            value for row in rows
            for value in row[
                "maximum_rank_step_reducer_backend_allocation_calls"]
            [args.steady_skip_steps:]]
        steady_allocation_calls = [
            value for row in rows
            for value in row["maximum_rank_step_reducer_allocation_calls"]
            [args.steady_skip_steps:]]
        steady_allocated_bytes = [
            value for row in rows
            for value in row["maximum_rank_step_reducer_total_allocated_bytes"]
            [args.steady_skip_steps:]]
        steady_pack_copies = [
            value for row in rows
            for value in row["maximum_rank_step_pack_copies"]
            [args.steady_skip_steps:]]
        steady_unpack_copies = [
            value for row in rows
            for value in row["maximum_rank_step_unpack_copies"]
            [args.steady_skip_steps:]]
        steady_deallocation_calls = [
            value for row in rows
            for value in row["maximum_rank_step_reducer_deallocation_calls"]
            [args.steady_skip_steps:]]
        reducer_mean = statistics.mean(steady_reducer)
        policies[policy] = {
            "runs": len(rows),
            "collectives_per_rank": rows[0]["collectives_per_rank"],
            "buckets_per_rank": rows[0]["buckets_per_rank"],
            "persistent_storage": rows[0]["persistent_storage"],
            "plan_reuses_per_rank": rows[0]["plan_reuses_per_rank"],
            "plan_capacity_elements_per_rank":
                rows[0]["plan_capacity_elements_per_rank"],
            "plan_capacity_bytes_per_rank":
                rows[0]["plan_capacity_bytes_per_rank"],
            "gradient_views_per_rank": rows[0]["gradient_views_per_rank"],
            "overlap_steps_per_rank": rows[0]["overlap_steps_per_rank"],
            "overlapped_buckets_per_rank":
                rows[0]["overlapped_buckets_per_rank"],
            "maximum_engine_current_bytes": max(
                row["maximum_engine_current_bytes"] for row in rows),
            "maximum_engine_peak_bytes": max(
                row["maximum_engine_peak_bytes"] for row in rows),
            "maximum_engine_cached_bytes": max(
                row["maximum_engine_cached_bytes"] for row in rows),
            "maximum_engine_reserved_bytes": max(
                row["maximum_engine_reserved_bytes"] for row in rows),
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
            "cold_reducer_samples": cold_reducer,
            "median_cold_maximum_rank_reducer_ms": statistics.median(
                cold_reducer),
            "steady_step_samples": len(steady_reducer),
            "median_steady_maximum_rank_reducer_ms": statistics.median(
                steady_reducer),
            "minimum_steady_maximum_rank_reducer_ms": min(steady_reducer),
            "maximum_steady_maximum_rank_reducer_ms": max(steady_reducer),
            "steady_maximum_rank_reducer_cv": (
                statistics.pstdev(steady_reducer) / reducer_mean),
            "median_steady_maximum_rank_training_ms": statistics.median(
                steady_training),
            "median_steady_maximum_rank_forward_backward_ms": statistics.median(
                steady_forward_backward),
            "median_steady_maximum_rank_optimizer_ms": statistics.median(
                steady_optimizer),
            "median_steady_reducer_allocation_calls": statistics.median(
                steady_allocation_calls),
            "median_steady_reducer_backend_allocation_calls": statistics.median(
                steady_backend_allocations),
            "maximum_steady_reducer_backend_allocation_calls": max(
                steady_backend_allocations),
            "median_steady_reducer_total_allocated_bytes": statistics.median(
                steady_allocated_bytes),
            "median_steady_pack_copies": statistics.median(
                steady_pack_copies),
            "median_steady_unpack_copies": statistics.median(
                steady_unpack_copies),
            "median_steady_reducer_deallocation_calls": statistics.median(
                steady_deallocation_calls),
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
        "context": args.context,
        "selected_policies": args.policies,
        "runs_per_policy": args.runs,
        "policy_runs": len(raw),
        "rank_processes": len(raw) * args.world_size,
        "world_size": args.world_size,
        "steps_per_rank": args.steps,
        "steady_skip_steps": args.steady_skip_steps,
        "steady_steps_per_run": args.steps - args.steady_skip_steps,
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
        "bucket_cold_reducer_speedup": (
            policies["per-parameter"]["median_cold_maximum_rank_reducer_ms"] /
            policies["bucket"]["median_cold_maximum_rank_reducer_ms"]),
        "bucket_steady_reducer_speedup": (
            policies["per-parameter"]["median_steady_maximum_rank_reducer_ms"] /
            policies["bucket"]["median_steady_maximum_rank_reducer_ms"]),
        "bucket_steady_training_speedup": (
            policies["per-parameter"]["median_steady_maximum_rank_training_ms"] /
            policies["bucket"]["median_steady_maximum_rank_training_ms"]),
        "peer_failure_detected": True,
        "peer_processes_terminated": failure["peer_processes_terminated"],
        "failure_returncodes": failure["returncodes"],
        "decision": ("measure ranked gradient-ready overlap"
                     if "overlap-views" in args.policies else
                     "measure ranked gradient bucket views"
                     if "bucket-views" in args.policies else
                     "measure persistent ranked Model-S buckets"
                     if "persistent-bucket" in args.policies else
                     "profile ranked Model-S cold and steady reducer"
                     if args.model == "model-s" and args.steady_skip_steps > 0 else
                     "admit measured ranked Model-S bucket baseline"
                     if args.model == "model-s" else
                     "admit one-process-per-GPU ready-bucket migration"),
    }
    if "persistent-bucket" in policies:
        persistent = policies["persistent-bucket"]
        summary.update({
            "persistent_steady_reducer_speedup_vs_per_parameter": (
                policies["per-parameter"]
                ["median_steady_maximum_rank_reducer_ms"] /
                persistent["median_steady_maximum_rank_reducer_ms"]),
            "persistent_steady_reducer_speedup_vs_transient": (
                policies["bucket"]["median_steady_maximum_rank_reducer_ms"] /
                persistent["median_steady_maximum_rank_reducer_ms"]),
            "persistent_steady_training_speedup_vs_per_parameter": (
                policies["per-parameter"]
                ["median_steady_maximum_rank_training_ms"] /
                persistent["median_steady_maximum_rank_training_ms"]),
            "persistent_steady_training_speedup_vs_transient": (
                policies["bucket"]["median_steady_maximum_rank_training_ms"] /
                persistent["median_steady_maximum_rank_training_ms"]),
            "persistent_maximum_steady_backend_allocation_calls":
                persistent["maximum_steady_reducer_backend_allocation_calls"],
            "persistent_plan_capacity_bytes_per_rank":
                persistent["plan_capacity_bytes_per_rank"],
            "persistent_current_bytes_added_vs_per_parameter":
                persistent["maximum_engine_current_bytes"] -
                policies["per-parameter"]["maximum_engine_current_bytes"],
            "persistent_current_bytes_added_vs_transient":
                persistent["maximum_engine_current_bytes"] -
                policies["bucket"]["maximum_engine_current_bytes"],
            "persistent_peak_bytes_added_vs_per_parameter":
                persistent["maximum_engine_peak_bytes"] -
                policies["per-parameter"]["maximum_engine_peak_bytes"],
            "persistent_peak_bytes_added_vs_transient":
                persistent["maximum_engine_peak_bytes"] -
                policies["bucket"]["maximum_engine_peak_bytes"],
        })
    if "bucket-views" in policies:
        views = policies["bucket-views"]
        persistent = policies["persistent-bucket"]
        summary.update({
            "views_steady_reducer_speedup_vs_per_parameter": (
                policies["per-parameter"]
                ["median_steady_maximum_rank_reducer_ms"] /
                views["median_steady_maximum_rank_reducer_ms"]),
            "views_steady_reducer_speedup_vs_persistent_copy": (
                persistent["median_steady_maximum_rank_reducer_ms"] /
                views["median_steady_maximum_rank_reducer_ms"]),
            "views_steady_reducer_speedup_vs_transient": (
                policies["bucket"]["median_steady_maximum_rank_reducer_ms"] /
                views["median_steady_maximum_rank_reducer_ms"]),
            "views_steady_training_speedup_vs_per_parameter": (
                policies["per-parameter"]
                ["median_steady_maximum_rank_training_ms"] /
                views["median_steady_maximum_rank_training_ms"]),
            "views_steady_training_speedup_vs_persistent_copy": (
                persistent["median_steady_maximum_rank_training_ms"] /
                views["median_steady_maximum_rank_training_ms"]),
            "views_steady_training_speedup_vs_transient": (
                policies["bucket"]["median_steady_maximum_rank_training_ms"] /
                views["median_steady_maximum_rank_training_ms"]),
            "views_maximum_steady_backend_allocation_calls":
                views["maximum_steady_reducer_backend_allocation_calls"],
            "views_plan_capacity_bytes_per_rank":
                views["plan_capacity_bytes_per_rank"],
            "views_current_bytes_added_vs_per_parameter":
                views["maximum_engine_current_bytes"] -
                policies["per-parameter"]["maximum_engine_current_bytes"],
            "views_current_bytes_added_vs_persistent_copy":
                views["maximum_engine_current_bytes"] -
                persistent["maximum_engine_current_bytes"],
            "views_peak_bytes_added_vs_per_parameter":
                views["maximum_engine_peak_bytes"] -
                policies["per-parameter"]["maximum_engine_peak_bytes"],
            "views_peak_bytes_added_vs_persistent_copy":
                views["maximum_engine_peak_bytes"] -
                persistent["maximum_engine_peak_bytes"],
        })
    if "overlap-views" in policies:
        overlap = policies["overlap-views"]
        views = policies["bucket-views"]
        summary.update({
            "overlap_steady_finish_speedup_vs_synchronous_views": (
                views["median_steady_maximum_rank_reducer_ms"] /
                overlap["median_steady_maximum_rank_reducer_ms"]),
            "overlap_steady_training_speedup_vs_synchronous_views": (
                views["median_steady_maximum_rank_training_ms"] /
                overlap["median_steady_maximum_rank_training_ms"]),
            "overlap_steady_training_speedup_vs_per_parameter": (
                policies["per-parameter"]
                ["median_steady_maximum_rank_training_ms"] /
                overlap["median_steady_maximum_rank_training_ms"]),
            "overlap_steady_training_speedup_vs_transient": (
                policies["bucket"]["median_steady_maximum_rank_training_ms"] /
                overlap["median_steady_maximum_rank_training_ms"]),
            "overlap_maximum_steady_backend_allocation_calls":
                overlap["maximum_steady_reducer_backend_allocation_calls"],
            "overlap_steps_per_rank": overlap["overlap_steps_per_rank"],
            "overlapped_buckets_per_rank":
                overlap["overlapped_buckets_per_rank"],
            "overlap_current_bytes_added_vs_synchronous_views":
                overlap["maximum_engine_current_bytes"] -
                views["maximum_engine_current_bytes"],
            "overlap_peak_bytes_added_vs_synchronous_views":
                overlap["maximum_engine_peak_bytes"] -
                views["maximum_engine_peak_bytes"],
        })
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
