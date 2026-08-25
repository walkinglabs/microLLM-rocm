#!/usr/bin/env python3
"""Measure the context-scale boundary of ranked gradient-ready overlap."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYNCHRONOUS_POLICY = "bucket-views"


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--compare-binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--contexts", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--rank-batch-rows", default="1,1")
    parser.add_argument("--input-weighting",
                        choices=("equal-only", "token-weighted"),
                        default="equal-only")
    parser.add_argument("--overlap-policy",
                        choices=("overlap-views", "bucket-weighted-overlap"),
                        default="overlap-views")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--steady-skip-steps", type=int, default=1)
    parser.add_argument("--bucket-bytes", type=int, default=26214400)
    parser.add_argument("--mean-loss-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (not args.launcher.is_file() or not args.binary.is_file() or
            not args.compare_binary.is_file() or args.runs <= 0 or
            args.steps <= 1 or args.steady_skip_steps < 1 or
            args.steady_skip_steps >= args.steps or args.bucket_bytes < 4 or
            args.timeout_seconds <= 0 or args.mean_loss_tolerance <= 0.0 or
            not args.contexts or
            len(set(args.contexts)) != len(args.contexts) or
            any(context < 1 or context > 512 for context in args.contexts)):
        parser.error("ranked overlap context matrix inputs are invalid")
    try:
        args.rank_batch_rows = [
            int(value) for value in args.rank_batch_rows.split(",")]
    except ValueError:
        parser.error("rank batch rows are invalid")
    if (len(args.rank_batch_rows) != 2 or
            any(value <= 0 for value in args.rank_batch_rows)):
        parser.error("rank batch rows must contain two positive values")
    if (args.input_weighting == "equal-only" and
            len(set(args.rank_batch_rows)) != 1):
        parser.error("uneven rank rows require token-weighted input")
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

    policies = (SYNCHRONOUS_POLICY, args.overlap_policy)
    cases = [(context, policy) for context in args.contexts
             for policy in policies]
    raw = []
    for process_run in range(1, args.runs + 1):
        order = cases if process_run % 2 else list(reversed(cases))
        for context, policy in order:
            command = [
                sys.executable, str(args.launcher.resolve()),
                "--binary", str(args.binary.resolve()),
                "--compare-binary", str(args.compare_binary.resolve()),
                "--output-directory",
                str(output / f"run-{process_run}-t{context}-{policy}"),
                "--model", "model-s", "--context", str(context),
                "--steps", str(args.steps), "--reducer", policy,
                "--rank-batch-rows",
                ",".join(str(value) for value in args.rank_batch_rows),
                "--input-weighting", args.input_weighting,
                "--mean-loss-tolerance", str(args.mean_loss_tolerance),
                "--retain-consensus-parameter-file",
                "--bucket-bytes", str(args.bucket_bytes),
                "--timeout-seconds", str(args.timeout_seconds), "--overwrite",
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False,
                timeout=args.timeout_seconds + 10)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            value = record(completed.stdout, f"T{context}-{policy}")
            expected_overlap = policy != SYNCHRONOUS_POLICY
            expected_scales = (
                57 if args.input_weighting == "token-weighted" and
                len(set(args.rank_batch_rows)) > 1 and
                policy != "bucket-weighted-overlap" else 0)
            expected_bucket_scales = (
                3 if args.input_weighting == "token-weighted" and
                len(set(args.rank_batch_rows)) > 1 and
                policy == "bucket-weighted-overlap" else 0)
            if (value.get("record_type") != "ranked_training_summary" or
                    value.get("model") != "model-s" or
                    value.get("context") != context or
                    value.get("rank_batch_rows") != args.rank_batch_rows or
                    value.get("input_weighting") != args.input_weighting or
                    value.get("reducer") != policy or
                    value.get("steps") != args.steps or
                    value.get("parameter_tensors") != 57 or
                    value.get("parameter_values") != 15586176 or
                    value.get("maximum_rank_difference") != 0.0 or
                    value.get("rank_rms_difference") != 0.0 or
                    value.get("maximum_reference_difference", 1.0) > 1.0e-2 or
                    value.get("reference_rms_difference", 1.0) > 1.0e-5 or
                    value.get("maximum_mean_loss_difference", 1.0) >
                    args.mean_loss_tolerance or
                    value.get("mean_loss_tolerance") !=
                    args.mean_loss_tolerance or
                    value.get("maximum_rank_step_collectives") != [3] * args.steps or
                    value.get("maximum_rank_step_unpack_copies") != [0] * args.steps or
                    value.get("maximum_rank_step_reducer_backend_allocation_calls") !=
                    [3] + [0] * (args.steps - 1) or
                    value.get("maximum_rank_step_overlap_enabled") !=
                    ([0] + [1] * (args.steps - 1) if expected_overlap else
                     [0] * args.steps) or
                    value.get("maximum_rank_step_overlapped_buckets") !=
                    ([0] + [3] * (args.steps - 1) if expected_overlap else
                     [0] * args.steps) or
                    value.get("maximum_rank_step_weighted_gradient_scales") !=
                    [expected_scales] * args.steps or
                    value.get("maximum_weighted_gradient_scales_per_rank") !=
                    expected_scales * args.steps or
                    value.get("maximum_rank_step_weighted_bucket_scales") !=
                    [expected_bucket_scales] * args.steps or
                    value.get("maximum_weighted_bucket_scales_per_rank") !=
                    expected_bucket_scales * args.steps or
                    value.get("maximum_engine_current_bytes", 0) <= 0 or
                    value.get("maximum_engine_peak_bytes", 0) <
                    value.get("maximum_engine_current_bytes", 0) or
                    value.get("parameter_files_retained") is not True or
                    not Path(value.get(
                        "consensus_parameter_file", "")).is_file()):
                raise RuntimeError("ranked overlap context result contract changed")
            value["process_run"] = process_run
            raw.append(value)

    retained_files = [Path(row["consensus_parameter_file"]) for row in raw]
    policy_comparisons = []
    try:
        for process_run in range(1, args.runs + 1):
            for context in args.contexts:
                rows = {row["reducer"]: row for row in raw
                        if row["process_run"] == process_run and
                        row["context"] == context}
                command = [
                    str(args.compare_binary.resolve()),
                    rows[SYNCHRONOUS_POLICY]["consensus_parameter_file"],
                    rows[args.overlap_policy]["consensus_parameter_file"],
                ]
                completed = subprocess.run(
                    command, cwd=ROOT, text=True, capture_output=True,
                    check=False, timeout=args.timeout_seconds)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stdout + completed.stderr)
                comparison = record(
                    completed.stdout,
                    f"T{context}-run{process_run}-policy-comparison")
                if (comparison.get("record_type") !=
                        "safetensors_complete_comparison" or
                        comparison.get("tensor_count") != 57 or
                        comparison.get("compared_elements") != 15586176 or
                        comparison.get("maximum_absolute_difference") != 0.0 or
                        comparison.get("rms_difference") != 0.0):
                    raise RuntimeError(
                        "synchronous and overlap parameters diverged")
                comparison.update({
                    "context": context,
                    "process_run": process_run,
                })
                policy_comparisons.append(comparison)
    finally:
        for path in retained_files:
            path.unlink(missing_ok=True)

    failure_command = [
        sys.executable, str(args.launcher.resolve()),
        "--binary", str(args.binary.resolve()),
        "--output-directory", str(output / "peer-failure"),
        "--model", "tiny", "--context", "4", "--steps", "1",
        "--timeout-seconds", "5", "--reducer", "bucket",
        "--failure-mode", "peer-failure", "--overwrite",
    ]
    failure_completed = subprocess.run(
        failure_command, cwd=ROOT, text=True, capture_output=True, check=False,
        timeout=15)
    if failure_completed.returncode != 0:
        raise RuntimeError(failure_completed.stdout + failure_completed.stderr)
    failure = record(failure_completed.stdout, "peer-failure")
    if (failure.get("failure_detected") is not True or
            failure.get("peer_processes_terminated", 0) < 1):
        raise RuntimeError("ranked context peer-failure contract changed")

    contexts = {}
    for context in args.contexts:
        policy_summary = {}
        for policy in policies:
            rows = [row for row in raw if row["context"] == context and
                    row["reducer"] == policy]
            steady_total = [value for row in rows
                            for value in row["maximum_rank_step_training_ms"]
                            [args.steady_skip_steps:]]
            steady_forward = [
                value for row in rows
                for value in row["maximum_rank_step_forward_backward_ms"]
                [args.steady_skip_steps:]]
            steady_finish = [value for row in rows
                             for value in row["maximum_rank_step_reducer_ms"]
                             [args.steady_skip_steps:]]
            policy_summary[policy] = {
                "runs": len(rows),
                "steady_samples": len(steady_total),
                "median_steady_training_ms": statistics.median(steady_total),
                "steady_training_cv": (
                    statistics.pstdev(steady_total) /
                    statistics.mean(steady_total)),
                "median_steady_forward_backward_ms": statistics.median(
                    steady_forward),
                "median_steady_finish_ms": statistics.median(steady_finish),
                "maximum_engine_current_bytes": max(
                    row["maximum_engine_current_bytes"] for row in rows),
                "maximum_engine_peak_bytes": max(
                    row["maximum_engine_peak_bytes"] for row in rows),
                "maximum_reference_difference": max(
                    row["maximum_reference_difference"] for row in rows),
                "maximum_reference_rms_difference": max(
                    row["reference_rms_difference"] for row in rows),
                "maximum_mean_loss_difference": max(
                    row["maximum_mean_loss_difference"] for row in rows),
            }
        synchronous = policy_summary[SYNCHRONOUS_POLICY]
        overlap = policy_summary[args.overlap_policy]
        contexts[str(context)] = {
            "policies": policy_summary,
            "finish_speedup": (
                synchronous["median_steady_finish_ms"] /
                overlap["median_steady_finish_ms"]),
            "training_speedup": (
                synchronous["median_steady_training_ms"] /
                overlap["median_steady_training_ms"]),
            "forward_backward_added_ms": (
                overlap["median_steady_forward_backward_ms"] -
                synchronous["median_steady_forward_backward_ms"]),
            "current_bytes_added": (
                overlap["maximum_engine_current_bytes"] -
                synchronous["maximum_engine_current_bytes"]),
            "peak_bytes_added": (
                overlap["maximum_engine_peak_bytes"] -
                synchronous["maximum_engine_peak_bytes"]),
        }
    gate_contexts = ([context for context in args.contexts
                      if context > min(args.contexts)]
                     if len(args.contexts) > 1 else args.contexts)
    gated_speedups = [contexts[str(context)]["training_speedup"]
                       for context in gate_contexts]
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "ranked_overlap_context_summary",
        "contexts": args.contexts,
        "rank_batch_rows": args.rank_batch_rows,
        "input_weighting": args.input_weighting,
        "overlap_policy": args.overlap_policy,
        "runs_per_policy_context": args.runs,
        "policy_context_runs": len(raw),
        "rank_processes": len(raw) * 2,
        "steps_per_rank": args.steps,
        "steady_skip_steps": args.steady_skip_steps,
        "steady_steps_per_run": args.steps - args.steady_skip_steps,
        "bucket_bytes": args.bucket_bytes,
        "mean_loss_tolerance": args.mean_loss_tolerance,
        "policy_parameter_comparisons": len(policy_comparisons),
        "maximum_policy_parameter_difference": max(
            comparison["maximum_absolute_difference"]
            for comparison in policy_comparisons),
        "policy_parameter_rms_difference": max(
            comparison["rms_difference"]
            for comparison in policy_comparisons),
        "temporary_parameter_files_retained": any(
            path.exists() for path in retained_files),
        "results": contexts,
        "minimum_required_speedup": 1.01,
        "gate_contexts": gate_contexts,
        "longer_context_gate_passed": bool(gated_speedups) and all(
            speedup >= 1.01 for speedup in gated_speedups),
        "peer_failure_detected": True,
        "peer_processes_terminated": failure["peer_processes_terminated"],
        "failure_returncodes": failure["returncodes"],
        "decision": ("retain context-selective ranked weighted overlap"
                     if args.input_weighting == "token-weighted" and
                     bool(gated_speedups) and all(
                         speedup >= 1.01 for speedup in gated_speedups)
                     else "close Model-S ranked weighted overlap scale track"
                     if args.input_weighting == "token-weighted"
                     else "retain context-selective ranked overlap"
                     if bool(gated_speedups) and all(
                         speedup >= 1.01 for speedup in gated_speedups)
                     else "close Model-S ranked overlap scale track"),
    }
    (output / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw),
        encoding="utf-8")
    (output / "failure.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output / "policy-comparisons.json").write_text(
        json.dumps(policy_comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError,
            subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"ranked_overlap_context_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
