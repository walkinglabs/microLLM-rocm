#!/usr/bin/env python3
"""Launch and verify the one-process-per-GPU microLLM bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--failure-mode", choices=("none", "peer-failure"),
                        default="none")
    parser.add_argument("--reducer", choices=("per-parameter", "bucket"),
                        default="per-parameter")
    parser.add_argument("--bucket-bytes", type=int, default=4096)
    parser.add_argument("--model", choices=("tiny", "model-s"), default="tiny")
    parser.add_argument("--compare-binary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (not args.binary.is_file() or args.steps <= 0 or
            args.timeout_seconds <= 0 or args.bucket_bytes < 4):
        parser.error("ranked launcher inputs are invalid")
    if args.compare_binary is not None and not args.compare_binary.is_file():
        parser.error("--compare-binary is not a file")
    if args.model == "model-s" and (
            args.compare_binary is None):
        parser.error("Model-S requires --compare-binary")
    return args


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError("output directory is not empty; pass --overwrite")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def terminate(processes: list[subprocess.Popen[str]]) -> int:
    terminated = 0
    for process in processes:
        if process.poll() is None:
            process.terminate()
            terminated += 1
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and any(
            process.poll() is None for process in processes):
        time.sleep(0.02)
    for process in processes:
        if process.poll() is None:
            process.kill()
    return terminated


def wait_group(processes: list[subprocess.Popen[str]], timeout: float) -> tuple[bool, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = [process.poll() for process in processes]
        if any(status not in (None, 0) for status in statuses):
            return False, terminate(processes)
        if all(status == 0 for status in statuses):
            return True, 0
        time.sleep(0.02)
    return False, terminate(processes)


def load_record(text: str, name: str) -> dict:
    lines = [line for line in text.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"{name} emitted an unexpected record count")
    record = json.loads(lines[0])
    if record.get("schema_version") != 1 or record.get("status") != "pass":
        raise RuntimeError(f"{name} record failed")
    return record


def maximum_difference(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right):
        raise RuntimeError("parameter list count changed")
    maximum = 0.0
    for lhs, rhs in zip(left, right):
        if len(lhs) != len(rhs):
            raise RuntimeError("parameter element count changed")
        maximum = max(maximum, max((abs(a - b) for a, b in zip(lhs, rhs)),
                                   default=0.0))
    return maximum


def rms_difference(left: list[list[float]], right: list[list[float]]) -> float:
    squared = 0.0
    count = 0
    for lhs, rhs in zip(left, right):
        if len(lhs) != len(rhs):
            raise RuntimeError("parameter element count changed")
        for a, b in zip(lhs, rhs):
            squared += (a - b) ** 2
            count += 1
    return (squared / count) ** 0.5 if count else 0.0


def compare_safetensors(binary: Path, baseline: Path, candidate: Path,
                        timeout: float) -> dict:
    completed = subprocess.run(
        [str(binary.resolve()), str(baseline), str(candidate)],
        cwd=ROOT, text=True, capture_output=True, timeout=timeout,
        check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    return load_record(completed.stdout, "safetensors comparison")


def main() -> int:
    args = options()
    output = args.output_directory.resolve()
    prepare_output(output, args.overwrite)
    id_file = output / "communicator.id"
    common = ["--world-size", "2", "--id-file", str(id_file),
              "--steps", str(args.steps), "--seed", "607",
              "--timeout-ms", str(int(args.timeout_seconds * 1000)),
              "--reducer", args.reducer,
              "--bucket-bytes", str(args.bucket_bytes)]
    rank_parameter_files = [output / "rank1.safetensors",
                            output / "rank0.safetensors"]
    reference_parameter_file = output / "reference.safetensors"
    commands = [
        [str(args.binary.resolve()), "--mode", "rank", "--rank", "1",
         "--local-rank", "1", "--model", args.model, *common],
        [str(args.binary.resolve()), "--mode", "rank", "--rank", "0",
         "--local-rank", "0", "--model", args.model, *common],
    ]
    if args.model == "model-s":
        for command, parameter_file in zip(commands, rank_parameter_files):
            command.extend(["--parameter-file", str(parameter_file)])
    if args.failure_mode == "peer-failure":
        commands[0][commands[0].index("--rank") + 1] = "2"
    group_start = time.monotonic()
    processes = [subprocess.Popen(command, cwd=ROOT, text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
                 for command in commands]
    completed, terminated = wait_group(processes, args.timeout_seconds)
    rank_group_ms = (time.monotonic() - group_start) * 1000.0
    outputs = []
    errors = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=2)
        outputs.append(stdout)
        errors.append(stderr)
    id_file.unlink(missing_ok=True)
    for index, text in enumerate(outputs):
        (output / f"rank{index}.stdout").write_text(text, encoding="utf-8")
    for index, text in enumerate(errors):
        (output / f"rank{index}.stderr").write_text(text, encoding="utf-8")

    if args.failure_mode == "peer-failure":
        for path in [*rank_parameter_files, reference_parameter_file]:
            path.unlink(missing_ok=True)
        if completed or terminated < 1 or processes[0].returncode == 0:
            raise RuntimeError("peer failure did not terminate the waiting rank group")
        summary = {
            "schema_version": 1,
            "status": "pass",
            "record_type": "ranked_peer_failure_summary",
            "failure_detected": True,
            "peer_processes_terminated": terminated,
            "rank_group_ms": rank_group_ms,
            "returncodes": [process.returncode for process in processes],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 0

    if not completed:
        raise RuntimeError("ranked training timed out or one rank failed")
    ranks = [load_record(text, f"rank{index}")
             for index, text in enumerate(outputs)]
    reference_command = [str(args.binary.resolve()), "--mode", "reference",
                         "--steps", str(args.steps), "--seed", "607",
                         "--model", args.model]
    if args.model == "model-s":
        reference_command.extend(
            ["--parameter-file", str(reference_parameter_file)])
    reference_start = time.monotonic()
    reference_completed = subprocess.run(
        reference_command, cwd=ROOT, text=True, capture_output=True,
        timeout=args.timeout_seconds, check=False)
    reference_ms = (time.monotonic() - reference_start) * 1000.0
    (output / "reference.stdout").write_text(
        reference_completed.stdout, encoding="utf-8")
    (output / "reference.stderr").write_text(
        reference_completed.stderr, encoding="utf-8")
    if reference_completed.returncode != 0:
        raise RuntimeError("CPU global-batch reference failed")
    reference = load_record(reference_completed.stdout, "reference")
    if (ranks[0].get("rank") != 1 or ranks[1].get("rank") != 0 or
            ranks[0]["parameter_names"] != ranks[1]["parameter_names"] or
            ranks[0]["parameter_names"] != reference["parameter_names"]):
        raise RuntimeError("rank identity or parameter names changed")
    expected_collectives = (
        ranks[0]["buckets"] if args.reducer == "bucket" else
        args.steps * len(reference["parameter_names"]))
    if any(rank.get("reducer") != args.reducer or
           rank.get("collectives") != expected_collectives or
           rank.get("buckets") != ranks[0]["buckets"]
           for rank in ranks):
        raise RuntimeError("rank reducer collective count changed")
    if args.model == "model-s":
        assert args.compare_binary is not None
        rank_comparison = compare_safetensors(
            args.compare_binary, rank_parameter_files[0],
            rank_parameter_files[1], args.timeout_seconds)
        reference_comparisons = [
            compare_safetensors(
                args.compare_binary, reference_parameter_file,
                path, args.timeout_seconds)
            for path in rank_parameter_files]
        rank_difference = rank_comparison["maximum_absolute_difference"]
        rank_rms_difference = rank_comparison["rms_difference"]
        reference_difference = max(
            comparison["maximum_absolute_difference"]
            for comparison in reference_comparisons)
        reference_rms_difference = max(
            comparison["rms_difference"]
            for comparison in reference_comparisons)
        tensor_count = rank_comparison["tensor_count"]
        value_count = rank_comparison["compared_elements"]
        reference_tolerance = 1.0e-2
        reference_rms_tolerance = 1.0e-5
    else:
        rank_difference = maximum_difference(
            ranks[0]["parameters"], ranks[1]["parameters"])
        rank_rms_difference = rms_difference(
            ranks[0]["parameters"], ranks[1]["parameters"])
        reference_difference = max(
            maximum_difference(rank["parameters"], reference["parameters"])
            for rank in ranks)
        reference_rms_difference = max(
            rms_difference(rank["parameters"], reference["parameters"])
            for rank in ranks)
        tensor_count = len(reference["parameters"])
        value_count = sum(len(values) for values in reference["parameters"])
        reference_tolerance = 2.0e-5
        reference_rms_tolerance = 2.0e-5
    timing_fields = ("training_ms", "forward_backward_ms", "reducer_ms",
                     "optimizer_ms")
    if any(not isinstance(rank.get(field), (int, float)) or
           not math.isfinite(rank[field]) or rank[field] < 0.0
           for rank in ranks for field in timing_fields):
        raise RuntimeError("rank phase timing contract changed")
    if any(rank["training_ms"] <= 0.0 or
           rank["forward_backward_ms"] <= 0.0 or
           rank["reducer_ms"] <= 0.0 or
           rank["optimizer_ms"] <= 0.0 or
           rank["training_ms"] + 1.0e-6 <
           rank["forward_backward_ms"] + rank["reducer_ms"] +
           rank["optimizer_ms"]
           for rank in ranks):
        raise RuntimeError("rank phase timings do not form a complete interval")
    loss_difference = max(
        abs(sum(rank["losses"][step] for rank in ranks) / len(ranks) -
            reference["losses"][step])
        for step in range(args.steps))
    if (rank_difference != 0.0 or rank_rms_difference != 0.0 or
            reference_difference > reference_tolerance or
            reference_rms_difference > reference_rms_tolerance or
            loss_difference > 1.0e-4):
        raise RuntimeError("ranked parameters failed the global-batch gate")
    if args.model == "model-s":
        for path in [*rank_parameter_files, reference_parameter_file]:
            path.unlink(missing_ok=True)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "ranked_training_summary",
        "world_size": 2,
        "model": args.model,
        "reducer": args.reducer,
        "bucket_bytes": args.bucket_bytes,
        "steps": args.steps,
        "parameter_tensors": tensor_count,
        "parameter_values": value_count,
        "maximum_rank_difference": rank_difference,
        "rank_rms_difference": rank_rms_difference,
        "maximum_reference_difference": reference_difference,
        "reference_rms_difference": reference_rms_difference,
        "maximum_mean_loss_difference": loss_difference,
        "mean_loss_tolerance": 1.0e-4,
        "reference_max_tolerance": reference_tolerance,
        "reference_rms_tolerance": reference_rms_tolerance,
        "rank_losses": [rank["losses"] for rank in ranks],
        "reference_losses": reference["losses"],
        "rank_training_ms": [rank["training_ms"] for rank in ranks],
        "maximum_rank_training_ms": max(
            rank["training_ms"] for rank in ranks),
        "rank_forward_backward_ms": [
            rank["forward_backward_ms"] for rank in ranks],
        "maximum_rank_forward_backward_ms": max(
            rank["forward_backward_ms"] for rank in ranks),
        "rank_reducer_ms": [rank["reducer_ms"] for rank in ranks],
        "maximum_rank_reducer_ms": max(
            rank["reducer_ms"] for rank in ranks),
        "rank_optimizer_ms": [rank["optimizer_ms"] for rank in ranks],
        "maximum_rank_optimizer_ms": max(
            rank["optimizer_ms"] for rank in ranks),
        "parameter_files_retained": False,
        "peer_processes_terminated": terminated,
        "collectives_per_rank": expected_collectives,
        "buckets_per_rank": ranks[0]["buckets"],
        "pack_copies_per_rank": ranks[0]["pack_copies"],
        "unpack_copies_per_rank": ranks[0]["unpack_copies"],
        "rank_group_ms": rank_group_ms,
        "reference_ms": reference_ms,
        "commands": commands,
        "reference_command": reference_command,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"run_ranked: {error}", file=sys.stderr)
        raise SystemExit(2)
