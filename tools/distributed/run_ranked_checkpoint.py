#!/usr/bin/env python3
"""Verify rank0-only checkpoint publication and distributed resume."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--model", choices=("tiny", "model-s"), default="tiny")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--compare-binary", type=Path)
    parser.add_argument("--first-steps", type=int, default=2)
    parser.add_argument("--resumed-steps", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (not args.binary.is_file() or args.first_steps <= 0 or
            args.resumed_steps <= 0 or args.timeout_seconds <= 0):
        parser.error("ranked checkpoint inputs are invalid")
    if args.context == 0:
        args.context = 4 if args.model == "tiny" else 32
    if ((args.model == "tiny" and args.context != 4) or
            (args.model == "model-s" and not 1 <= args.context <= 512)):
        parser.error("checkpoint context exceeds the model contract")
    if args.compare_binary is not None and not args.compare_binary.is_file():
        parser.error("--compare-binary is not a file")
    if args.model == "model-s" and args.compare_binary is None:
        parser.error("Model-S checkpoint requires --compare-binary")
    return args


def prepare(path: Path, overwrite: bool) -> None:
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


def parse_record(text: str, name: str) -> dict:
    lines = [line for line in text.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"{name} emitted an unexpected record count")
    value = json.loads(lines[0])
    if value.get("schema_version") != 1 or value.get("status") != "pass":
        raise RuntimeError(f"{name} record failed")
    return value


def compare_safetensors(binary: Path, baseline: Path, candidate: Path,
                        timeout: float) -> dict:
    completed = subprocess.run(
        [str(binary.resolve()), str(baseline), str(candidate)],
        cwd=ROOT, text=True, capture_output=True, timeout=timeout,
        check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    return parse_record(completed.stdout, "rank safetensors comparison")


def run_group(binary: Path, output: Path, steps: int, timeout: float,
              checkpoint: Path, ready: Path, resume: Path | None = None,
              inject_failure: bool = False, model: str = "tiny",
              context: int = 4,
              compare_binary: Path | None = None) -> tuple[list[dict], dict]:
    output.mkdir(parents=True, exist_ok=True)
    id_file = output / "communicator.id"
    common = [
        "--world-size", "2", "--id-file", str(id_file),
        "--steps", str(steps), "--seed", "607", "--timeout-ms",
        str(int(timeout * 1000)), "--model", model, "--context", str(context),
        "--reducer", "per-parameter", "--bucket-bytes", "4096",
        "--checkpoint-file", str(checkpoint),
        "--checkpoint-ready-file", str(ready),
    ]
    if resume is not None:
        common.extend(["--resume-file", str(resume)])
    commands = [
        [str(binary.resolve()), "--mode", "rank", "--rank", "1",
         "--local-rank", "1", *common],
        [str(binary.resolve()), "--mode", "rank", "--rank", "0",
         "--local-rank", "0", *common],
    ]
    parameter_files = [output / "rank1.safetensors",
                       output / "rank0.safetensors"]
    if model == "model-s":
        for command, path in zip(commands, parameter_files):
            command.extend(["--parameter-file", str(path)])
    if inject_failure:
        commands[1].append("--inject-checkpoint-failure")
    started = time.monotonic()
    processes = [subprocess.Popen(command, cwd=ROOT, text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
                 for command in commands]
    completed, terminated = wait_group(processes, timeout)
    elapsed_ms = (time.monotonic() - started) * 1000.0
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
    process = {
        "completed": completed,
        "terminated": terminated,
        "returncodes": [item.returncode for item in processes],
        "group_ms": elapsed_ms,
        "commands": commands,
        "rank_parameter_difference": None,
    }
    if inject_failure:
        return [], process
    if not completed:
        raise RuntimeError("ranked checkpoint group failed or timed out")
    ranks = [parse_record(text, f"rank{index}")
             for index, text in enumerate(outputs)]
    if ranks[0].get("rank") != 1 or ranks[1].get("rank") != 0:
        raise RuntimeError("ranked checkpoint identity changed")
    if model == "model-s":
        if compare_binary is None:
            raise RuntimeError("Model-S rank comparison binary is missing")
        comparison = compare_safetensors(
            compare_binary, parameter_files[0], parameter_files[1], timeout)
        process["rank_parameter_difference"] = (
            comparison["maximum_absolute_difference"])
        process["rank_parameter_rms_difference"] = comparison["rms_difference"]
        process["parameter_tensors"] = comparison["tensor_count"]
        process["parameter_values"] = comparison["compared_elements"]
        for path in parameter_files:
            path.unlink(missing_ok=True)
    else:
        process["rank_parameter_difference"] = parameter_difference(
            ranks[0], ranks[1])
    return ranks, process


def parameter_difference(left: dict, right: dict) -> float:
    if (left["parameter_names"] != right["parameter_names"] or
            len(left["parameters"]) != len(right["parameters"])):
        raise RuntimeError("ranked checkpoint parameter contract changed")
    maximum = 0.0
    for lhs, rhs in zip(left["parameters"], right["parameters"]):
        if len(lhs) != len(rhs):
            raise RuntimeError("ranked checkpoint parameter shape changed")
        maximum = max(maximum, max(
            (abs(a - b) for a, b in zip(lhs, rhs)), default=0.0))
    return maximum


def verify_group(ranks: list[dict], process: dict, initial: int, final: int,
                 resumed: bool) -> None:
    if (process.get("rank_parameter_difference") != 0.0 or
            any(rank.get("initial_step") != initial or
                rank.get("final_step") != final or
                rank.get("optimizer_step") != final or
                rank.get("resumed") is not resumed or
                rank.get("checkpoint_requested") is not True or
                rank.get("checkpoint_verified") is not True
                for rank in ranks) or
            ranks[0].get("checkpoint_written") is not False or
            ranks[1].get("checkpoint_written") is not True):
        raise RuntimeError("ranked checkpoint group contract changed")


def main() -> int:
    args = options()
    output = args.output_directory.resolve()
    prepare(output, args.overwrite)
    final_step = args.first_steps + args.resumed_steps
    interrupted = output / "interrupted.ckpt"
    interrupted_ready = output / "interrupted.ready"
    resumed_final = output / "resumed-final.ckpt"
    resumed_ready = output / "resumed-final.ready"
    uninterrupted_final = output / "uninterrupted-final.ckpt"
    uninterrupted_ready = output / "uninterrupted-final.ready"

    first_ranks, first_process = run_group(
        args.binary, output / "first-segment", args.first_steps,
        args.timeout_seconds, interrupted, interrupted_ready,
        model=args.model, context=args.context,
        compare_binary=args.compare_binary)
    verify_group(first_ranks, first_process, 0, args.first_steps, False)
    resumed_ranks, resumed_process = run_group(
        args.binary, output / "resumed-segment", args.resumed_steps,
        args.timeout_seconds, resumed_final, resumed_ready, interrupted,
        model=args.model, context=args.context,
        compare_binary=args.compare_binary)
    verify_group(
        resumed_ranks, resumed_process, args.first_steps, final_step, True)
    uninterrupted_ranks, uninterrupted_process = run_group(
        args.binary, output / "uninterrupted", final_step,
        args.timeout_seconds, uninterrupted_final, uninterrupted_ready,
        model=args.model, context=args.context,
        compare_binary=args.compare_binary)
    verify_group(
        uninterrupted_ranks, uninterrupted_process, 0, final_step, False)

    resumed_rank_difference = resumed_process["rank_parameter_difference"]
    uninterrupted_rank_difference = uninterrupted_process[
        "rank_parameter_difference"]
    checkpoint_bytes_equal = (
        resumed_final.read_bytes() == uninterrupted_final.read_bytes())
    trajectory_difference = 0.0 if checkpoint_bytes_equal else 1.0
    if (resumed_rank_difference != 0.0 or
            uninterrupted_rank_difference != 0.0 or
            trajectory_difference != 0.0 or not checkpoint_bytes_equal):
        raise RuntimeError("ranked checkpoint resume trajectory changed")

    failure_checkpoint = output / "failure.ckpt"
    failure_ready = output / "failure.ready"
    failure_timeout = min(args.timeout_seconds, 15.0)
    _, failure_process = run_group(
        args.binary, output / "write-failure", 1, failure_timeout,
        failure_checkpoint, failure_ready, inject_failure=True)
    if (failure_process["completed"] or failure_process["terminated"] < 1 or
            failure_process["returncodes"][1] != 1 or
            failure_process["returncodes"][0] not in (-15, -9) or
            failure_checkpoint.exists() or failure_ready.exists()):
        raise RuntimeError("rank0 checkpoint failure did not terminate its peer")

    checkpoint_sizes = {
        "interrupted": interrupted.stat().st_size,
        "resumed_final": resumed_final.stat().st_size,
        "uninterrupted_final": uninterrupted_final.stat().st_size,
    }
    for path in (interrupted, interrupted_ready, resumed_final, resumed_ready,
                 uninterrupted_final, uninterrupted_ready, failure_checkpoint,
                 failure_ready):
        path.unlink(missing_ok=True)
    if (list(output.rglob("*.ckpt")) or list(output.rglob("*.ready")) or
            list(output.rglob("*.safetensors")) or
            list(output.rglob("*.tmp")) or
            list(output.rglob("communicator.id"))):
        raise RuntimeError("ranked checkpoint temporary files were retained")

    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "ranked_checkpoint_summary",
        "world_size": 2,
        "model": args.model,
        "context": args.context,
        "first_steps": args.first_steps,
        "resumed_steps": args.resumed_steps,
        "final_step": final_step,
        "rank_processes": 8,
        "rank0_checkpoint_writes": 3,
        "nonzero_rank_checkpoint_writes": 0,
        "resumed_rank_difference": resumed_rank_difference,
        "uninterrupted_rank_difference": uninterrupted_rank_difference,
        "resume_vs_uninterrupted_parameter_difference": trajectory_difference,
        "resume_vs_uninterrupted_checkpoint_bytes_equal": checkpoint_bytes_equal,
        "checkpoint_sizes": checkpoint_sizes,
        "checkpoint_write_ms": {
            "interrupted": first_ranks[1]["checkpoint_write_ms"],
            "resumed_final": resumed_ranks[1]["checkpoint_write_ms"],
            "uninterrupted_final":
                uninterrupted_ranks[1]["checkpoint_write_ms"],
        },
        "maximum_resume_ms": max(
            rank["resume_ms"] for rank in resumed_ranks),
        "maximum_checkpoint_wait_ms": max(
            rank["checkpoint_wait_ms"] for ranks in
            (first_ranks, resumed_ranks, uninterrupted_ranks)
            for rank in ranks),
        "maximum_checkpoint_verify_ms": max(
            rank["checkpoint_verify_ms"] for ranks in
            (first_ranks, resumed_ranks, uninterrupted_ranks)
            for rank in ranks),
        "checkpoint_files_retained": False,
        "failure_detected": True,
        "peer_processes_terminated": failure_process["terminated"],
        "failure_returncodes": failure_process["returncodes"],
        "processes": {
            "first": first_process,
            "resumed": resumed_process,
            "uninterrupted": uninterrupted_process,
            "failure": failure_process,
        },
        "decision": ("complete Model-S ranked checkpoint smoke"
                     if args.model == "model-s" else
                     "admit Model-S ranked checkpoint smoke"),
    }
    (output / "failure.json").write_text(
        json.dumps(failure_process, indent=2, sort_keys=True) + "\n",
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
        print(f"run_ranked_checkpoint: {error}", file=sys.stderr)
        raise SystemExit(2)
