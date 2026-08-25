#!/usr/bin/env python3
"""Launch and verify the one-process-per-GPU microLLM bootstrap."""

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
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--failure-mode", choices=("none", "peer-failure"),
                        default="none")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (not args.binary.is_file() or args.steps <= 0 or
            args.timeout_seconds <= 0):
        parser.error("ranked launcher inputs are invalid")
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


def main() -> int:
    args = options()
    output = args.output_directory.resolve()
    prepare_output(output, args.overwrite)
    id_file = output / "communicator.id"
    common = ["--world-size", "2", "--id-file", str(id_file),
              "--steps", str(args.steps), "--seed", "607",
              "--timeout-ms", str(int(args.timeout_seconds * 1000))]
    commands = [
        [str(args.binary.resolve()), "--mode", "rank", "--rank", "1",
         "--local-rank", "1", *common],
        [str(args.binary.resolve()), "--mode", "rank", "--rank", "0",
         "--local-rank", "0", *common],
    ]
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
                         "--steps", str(args.steps), "--seed", "607"]
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
    rank_difference = maximum_difference(
        ranks[0]["parameters"], ranks[1]["parameters"])
    reference_difference = max(
        maximum_difference(rank["parameters"], reference["parameters"])
        for rank in ranks)
    value_count = sum(len(values) for values in reference["parameters"])
    if rank_difference != 0.0 or reference_difference > 2.0e-5:
        raise RuntimeError("ranked parameters failed the global-batch gate")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "ranked_training_summary",
        "world_size": 2,
        "steps": args.steps,
        "parameter_tensors": len(reference["parameters"]),
        "parameter_values": value_count,
        "maximum_rank_difference": rank_difference,
        "maximum_reference_difference": reference_difference,
        "rank_losses": [rank["losses"] for rank in ranks],
        "reference_losses": reference["losses"],
        "peer_processes_terminated": terminated,
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
