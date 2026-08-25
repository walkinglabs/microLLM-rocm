#!/usr/bin/env python3
"""Launch and verify the one-process-per-GPU microLLM bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--failure-mode",
                        choices=("none", "peer-failure", "group-init",
                                 "uneven-input"),
                        default="none")
    parser.add_argument("--reducer",
                        choices=("per-parameter", "bucket",
                                 "persistent-bucket", "bucket-views",
                                 "overlap-views"),
                        default="per-parameter")
    parser.add_argument("--bucket-bytes", type=int, default=4096)
    parser.add_argument("--model", choices=("tiny", "model-s"), default="tiny")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--compare-binary", type=Path)
    parser.add_argument("--rccl-debug", action="store_true")
    parser.add_argument("--rank-batch-rows", default="")
    parser.add_argument("--input-weighting",
                        choices=("equal-only", "token-weighted"),
                        default="equal-only")
    parser.add_argument("--mean-loss-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--retain-consensus-parameter-file",
                        action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (not args.binary.is_file() or args.steps <= 0 or
            args.timeout_seconds <= 0 or args.bucket_bytes < 4 or
            not math.isfinite(args.mean_loss_tolerance) or
            args.mean_loss_tolerance <= 0.0 or
            args.world_size <= 0 or args.world_size > 8):
        parser.error("ranked launcher inputs are invalid")
    if args.context == 0:
        args.context = 4 if args.model == "tiny" else 32
    if ((args.model == "tiny" and args.context != 4) or
            (args.model == "model-s" and not 1 <= args.context <= 512)):
        parser.error("context exceeds the selected model contract")
    if args.compare_binary is not None and not args.compare_binary.is_file():
        parser.error("--compare-binary is not a file")
    if args.model == "model-s" and (
            args.compare_binary is None):
        parser.error("Model-S requires --compare-binary")
    if args.retain_consensus_parameter_file and args.model != "model-s":
        parser.error("consensus parameter retention requires Model-S")
    if args.rank_batch_rows:
        try:
            args.rank_batch_rows = [
                int(value) for value in args.rank_batch_rows.split(',')]
        except ValueError:
            parser.error("rank batch rows are invalid")
    else:
        args.rank_batch_rows = [1] * args.world_size
    if (len(args.rank_batch_rows) != args.world_size or
            any(rows <= 0 for rows in args.rank_batch_rows)):
        parser.error("rank batch rows must contain one positive value per rank")
    if args.failure_mode == "uneven-input":
        if args.world_size < 2:
            parser.error("uneven-input failure requires at least two ranks")
        args.rank_batch_rows = [1] * args.world_size
        args.rank_batch_rows[1] = 2
        args.input_weighting = "equal-only"
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


def visible_gpu_count() -> int:
    count = 0
    for path in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/gpu_id"):
        try:
            count += int(path.read_text(encoding="utf-8").strip()) != 0
        except (OSError, ValueError):
            continue
    return count


def resource_preflight(world_size: int) -> dict:
    shared = shutil.disk_usage("/dev/shm")
    visible = visible_gpu_count()
    return {
        "world_size": world_size,
        "visible_gpu_count": visible,
        "visible_gpu_count_sufficient": visible >= world_size,
        "shared_memory_total_bytes": shared.total,
        "shared_memory_free_bytes": shared.free,
        "required_shared_memory_bytes": None,
        "required_shared_memory_unknown": True,
    }


def collect_rccl_debug(path: Path) -> dict:
    logs = sorted(path.glob("*.log"))
    segment_sizes = []
    no_space_logs = 0
    rccl_versions = set()
    no_space_pattern = re.compile(
        r"shared memory segment .*\(size ([0-9]+)\).*No space left on device")
    version_pattern = re.compile(r"RCCL version : ([^\n]+)")
    total_bytes = 0
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        total_bytes += log.stat().st_size
        found_sizes = [int(value) for value in no_space_pattern.findall(text)]
        segment_sizes.extend(found_sizes)
        if found_sizes:
            no_space_logs += 1
        rccl_versions.update(value.strip() for value in version_pattern.findall(text))
        log.unlink()
    return {
        "enabled": True,
        "log_files": len(logs),
        "raw_log_bytes": total_bytes,
        "raw_logs_retained": False,
        "shared_memory_no_space_logs": no_space_logs,
        "shared_memory_segment_bytes": max(segment_sizes, default=0),
        "shared_memory_failure_observed": no_space_logs > 0,
        "diagnosis": ("shared-memory-capacity-exhausted"
                      if no_space_logs > 0 else "not-established"),
        "rccl_versions": sorted(rccl_versions),
    }


def main() -> int:
    args = options()
    output = args.output_directory.resolve()
    prepare_output(output, args.overwrite)
    preflight = resource_preflight(args.world_size)
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    rank_environment = None
    debug_directory = output / "rccl-debug"
    if args.rccl_debug:
        debug_directory.mkdir(parents=True, exist_ok=True)
        rank_environment = os.environ.copy()
        rank_environment.update({
            "RCCL_LOG_LEVEL": "5",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,SHM,NET,ALLOC",
            "NCCL_DEBUG_FILE": str(debug_directory / "rank.%p.log"),
        })
    id_file = output / "communicator.id"
    common = ["--world-size", str(args.world_size),
              "--id-file", str(id_file),
              "--steps", str(args.steps), "--seed", "607",
              "--timeout-ms", str(int(args.timeout_seconds * 1000)),
              "--reducer", args.reducer,
              "--bucket-bytes", str(args.bucket_bytes),
              "--context", str(args.context),
              "--rank-batch-rows",
              ",".join(str(value) for value in args.rank_batch_rows),
              "--input-weighting", args.input_weighting]
    command_ranks = [*range(args.world_size - 1, 0, -1), 0]
    rank_parameter_files = {
        rank: output / f"rank{rank}.safetensors" for rank in command_ranks}
    reference_parameter_file = output / "reference.safetensors"
    commands = [
        [str(args.binary.resolve()), "--mode", "rank", "--rank", str(rank),
         "--local-rank", str(rank), "--model", args.model, *common]
        for rank in command_ranks
    ]
    if args.model == "model-s":
        for command, rank in zip(commands, command_ranks):
            command.extend(
                ["--parameter-file", str(rank_parameter_files[rank])])
    if args.failure_mode == "peer-failure":
        commands[0][commands[0].index("--rank") + 1] = str(args.world_size)
    group_start = time.monotonic()
    processes = [subprocess.Popen(command, cwd=ROOT, text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=rank_environment)
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
    rccl_debug = (collect_rccl_debug(debug_directory)
                  if args.rccl_debug else {"enabled": False})
    if args.rccl_debug:
        (output / "rccl-debug-summary.json").write_text(
            json.dumps(rccl_debug, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    if args.failure_mode == "peer-failure":
        for path in [*rank_parameter_files.values(), reference_parameter_file]:
            path.unlink(missing_ok=True)
        if completed or terminated < 1 or processes[0].returncode == 0:
            raise RuntimeError("peer failure did not terminate the waiting rank group")
        summary = {
            "schema_version": 1,
            "status": "pass",
            "record_type": "ranked_peer_failure_summary",
            "failure_detected": True,
            "world_size": args.world_size,
            "peer_processes_terminated": terminated,
            "rank_group_ms": rank_group_ms,
            "returncodes": [process.returncode for process in processes],
            "preflight": preflight,
            "rccl_debug": rccl_debug,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 0

    if not completed:
        for path in [*rank_parameter_files.values(), reference_parameter_file]:
            path.unlink(missing_ok=True)
        if args.failure_mode == "uneven-input":
            contract_errors = sum(
                "uneven local token counts require token-weighted input mode"
                in error for error in errors)
            if (contract_errors == 0 or
                    any(process.returncode == 0 for process in processes)):
                raise RuntimeError(
                    "uneven-input failure did not match the weighting contract")
            summary = {
                "schema_version": 1,
                "status": "pass",
                "record_type": "ranked_uneven_input_failure_summary",
                "world_size": args.world_size,
                "rank_batch_rows": args.rank_batch_rows,
                "input_weighting": args.input_weighting,
                "failure_detected": True,
                "weighting_contract_error_processes": contract_errors,
                "peer_processes_terminated": terminated,
                "rank_group_ms": rank_group_ms,
                "returncodes": [process.returncode for process in processes],
                "decision": "admit explicit token-weighted input mode",
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            print(json.dumps(summary, sort_keys=True))
            return 0
        if args.failure_mode == "group-init":
            system_error_ranks = sum(
                "ncclCommInitRank" in error and
                "system error" in error for error in errors)
            if (system_error_ranks == 0 or
                    any(process.returncode == 0 for process in processes)):
                raise RuntimeError(
                    "ranked group-init failure did not match the expected boundary")
            shared_memory_bytes = preflight["shared_memory_total_bytes"]
            summary = {
                "schema_version": 1,
                "status": "pass",
                "record_type": "ranked_group_init_failure_summary",
                "world_size": args.world_size,
                "group_initialized": False,
                "failure_detected": True,
                "system_error_ranks": system_error_ranks,
                "peer_processes_terminated": terminated,
                "rank_group_ms": rank_group_ms,
                "returncodes": [process.returncode for process in processes],
                "shared_memory_bytes": shared_memory_bytes,
                "preflight": preflight,
                "rccl_debug": rccl_debug,
                "decision": "retain world-size interface and record environment boundary",
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            print(json.dumps(summary, sort_keys=True))
            return 0
        raise RuntimeError("ranked training timed out or one rank failed")
    emitted = [load_record(text, f"process{index}")
               for index, text in enumerate(outputs)]
    ranks_by_identity = {rank.get("rank"): rank for rank in emitted}
    if set(ranks_by_identity) != set(range(args.world_size)):
        raise RuntimeError("ranked process identities changed")
    ranks = [ranks_by_identity[rank] for rank in range(args.world_size)]
    reference_command = [str(args.binary.resolve()), "--mode", "reference",
                         "--steps", str(args.steps), "--seed", "607",
                         "--model", args.model,
                         "--context", str(args.context),
                         "--world-size", str(args.world_size),
                         "--rank-batch-rows",
                         ",".join(str(value) for value in args.rank_batch_rows),
                         "--input-weighting", args.input_weighting]
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
    if (reference.get("context") != args.context or
            reference.get("world_size") != args.world_size or
            any(rank.get("context") != args.context or
                rank.get("world_size") != args.world_size or
                rank["parameter_names"] != reference["parameter_names"]
                for rank in ranks)):
        raise RuntimeError("rank identity or parameter names changed")
    average_tokens = (
        sum(args.rank_batch_rows) * args.context / args.world_size)
    for rank_index, rank in enumerate(ranks):
        local_tokens = args.rank_batch_rows[rank_index] * args.context
        expected_scale = (local_tokens / average_tokens
                          if args.input_weighting == "token-weighted" else 1.0)
        if (rank.get("input_weighting") != args.input_weighting or
                rank.get("local_batch_rows") !=
                args.rank_batch_rows[rank_index] or
                rank.get("local_tokens") != local_tokens or
                abs(rank.get("average_tokens", -1.0) - average_tokens) > 1.0e-6 or
                abs(rank.get("local_gradient_scale", -1.0) -
                    expected_scale) > 1.0e-6):
            raise RuntimeError("rank input weighting metadata changed")
    if (reference.get("input_weighting") != args.input_weighting or
            abs(reference.get("average_tokens", -1.0) - average_tokens) > 1.0e-6):
        raise RuntimeError("reference input weighting metadata changed")
    expected_collectives = (
        ranks[0]["buckets"] if args.reducer != "per-parameter" else
        args.steps * len(reference["parameter_names"]))
    if any(rank.get("reducer") != args.reducer or
           rank.get("collectives") != expected_collectives or
           rank.get("buckets") != ranks[0]["buckets"]
           for rank in ranks):
        raise RuntimeError("rank reducer collective count changed")
    if args.model == "model-s":
        assert args.compare_binary is not None
        rank_comparisons = [
            compare_safetensors(
                args.compare_binary, rank_parameter_files[0],
                rank_parameter_files[rank], args.timeout_seconds)
            for rank in range(1, args.world_size)]
        reference_comparisons = [
            compare_safetensors(
                args.compare_binary, reference_parameter_file,
                rank_parameter_files[rank], args.timeout_seconds)
            for rank in range(args.world_size)]
        rank_difference = max(
            (comparison["maximum_absolute_difference"]
             for comparison in rank_comparisons), default=0.0)
        rank_rms_difference = max(
            (comparison["rms_difference"]
             for comparison in rank_comparisons), default=0.0)
        reference_difference = max(
            comparison["maximum_absolute_difference"]
            for comparison in reference_comparisons)
        reference_rms_difference = max(
            comparison["rms_difference"]
            for comparison in reference_comparisons)
        tensor_count = reference_comparisons[0]["tensor_count"]
        value_count = reference_comparisons[0]["compared_elements"]
        reference_tolerance = 1.0e-2
        reference_rms_tolerance = 1.0e-5
    else:
        rank_difference = max(
            (maximum_difference(ranks[0]["parameters"], rank["parameters"])
             for rank in ranks[1:]), default=0.0)
        rank_rms_difference = max(
            (rms_difference(ranks[0]["parameters"], rank["parameters"])
             for rank in ranks[1:]), default=0.0)
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
    step_timing_fields = ("step_training_ms", "step_forward_backward_ms",
                          "step_reducer_ms", "step_optimizer_ms")
    step_count_fields = ("step_collectives", "step_buckets",
                         "step_pack_copies", "step_unpack_copies",
                         "step_gradient_views",
                         "step_reducer_allocation_calls",
                         "step_reducer_backend_allocation_calls",
                         "step_reducer_deallocation_calls",
                         "step_reducer_total_allocated_bytes",
                         "step_plan_reused",
                         "step_reducer_current_bytes_before",
                         "step_reducer_current_bytes_after",
                         "step_reducer_peak_bytes_after",
                         "step_overlap_enabled",
                         "step_overlapped_buckets",
                         "step_weighted_gradient_scales")
    if any(not isinstance(rank.get(field), list) or
           len(rank[field]) != args.steps
           for rank in ranks for field in
           (*step_timing_fields, *step_count_fields)):
        raise RuntimeError("rank per-step record count changed")
    if any(not math.isfinite(value) or value < 0.0
           for rank in ranks for field in step_timing_fields
           for value in rank[field]):
        raise RuntimeError("rank per-step timing is invalid")
    if any(rank["step_training_ms"][step] <= 0.0 or
           rank["step_forward_backward_ms"][step] <= 0.0 or
           rank["step_reducer_ms"][step] <= 0.0 or
           rank["step_optimizer_ms"][step] <= 0.0 or
           rank["step_training_ms"][step] + 1.0e-6 <
           rank["step_forward_backward_ms"][step] +
           rank["step_reducer_ms"][step] +
           rank["step_optimizer_ms"][step]
           for rank in ranks for step in range(args.steps)):
        raise RuntimeError("rank per-step timings do not form a complete interval")
    if any(not isinstance(value, int) or value < 0
           for rank in ranks for field in step_count_fields
           for value in rank[field]):
        raise RuntimeError("rank per-step reducer counters are invalid")
    engine_fields = ("engine_current_bytes", "engine_peak_bytes",
                     "engine_cached_bytes", "engine_reserved_bytes",
                     "engine_allocation_calls",
                     "engine_backend_allocation_calls")
    if any(not isinstance(rank.get(field), int) or rank[field] < 0
           for rank in ranks for field in engine_fields):
        raise RuntimeError("rank engine memory counters are invalid")
    if any(sum(rank["step_collectives"]) != rank["collectives"] or
           sum(rank["step_buckets"]) != rank["buckets"] or
           sum(rank["step_pack_copies"]) != rank["pack_copies"] or
           sum(rank["step_unpack_copies"]) != rank["unpack_copies"] or
           sum(rank["step_gradient_views"]) != rank["gradient_views"]
           for rank in ranks):
        raise RuntimeError("rank per-step reducer totals changed")
    expected_weighted_gradient_scales = [
        (len(reference["parameter_names"])
         if args.input_weighting == "token-weighted" and
         args.rank_batch_rows[rank_index] * args.context != average_tokens
         else 0)
        for rank_index in range(args.world_size)]
    if any(rank.get("weighted_gradient_scales") !=
           args.steps * expected_weighted_gradient_scales[rank_index] or
           rank["step_weighted_gradient_scales"] !=
           [expected_weighted_gradient_scales[rank_index]] * args.steps
           for rank_index, rank in enumerate(ranks)):
        raise RuntimeError("rank per-leaf gradient weighting count changed")
    expected_reuse = [0] + [1] * (args.steps - 1)
    if args.reducer in ("persistent-bucket", "bucket-views",
                        "overlap-views"):
        if any(rank.get("persistent_storage") is not True or
               rank.get("plan_reuses") != args.steps - 1 or
               rank.get("plan_capacity_elements", 0) <= 0 or
               rank.get("plan_capacity_bytes", 0) <= 0 or
               rank["step_plan_reused"] != expected_reuse
               for rank in ranks):
            raise RuntimeError("persistent rank bucket plan contract changed")
    elif any(rank.get("persistent_storage") is not False or
             rank.get("plan_reuses") != 0 or
             rank.get("plan_capacity_elements") != 0 or
             rank.get("plan_capacity_bytes") != 0 or
             any(rank["step_plan_reused"])
             for rank in ranks):
        raise RuntimeError("non-persistent reducer exposed a bucket plan")
    if args.reducer in ("bucket-views", "overlap-views"):
        if any(rank.get("gradient_views") !=
               args.steps * len(reference["parameter_names"]) or
               rank.get("unpack_copies") != 0 or
               rank["step_gradient_views"] !=
               [len(reference["parameter_names"])] * args.steps
               for rank in ranks):
            raise RuntimeError("rank bucket-view contract changed")
    elif any(rank.get("gradient_views") != 0 or
             any(rank["step_gradient_views"])
             for rank in ranks):
        raise RuntimeError("non-view reducer exposed gradient views")
    expected_overlap = [0] + [1] * (args.steps - 1)
    expected_overlapped_buckets = [0] + [
        ranks[0]["buckets"] // args.steps] * (args.steps - 1)
    if args.reducer == "overlap-views":
        if any(rank.get("overlap_steps") != args.steps - 1 or
               rank.get("overlapped_buckets") !=
               (args.steps - 1) * ranks[0]["buckets"] // args.steps or
               rank["step_overlap_enabled"] != expected_overlap or
               rank["step_overlapped_buckets"] !=
               expected_overlapped_buckets
               for rank in ranks):
            raise RuntimeError("rank gradient overlap contract changed")
    elif any(rank.get("overlap_steps") != 0 or
             rank.get("overlapped_buckets") != 0 or
             any(rank["step_overlap_enabled"]) or
             any(rank["step_overlapped_buckets"])
             for rank in ranks):
        raise RuntimeError("non-overlap reducer exposed overlap state")
    loss_difference = max(
        abs(sum(rank["losses"][step] * args.rank_batch_rows[index]
                for index, rank in enumerate(ranks)) /
            sum(args.rank_batch_rows) - reference["losses"][step])
        for step in range(args.steps))
    gate_passed = not (
        rank_difference != 0.0 or rank_rms_difference != 0.0 or
        reference_difference > reference_tolerance or
        reference_rms_difference > reference_rms_tolerance or
        loss_difference > args.mean_loss_tolerance)
    consensus_parameter_file: Path | None = None
    if args.model == "model-s":
        if gate_passed and args.retain_consensus_parameter_file:
            consensus_parameter_file = rank_parameter_files[0]
        for rank, path in rank_parameter_files.items():
            if not (gate_passed and args.retain_consensus_parameter_file and
                    rank == 0):
                path.unlink(missing_ok=True)
        reference_parameter_file.unlink(missing_ok=True)
    if not gate_passed:
        raise RuntimeError("ranked parameters failed the global-batch gate")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "record_type": "ranked_training_summary",
        "world_size": args.world_size,
        "model": args.model,
        "context": args.context,
        "rank_batch_rows": args.rank_batch_rows,
        "input_weighting": args.input_weighting,
        "average_tokens": average_tokens,
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
        "mean_loss_tolerance": args.mean_loss_tolerance,
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
        "maximum_rank_step_training_ms": [
            max(rank["step_training_ms"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_forward_backward_ms": [
            max(rank["step_forward_backward_ms"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_reducer_ms": [
            max(rank["step_reducer_ms"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_optimizer_ms": [
            max(rank["step_optimizer_ms"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_collectives": [
            max(rank["step_collectives"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_buckets": [
            max(rank["step_buckets"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_pack_copies": [
            max(rank["step_pack_copies"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_unpack_copies": [
            max(rank["step_unpack_copies"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_gradient_views": [
            max(rank["step_gradient_views"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_reducer_allocation_calls": [
            max(rank["step_reducer_allocation_calls"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_rank_step_reducer_backend_allocation_calls": [
            max(rank["step_reducer_backend_allocation_calls"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_rank_step_reducer_deallocation_calls": [
            max(rank["step_reducer_deallocation_calls"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_rank_step_reducer_total_allocated_bytes": [
            max(rank["step_reducer_total_allocated_bytes"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_rank_step_plan_reused": [
            max(rank["step_plan_reused"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_reducer_current_bytes_before": [
            max(rank["step_reducer_current_bytes_before"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_rank_step_reducer_current_bytes_after": [
            max(rank["step_reducer_current_bytes_after"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_rank_step_reducer_peak_bytes_after": [
            max(rank["step_reducer_peak_bytes_after"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_rank_step_overlap_enabled": [
            max(rank["step_overlap_enabled"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_overlapped_buckets": [
            max(rank["step_overlapped_buckets"][step] for rank in ranks)
            for step in range(args.steps)],
        "maximum_rank_step_weighted_gradient_scales": [
            max(rank["step_weighted_gradient_scales"][step]
                for rank in ranks) for step in range(args.steps)],
        "maximum_engine_current_bytes": max(
            rank["engine_current_bytes"] for rank in ranks),
        "maximum_engine_peak_bytes": max(
            rank["engine_peak_bytes"] for rank in ranks),
        "maximum_engine_cached_bytes": max(
            rank["engine_cached_bytes"] for rank in ranks),
        "maximum_engine_reserved_bytes": max(
            rank["engine_reserved_bytes"] for rank in ranks),
        "maximum_engine_allocation_calls": max(
            rank["engine_allocation_calls"] for rank in ranks),
        "maximum_engine_backend_allocation_calls": max(
            rank["engine_backend_allocation_calls"] for rank in ranks),
        "persistent_storage": ranks[0]["persistent_storage"],
        "plan_reuses_per_rank": ranks[0]["plan_reuses"],
        "plan_capacity_elements_per_rank": ranks[0]["plan_capacity_elements"],
        "plan_capacity_bytes_per_rank": ranks[0]["plan_capacity_bytes"],
        "gradient_views_per_rank": ranks[0]["gradient_views"],
        "overlap_steps_per_rank": ranks[0]["overlap_steps"],
        "overlapped_buckets_per_rank": ranks[0]["overlapped_buckets"],
        "rank_weighted_gradient_scales": [
            rank["weighted_gradient_scales"] for rank in ranks],
        "maximum_weighted_gradient_scales_per_rank": max(
            rank["weighted_gradient_scales"] for rank in ranks),
        "parameter_files_retained": consensus_parameter_file is not None,
        "consensus_parameter_file": (
            str(consensus_parameter_file)
            if consensus_parameter_file is not None else ""),
        "peer_processes_terminated": terminated,
        "collectives_per_rank": expected_collectives,
        "buckets_per_rank": ranks[0]["buckets"],
        "pack_copies_per_rank": ranks[0]["pack_copies"],
        "unpack_copies_per_rank": ranks[0]["unpack_copies"],
        "rank_group_ms": rank_group_ms,
        "reference_ms": reference_ms,
        "commands": commands,
        "reference_command": reference_command,
        "preflight": preflight,
        "rccl_debug": rccl_debug,
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
