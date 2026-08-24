#!/usr/bin/env python3
"""Gate exact BF16 gate/up solutions on cold start and steady official models."""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


MODELS = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
PHASES = ("cold", "steady")
POLICIES = ("default", "exact")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--tuner", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--maximum-algorithms", type=int, default=64)
    parser.add_argument("--workspace-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--rms-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.005)
    parser.add_argument("--minimum-cold-speedup", type=float, default=1.02)
    parser.add_argument("--minimum-steady-speedup", type=float, default=1.01)
    result = parser.parse_args()
    if (result.runs <= 0 or result.sequence <= 0 or
            result.maximum_algorithms <= 0 or result.maximum_algorithms > 256 or
            result.workspace_bytes < 0 or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.maximum_peak_ratio < 1 or result.minimum_cold_speedup <= 1 or
            result.minimum_steady_speedup <= 1 or
            not result.manifest.is_file() or not result.binary.is_file() or
            not result.tuner.is_file()):
        parser.error("exact BF16 startup inputs are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != MODELS:
        raise RuntimeError("exact BF16 startup gate requires pinned Qwen and DeepSeek")
    for model in result:
        config = Path(model["config"])
        if not config.is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
        external = json.loads(config.read_text(encoding="utf-8"))
        model["hidden_size"] = int(external["hidden_size"])
        model["intermediate_size"] = int(external["intermediate_size"])
    return result


def clean_environment() -> dict[str, str]:
    result = os.environ.copy()
    result["HIPBLASLT_PRELOAD_KERNELS"] = "0"
    return result


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("child process emitted no JSON")


def tune(args: argparse.Namespace, model: dict) -> tuple[int, list[dict], dict]:
    records = []
    for process_run in range(1, args.runs + 1):
        command = [
            str(args.tuner), "--rows", str(args.sequence),
            "--inner", str(model["hidden_size"]),
            "--columns", str(model["intermediate_size"]),
            "--output-dtype", "bf16",
            "--maximum-algorithms", str(args.maximum_algorithms),
            "--workspace-bytes", str(args.workspace_bytes),
            "--warmup", "2", "--repetitions", "5",
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False,
            env=clean_environment())
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        record = last_json(completed.stdout)
        if record.get("status") != "pass":
            raise RuntimeError(f"invalid tuner record: {model['name']}")
        record.update({
            "record_type": "bf16_exact_startup_tuning",
            "model": model["name"], "revision": model["revision"],
            "process_run": process_run,
        })
        records.append(record)
    passing_sets = [
        {int(candidate["index"]) for candidate in record["candidates"]
         if candidate.get("correctness_passed") is True}
        for record in records
    ]
    common = set.intersection(*passing_sets)
    if not common:
        raise RuntimeError(f"no common exact BF16 candidate: {model['name']}")
    candidate_times = {}
    for index in common:
        times = []
        for record in records:
            found = next(candidate for candidate in record["candidates"]
                         if int(candidate["index"]) == index)
            times.append(float(found["event_ms_p50"]))
        candidate_times[index] = statistics.median(times)
    selected = min(common, key=lambda index: (candidate_times[index], index))
    default_event = statistics.median(
        float(record["default_event_ms_p50"]) for record in records)
    selection = {
        "model": model["name"], "revision": model["revision"],
        "rows": args.sequence, "inner": model["hidden_size"],
        "columns": model["intermediate_size"], "output_dtype": "bf16",
        "tuner_processes": len(records),
        "common_passing_candidates": len(common),
        "selected_index": selected,
        "default_event_ms_p50_median": default_event,
        "selected_event_ms_p50_median": candidate_times[selected],
        "operator_speedup": default_event / candidate_times[selected],
    }
    return selected, records, selection


def repeated(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise RuntimeError("exact BF16 startup token seed cannot be empty")
    return [seed[index % len(seed)] for index in range(length)]


def model_command(args: argparse.Namespace, model: dict, phase: str,
                  policy: str, selected_index: int, logits: Path) -> list[str]:
    tokens = repeated(model["inference"]["token_ids"], args.sequence)
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", "1",
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true",
        "--bf16-ffn-arena-minimum-rows", "512",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0" if phase == "cold" else "2",
        "--prefill-steps", "1" if phase == "cold" else "5",
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]
    if policy == "exact":
        result.extend(["--bf16-algorithm-index", str(selected_index)])
    return result


def floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def error(reference: array.array, actual: array.array) -> tuple[float, float, bool]:
    if len(reference) != len(actual) or not reference:
        raise RuntimeError("exact BF16 complete-logit size changed")
    maximum = 0.0
    squared = 0.0
    finite = True
    for expected, observed in zip(reference, actual, strict=True):
        difference = abs(expected - observed)
        maximum = max(maximum, difference)
        squared += difference * difference
        finite = finite and math.isfinite(observed)
    return maximum, math.sqrt(squared / len(reference)), finite


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    tuning_records = []
    selections = {}
    for model in selected_models:
        selected, records, summary = tune(args, model)
        selections[model["name"]] = (selected, summary)
        tuning_records.extend(records)

    model_records = []
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="microllm-bf16-exact-startup-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            selected_index = selections[model["name"]][0]
            for process_run in range(1, args.runs + 1):
                phases = list(PHASES)
                policies = list(POLICIES)
                if process_run % 2 == 0:
                    phases.reverse()
                    policies.reverse()
                for phase in phases:
                    for policy in policies:
                        stem = (f"{model['name']}-p{process_run}-"
                                f"{phase}-{policy}")
                        logits = temporary / f"{stem}.bin"
                        started = time.perf_counter()
                        completed = subprocess.run(
                            model_command(args, model, phase, policy,
                                          selected_index, logits),
                            text=True, capture_output=True, check=False,
                            env=clean_environment())
                        process_wall_ms = (
                            time.perf_counter() - started) * 1000.0
                        if completed.returncode != 0:
                            raise RuntimeError(
                                completed.stdout + completed.stderr)
                        record = last_json(completed.stdout)
                        if record.get("status") != "pass":
                            raise RuntimeError(f"invalid model record: {stem}")
                        record.update({
                            "record_type": "bf16_exact_startup_model",
                            "model": model["name"],
                            "revision": model["revision"],
                            "phase": phase, "policy": policy,
                            "process_run": process_run,
                            "phase_order": phases,
                            "policy_order": policies,
                            "selected_index":
                                selected_index if policy == "exact" else -1,
                            "process_wall_ms": process_wall_ms,
                        })
                        model_records.append(record)
                        outputs[(model["name"], phase, policy, process_run)] = \
                            floats(logits)

    comparisons = []
    for model in selected_models:
        name = model["name"]
        selected = [row for row in model_records if row["model"] == name]
        grouped = {
            (phase, policy): [
                row for row in selected
                if row["phase"] == phase and row["policy"] == policy]
            for phase in PHASES for policy in POLICIES
        }
        reference = outputs[(name, "cold", "default", 1)]
        maximum = 0.0
        rms = 0.0
        finite = True
        for phase in PHASES:
            for policy in POLICIES:
                for process_run in range(1, args.runs + 1):
                    current = error(
                        reference, outputs[(name, phase, policy, process_run)])
                    maximum = max(maximum, current[0])
                    rms = max(rms, current[1])
                    finite = finite and current[2]
        default_cold = median(grouped[("cold", "default")], "forward_ms")
        exact_cold = median(grouped[("cold", "exact")], "forward_ms")
        default_wall = median(
            grouped[("cold", "default")], "process_wall_ms")
        exact_wall = median(grouped[("cold", "exact")], "process_wall_ms")
        default_steady = median(
            grouped[("steady", "default")], "prefill_tokens_per_second")
        exact_steady = median(
            grouped[("steady", "exact")], "prefill_tokens_per_second")
        default_peak = int(median(
            grouped[("steady", "default")], "engine_peak_bytes"))
        exact_peak = int(median(
            grouped[("steady", "exact")], "engine_peak_bytes"))
        comparisons.append({
            **selections[name][1],
            "default_cold_forward_ms": default_cold,
            "exact_cold_forward_ms": exact_cold,
            "cold_forward_speedup": default_cold / exact_cold,
            "default_cold_process_wall_ms": default_wall,
            "exact_cold_process_wall_ms": exact_wall,
            "cold_process_speedup": default_wall / exact_wall,
            "default_steady_tokens_per_second": default_steady,
            "exact_steady_tokens_per_second": exact_steady,
            "steady_speedup": exact_steady / default_steady,
            "default_peak_bytes": default_peak,
            "exact_peak_bytes": exact_peak,
            "peak_ratio": exact_peak / default_peak,
            "maximum_absolute_logit_difference": maximum,
            "maximum_rms_logit_difference": rms,
            "finite_complete_logits": finite,
        })

    correctness = all(
        row["finite_complete_logits"] and
        row["maximum_absolute_logit_difference"] <=
            args.maximum_absolute_tolerance and
        row["maximum_rms_logit_difference"] <= args.rms_tolerance
        for row in comparisons)
    memory = all(row["peak_ratio"] <= args.maximum_peak_ratio
                 for row in comparisons)
    cold = all(row["cold_forward_speedup"] >= args.minimum_cold_speedup
               for row in comparisons)
    steady = all(row["steady_speedup"] >= args.minimum_steady_speedup
                 for row in comparisons)
    performance = cold and steady
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and memory else "fail",
        "record_type": "bf16_exact_startup_summary",
        "tuner_processes": len(tuning_records),
        "model_processes": len(model_records),
        "correctness_gate": correctness,
        "memory_gate": memory,
        "cold_performance_gate": cold,
        "steady_performance_gate": steady,
        "performance_gate": performance,
        "comparisons": comparisons,
        "decision": ("keep exact gate/up solution"
                     if performance else
                     "reject exact gate/up solution; default unchanged"),
    }
    for name, rows in (("tuning-raw.jsonl", tuning_records),
                       ("model-raw.jsonl", model_records)):
        with (args.output_directory / name).open(
                "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
