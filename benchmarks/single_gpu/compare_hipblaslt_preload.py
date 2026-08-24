#!/usr/bin/env python3
"""Measure whether hipBLASLt's all-kernel preload improves first-request latency."""

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


POLICIES = ("fp32", "bf16_lazy", "bf16_preload_all")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=0.25)
    parser.add_argument("--rms-tolerance", type=float, default=0.05)
    parser.add_argument("--minimum-preload-slowdown", type=float, default=1.25)
    result = parser.parse_args()
    if (result.runs <= 0 or result.sequence <= 0 or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.minimum_preload_slowdown <= 1 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("cold-start inputs are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("cold-start gate requires pinned Qwen and DeepSeek")
    for model in result:
        if not Path(model["config"]).is_file() or \
                not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise RuntimeError("cold-start token seed cannot be empty")
    return [seed[index % len(seed)] for index in range(length)]


def command(args: argparse.Namespace, model: dict, policy: str,
            logits: Path) -> list[str]:
    tokens = repeated(model["inference"]["token_ids"], args.sequence)
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", "1",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]
    if policy != "fp32":
        result.extend([
            "--bf16-ffn", "true", "--bf16-attention", "true",
            "--bf16-ffn-arena", "true",
            "--bf16-ffn-arena-minimum-rows", "512",
        ])
    return result


def environment(policy: str) -> dict[str, str]:
    result = os.environ.copy()
    # Set both sides explicitly so a caller's shell cannot change the experiment.
    result["HIPBLASLT_PRELOAD_KERNELS"] = \
        "1" if policy == "bf16_preload_all" else "0"
    return result


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("hf_infer emitted no JSON")


def floats(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    return values


def error(reference: array.array, actual: array.array) -> tuple[float, float, bool]:
    if len(reference) != len(actual) or not reference:
        raise RuntimeError("cold-start complete-logit size changed")
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
    records: list[dict] = []
    outputs: dict[tuple[str, str, int], array.array] = {}
    with tempfile.TemporaryDirectory(prefix="microllm-hipblaslt-preload-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            for process_run in range(1, args.runs + 1):
                order = list(POLICIES)
                if process_run % 2 == 0:
                    order.reverse()
                for policy in order:
                    stem = f"{model['name']}-p{process_run}-{policy}"
                    logits = temporary / f"{stem}.bin"
                    started = time.perf_counter()
                    completed = subprocess.run(
                        command(args, model, policy, logits),
                        text=True, capture_output=True, check=False,
                        env=environment(policy))
                    process_wall_ms = (time.perf_counter() - started) * 1000.0
                    if completed.returncode != 0:
                        raise RuntimeError(completed.stdout + completed.stderr)
                    record = last_json(completed.stdout)
                    if record.get("status") != "pass":
                        raise RuntimeError(f"invalid cold-start record: {stem}")
                    record.update({
                        "record_type": "hipblaslt_preload_measurement",
                        "model": model["name"], "revision": model["revision"],
                        "policy": policy, "process_run": process_run,
                        "process_order": order,
                        "hipblaslt_preload_kernels":
                            1 if policy == "bf16_preload_all" else 0,
                        "process_wall_ms": process_wall_ms,
                    })
                    records.append(record)
                    outputs[(model["name"], policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        reference = outputs[(model["name"], "fp32", 1)]
        selected = [row for row in records if row["model"] == model["name"]]
        grouped = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in POLICIES
        }
        maximum = 0.0
        rms = 0.0
        finite = True
        for policy in POLICIES:
            for process_run in range(1, args.runs + 1):
                current = error(
                    reference, outputs[(model["name"], policy, process_run)])
                maximum = max(maximum, current[0])
                rms = max(rms, current[1])
                finite = finite and current[2]
        lazy_forward = median(grouped["bf16_lazy"], "forward_ms")
        preload_forward = median(grouped["bf16_preload_all"], "forward_ms")
        lazy_wall = median(grouped["bf16_lazy"], "process_wall_ms")
        preload_wall = median(grouped["bf16_preload_all"], "process_wall_ms")
        comparisons.append({
            "model": model["name"], "revision": model["revision"],
            "fp32_first_forward_ms": median(grouped["fp32"], "forward_ms"),
            "bf16_lazy_first_forward_ms": lazy_forward,
            "bf16_preload_first_forward_ms": preload_forward,
            "bf16_lazy_process_wall_ms": lazy_wall,
            "bf16_preload_process_wall_ms": preload_wall,
            "preload_forward_slowdown": preload_forward / lazy_forward,
            "preload_process_slowdown": preload_wall / lazy_wall,
            "bf16_lazy_peak_bytes": int(median(
                grouped["bf16_lazy"], "engine_peak_bytes")),
            "bf16_preload_peak_bytes": int(median(
                grouped["bf16_preload_all"], "engine_peak_bytes")),
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
    counterexample = all(
        row["preload_forward_slowdown"] >= args.minimum_preload_slowdown and
        row["preload_process_slowdown"] >= args.minimum_preload_slowdown
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and counterexample else "fail",
        "record_type": "hipblaslt_preload_summary",
        "raw_processes": len(records),
        "correctness_gate": correctness,
        "preload_counterexample_gate": counterexample,
        "minimum_preload_slowdown": args.minimum_preload_slowdown,
        "comparisons": comparisons,
        "decision": "reject all-kernel preload; retain targeted explicit prewarm",
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
