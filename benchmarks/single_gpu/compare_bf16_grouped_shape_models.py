#!/usr/bin/env python3
"""Gate composed grouped policies across sequence and batch workloads."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


CASES = (
    ("b1t256", 1, 256, 256),
    ("b1t1024", 1, 1024, 1024),
    ("b2t512", 2, 512, 1024),
)
POLICIES = ("baseline", "both")
INDICES = {
    ("qwen2.5-0.5b", 256): (64713, 65197),
    ("deepseek-r1-distill-qwen-1.5b", 256): (64713, 65168),
    ("qwen2.5-0.5b", 1024): (64755, 65200),
    ("deepseek-r1-distill-qwen-1.5b", 1024): (64755, 65212),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=0.25)
    parser.add_argument("--rms-tolerance", type=float, default=0.05)
    parser.add_argument("--minimum-speedup", type=float, default=1.01)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.01)
    parser.add_argument("--maximum-combined-setup-ms", type=float, default=250.0)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.minimum_speedup <= 1 or result.maximum_peak_ratio < 1 or
            result.maximum_combined_setup_ms < 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("grouped shape model options are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("grouped shape model gate requires pinned models")
    for model in result:
        config = Path(model["config"])
        if not config.is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
        external = json.loads(config.read_text(encoding="utf-8"))
        model["vocabulary_size"] = int(external["vocab_size"])
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise RuntimeError("grouped shape token seed cannot be empty")
    return [seed[index % len(seed)] for index in range(length)]


def command(args: argparse.Namespace, model: dict, case: tuple,
            policy: str, logits: Path) -> list[str]:
    _, batch, sequence, rows = case
    tokens = repeated(model["inference"]["token_ids"], sequence)
    qkv_index, gate_up_index = INDICES[(model["name"], rows)]
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", str(batch),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true",
        "--bf16-ffn-arena-minimum-rows", str(rows),
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", str(args.warmup),
        "--prefill-steps", str(args.steps),
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]
    if policy == "both":
        result.extend([
            "--bf16-qkv-arena", "true",
            "--bf16-qkv-arena-minimum-rows", str(rows),
            "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
            "--bf16-grouped-gate-up-algorithm-index", str(gate_up_index),
        ])
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
        raise RuntimeError("grouped shape complete-logit size changed")
    maximum = 0.0
    squared = 0.0
    finite = True
    for expected, observed in zip(reference, actual, strict=True):
        difference = abs(expected - observed)
        maximum = max(maximum, difference)
        squared += difference * difference
        finite = finite and math.isfinite(observed)
    return maximum, math.sqrt(squared / len(reference)), finite


def row_top_indices(values: array.array, batch: int,
                    vocabulary_size: int) -> list[int]:
    if len(values) != batch * vocabulary_size:
        raise RuntimeError("grouped shape logits do not match batch/vocabulary")
    return [
        max(range(vocabulary_size),
            key=lambda index: values[row * vocabulary_size + index])
        for row in range(batch)
    ]


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    outputs = {}
    with tempfile.TemporaryDirectory(
            prefix="microllm-grouped-shape-model-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            blocks = 24 if model["name"] == "qwen2.5-0.5b" else 28
            forwards = args.warmup + args.steps
            for process_run in range(1, args.runs + 1):
                cases = list(CASES)
                policies = list(POLICIES)
                if process_run % 2 == 0:
                    cases.reverse()
                    policies.reverse()
                for case in cases:
                    case_name, batch, sequence, rows = case
                    for policy in policies:
                        stem = (f"{model['name']}-p{process_run}-"
                                f"{case_name}-{policy}")
                        logits = temporary / f"{stem}.bin"
                        completed = subprocess.run(
                            command(args, model, case, policy, logits),
                            text=True, capture_output=True, check=False)
                        if completed.returncode != 0:
                            raise RuntimeError(
                                completed.stdout + completed.stderr)
                        record = last_json(completed.stdout)
                        if record.get("status") != "pass":
                            raise RuntimeError(f"invalid shape record: {stem}")
                        expected_dispatches = blocks * forwards
                        if int(record["bf16_grouped_qkv_dispatches"]) != (
                                expected_dispatches
                                if policy == "both" else 0) or \
                                int(record[
                                    "bf16_grouped_gate_up_dispatches"]) != (
                                        expected_dispatches
                                        if policy == "both" else 0):
                            raise RuntimeError(
                                "grouped shape dispatch mismatch")
                        qkv_index, gate_up_index = INDICES[
                            (model["name"], rows)]
                        record.update({
                            "record_type":
                                "bf16_grouped_shape_model_measurement",
                            "model": model["name"],
                            "revision": model["revision"],
                            "case": case_name, "batch": batch,
                            "sequence": sequence, "rows": rows,
                            "policy": policy,
                            "process_run": process_run,
                            "case_order": [item[0] for item in cases],
                            "policy_order": policies,
                            "qkv_solution_index":
                                qkv_index if policy == "both" else -1,
                            "gate_up_solution_index":
                                gate_up_index if policy == "both" else -1,
                        })
                        records.append(record)
                        outputs[(model["name"], case_name,
                                 policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        for case_name, batch, sequence, rows in CASES:
            reference = outputs[(model["name"], case_name, "baseline", 1)]
            selected = [
                row for row in records
                if row["model"] == model["name"] and
                row["case"] == case_name]
            grouped = {
                policy: [row for row in selected if row["policy"] == policy]
                for policy in POLICIES
            }
            maximum = 0.0
            rms = 0.0
            finite = True
            top_rows_equal = True
            reference_top = row_top_indices(
                reference, batch, model["vocabulary_size"])
            for policy in POLICIES:
                for process_run in range(1, args.runs + 1):
                    actual = outputs[(model["name"], case_name,
                                      policy, process_run)]
                    current = error(reference, actual)
                    maximum = max(maximum, current[0])
                    rms = max(rms, current[1])
                    finite = finite and current[2]
                    top_rows_equal = (
                        top_rows_equal and reference_top ==
                        row_top_indices(
                            actual, batch, model["vocabulary_size"]))
            baseline_tps = median(
                grouped["baseline"], "prefill_tokens_per_second")
            both_tps = median(
                grouped["both"], "prefill_tokens_per_second")
            baseline_peak = int(median(
                grouped["baseline"], "engine_peak_bytes"))
            both_peak = int(median(
                grouped["both"], "engine_peak_bytes"))
            qkv_index, gate_up_index = INDICES[(model["name"], rows)]
            comparisons.append({
                "model": model["name"],
                "revision": model["revision"],
                "case": case_name, "batch": batch,
                "sequence": sequence, "rows": rows,
                "qkv_solution_index": qkv_index,
                "gate_up_solution_index": gate_up_index,
                "baseline_tokens_per_second": baseline_tps,
                "both_tokens_per_second": both_tps,
                "speedup": both_tps / baseline_tps,
                "baseline_peak_bytes": baseline_peak,
                "both_peak_bytes": both_peak,
                "peak_ratio": both_peak / baseline_peak,
                "combined_kernel_setup_ms": median(
                    grouped["both"],
                    "bf16_grouped_qkv_kernel_setup_ms") +
                    median(grouped["both"],
                           "bf16_grouped_gate_up_kernel_setup_ms"),
                "maximum_absolute_logit_difference": maximum,
                "maximum_rms_logit_difference": rms,
                "finite_complete_logits": finite,
                "top_rows_equal": top_rows_equal,
            })

    correctness = all(
        row["finite_complete_logits"] and row["top_rows_equal"] and
        row["maximum_absolute_logit_difference"] <=
            args.maximum_absolute_tolerance and
        row["maximum_rms_logit_difference"] <= args.rms_tolerance
        for row in comparisons)
    performance = all(
        row["speedup"] >= args.minimum_speedup for row in comparisons)
    memory = all(
        row["peak_ratio"] <= args.maximum_peak_ratio
        for row in comparisons)
    setup = all(
        row["combined_kernel_setup_ms"] <=
            args.maximum_combined_setup_ms
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and memory else "fail",
        "record_type": "bf16_grouped_shape_model_summary",
        "raw_processes": len(records),
        "correctness_gate": correctness,
        "performance_gate": performance,
        "memory_gate": memory,
        "setup_gate": setup,
        "comparisons": comparisons,
        "decision": (
            "keep explicit rows 256/1024 composed policies"
            if correctness and performance and memory and setup else
            "retain only individually passing shape cases"),
    }
    with (args.output_directory / "raw.jsonl").open(
            "w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
