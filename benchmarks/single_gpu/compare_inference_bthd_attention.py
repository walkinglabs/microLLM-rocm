#!/usr/bin/env python3
"""Gate the inference BTHD Attention island on official composed models."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


POLICIES = ("baseline", "bthd")
INDICES = {
    "qwen2.5-0.5b": (64713, 65168),
    "deepseek-r1-distill-qwen-1.5b": (64755, 65200),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.0)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.sequence < 256 or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.minimum_speedup <= 1 or result.maximum_peak_ratio <= 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("BTHD Attention options are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != set(INDICES):
        raise RuntimeError("BTHD Attention gate requires pinned official models")
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(length)]


def command(args: argparse.Namespace, model: dict, policy: str,
            diagnostics: bool, logits: Path | None = None) -> list[str]:
    tokens = repeated(model["inference"]["token_ids"], args.sequence)
    qkv_index, gate_up_index = INDICES[model["name"]]
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", "1",
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true",
        "--bf16-ffn-arena-minimum-rows", str(args.sequence),
        "--bf16-qkv-arena", "true",
        "--bf16-qkv-arena-minimum-rows", str(args.sequence),
        "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
        "--bf16-grouped-gate-up-algorithm-index", str(gate_up_index),
        "--inference-bthd-attention",
        "true" if policy == "bthd" else "false",
        "--strided-copy-diagnostics",
        "true" if diagnostics else "false",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0" if diagnostics else str(args.warmup),
        "--prefill-steps", "1" if diagnostics else str(args.steps),
        "--prefill-logits", "last",
    ]
    if logits is not None:
        result.extend(["--logits-output", str(logits)])
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
        raise RuntimeError("BTHD complete-logit size changed")
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
    performance_records = []
    diagnostic_records = []
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="microllm-bthd-attention-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            blocks = 24 if model["name"] == "qwen2.5-0.5b" else 28
            for process_run in range(1, args.runs + 1):
                order = list(POLICIES)
                if process_run % 2 == 0:
                    order.reverse()
                for policy in order:
                    stem = f"{model['name']}-p{process_run}-{policy}"
                    logits = temporary / f"{stem}.bin"
                    completed = subprocess.run(
                        command(args, model, policy, False, logits),
                        text=True, capture_output=True, check=False)
                    if completed.returncode != 0:
                        raise RuntimeError(completed.stdout + completed.stderr)
                    record = last_json(completed.stdout)
                    expected_dispatches = blocks * (args.warmup + args.steps)
                    if record.get("status") != "pass" or \
                            record.get("inference_bthd_attention") is not (
                                policy == "bthd") or \
                            int(record["bf16_grouped_qkv_dispatches"]) != \
                                expected_dispatches or \
                            int(record[
                                "bf16_grouped_gate_up_dispatches"]) != \
                                expected_dispatches:
                        raise RuntimeError("invalid BTHD performance record")
                    record.update({
                        "record_type":
                            "inference_bthd_attention_performance",
                        "model": model["name"],
                        "revision": model["revision"],
                        "policy": policy,
                        "process_run": process_run,
                        "process_order": order,
                    })
                    performance_records.append(record)
                    outputs[(model["name"], policy, process_run)] = floats(logits)

                    diagnostic = subprocess.run(
                        command(args, model, policy, True),
                        text=True, capture_output=True, check=False)
                    if diagnostic.returncode != 0:
                        raise RuntimeError(diagnostic.stdout + diagnostic.stderr)
                    diag_record = last_json(diagnostic.stdout)
                    expected_copies = (
                        0 if policy == "bthd" else blocks * 4)
                    if diag_record.get("status") != "pass" or \
                            int(diag_record["strided_copy_calls"]) != \
                                expected_copies:
                        raise RuntimeError("invalid BTHD diagnostic record")
                    diag_record.update({
                        "record_type":
                            "inference_bthd_attention_diagnostic",
                        "model": model["name"],
                        "revision": model["revision"],
                        "policy": policy,
                        "process_run": process_run,
                        "process_order": order,
                    })
                    diagnostic_records.append(diag_record)

    comparisons = []
    for model in selected_models:
        name = model["name"]
        reference = outputs[(name, "baseline", 1)]
        selected = [row for row in performance_records if row["model"] == name]
        grouped = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in POLICIES
        }
        diagnostics = {
            policy: [row for row in diagnostic_records
                     if row["model"] == name and row["policy"] == policy]
            for policy in POLICIES
        }
        maximum = 0.0
        rms = 0.0
        finite = True
        for policy in POLICIES:
            for process_run in range(1, args.runs + 1):
                current = error(
                    reference, outputs[(name, policy, process_run)])
                maximum = max(maximum, current[0])
                rms = max(rms, current[1])
                finite = finite and current[2]
        baseline_tps = median(
            grouped["baseline"], "prefill_tokens_per_second")
        bthd_tps = median(
            grouped["bthd"], "prefill_tokens_per_second")
        baseline_peak = int(median(
            grouped["baseline"], "engine_peak_bytes"))
        bthd_peak = int(median(
            grouped["bthd"], "engine_peak_bytes"))
        comparisons.append({
            "model": name,
            "revision": model["revision"],
            "baseline_tokens_per_second": baseline_tps,
            "bthd_tokens_per_second": bthd_tps,
            "speedup": bthd_tps / baseline_tps,
            "baseline_peak_bytes": baseline_peak,
            "bthd_peak_bytes": bthd_peak,
            "peak_ratio": bthd_peak / baseline_peak,
            "peak_bytes_saved": baseline_peak - bthd_peak,
            "baseline_strided_calls": int(median(
                diagnostics["baseline"], "strided_copy_calls")),
            "bthd_strided_calls": int(median(
                diagnostics["bthd"], "strided_copy_calls")),
            "baseline_strided_bytes": int(median(
                diagnostics["baseline"], "strided_copy_bytes")),
            "bthd_strided_bytes": int(median(
                diagnostics["bthd"], "strided_copy_bytes")),
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
    copies = all(
        row["baseline_strided_calls"] > 0 and
        row["baseline_strided_bytes"] > 0 and
        row["bthd_strided_calls"] == 0 and
        row["bthd_strided_bytes"] == 0
        for row in comparisons)
    performance = all(
        row["speedup"] >= args.minimum_speedup for row in comparisons)
    memory = all(
        row["peak_ratio"] <= args.maximum_peak_ratio
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and copies and memory else "fail",
        "record_type": "inference_bthd_attention_summary",
        "performance_processes": len(performance_records),
        "diagnostic_processes": len(diagnostic_records),
        "correctness_gate": correctness,
        "copy_elimination_gate": copies,
        "performance_gate": performance,
        "memory_gate": memory,
        "comparisons": comparisons,
        "decision": (
            "keep explicit inference BTHD Attention policy"
            if correctness and copies and performance and memory else
            "reject inference BTHD Attention policy"),
    }
    for name, rows in (
            ("performance-raw.jsonl", performance_records),
            ("diagnostic-raw.jsonl", diagnostic_records)):
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
