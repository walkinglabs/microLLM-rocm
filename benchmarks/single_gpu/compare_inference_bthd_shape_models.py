#!/usr/bin/env python3
"""Gate inference BTHD Attention across sequence and batch cases."""

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
POLICIES = ("baseline", "bthd")
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
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--rms-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--maximum-peak-ratio", type=float, default=1.0)
    result = parser.parse_args()
    if (result.runs <= 0 or result.warmup < 0 or result.steps <= 0 or
            result.maximum_absolute_tolerance < 0 or result.rms_tolerance < 0 or
            result.minimum_speedup <= 1 or result.maximum_peak_ratio <= 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("BTHD shape model options are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("BTHD shape gate requires pinned official models")
    for model in result:
        external = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
        model["vocabulary_size"] = int(external["vocab_size"])
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(length)]


def command(args: argparse.Namespace, model: dict, case: tuple,
            policy: str, diagnostics: bool,
            logits: Path | None = None) -> list[str]:
    _, batch, sequence, rows = case
    qkv_index, gate_up_index = INDICES[(model["name"], rows)]
    tokens = repeated(model["inference"]["token_ids"], sequence)
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", str(batch),
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true",
        "--bf16-ffn-arena-minimum-rows", str(rows),
        "--bf16-qkv-arena", "true",
        "--bf16-qkv-arena-minimum-rows", str(rows),
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
        raise RuntimeError("BTHD shape complete-logit size changed")
    differences = [abs(expected - observed)
                   for expected, observed in zip(reference, actual, strict=True)]
    return (max(differences),
            math.sqrt(sum(value * value for value in differences) /
                      len(differences)),
            all(math.isfinite(value) for value in actual))


def row_top(values: array.array, batch: int, vocabulary: int) -> list[int]:
    if len(values) != batch * vocabulary:
        raise RuntimeError("BTHD shape logits do not match batch")
    return [
        max(range(vocabulary),
            key=lambda index: values[row * vocabulary + index])
        for row in range(batch)
    ]


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    performance_records = []
    diagnostic_records = []
    outputs = {}
    with tempfile.TemporaryDirectory(
            prefix="microllm-bthd-shape-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            for process_run in range(1, args.runs + 1):
                cases = list(CASES)
                policies = list(POLICIES)
                if process_run % 2 == 0:
                    cases.reverse()
                    policies.reverse()
                for case in cases:
                    for policy in policies:
                        stem = (f"{model['name']}-p{process_run}-"
                                f"{case[0]}-{policy}")
                        logits = temporary / f"{stem}.bin"
                        completed = subprocess.run(
                            command(args, model, case, policy, False, logits),
                            text=True, capture_output=True, check=False)
                        if completed.returncode != 0:
                            raise RuntimeError(
                                completed.stdout + completed.stderr)
                        record = last_json(completed.stdout)
                        if record.get("status") != "pass":
                            raise RuntimeError("invalid BTHD shape record")
                        record.update({
                            "record_type":
                                "inference_bthd_shape_performance",
                            "model": model["name"],
                            "revision": model["revision"],
                            "case": case[0], "batch": case[1],
                            "sequence": case[2], "rows": case[3],
                            "policy": policy,
                            "process_run": process_run,
                        })
                        performance_records.append(record)
                        outputs[(model["name"], case[0],
                                 policy, process_run)] = floats(logits)

            for case in CASES:
                diagnostic = subprocess.run(
                    command(args, model, case, "bthd", True),
                    text=True, capture_output=True, check=False)
                if diagnostic.returncode != 0:
                    raise RuntimeError(
                        diagnostic.stdout + diagnostic.stderr)
                record = last_json(diagnostic.stdout)
                attention_records = [
                    item for item in record.get(
                        "strided_copy_records", [])
                    if item.get("source") in (
                        "attention.layout", "attention.core")]
                attention_calls = sum(
                    int(item["calls"]) for item in attention_records)
                attention_bytes = sum(
                    int(item["bytes"]) for item in attention_records)
                if record.get("status") != "pass" or \
                        attention_calls != 0 or attention_bytes != 0:
                    raise RuntimeError(
                        "BTHD shape candidate retained Attention copies")
                record.update({
                    "record_type": "inference_bthd_shape_diagnostic",
                    "model": model["name"],
                    "revision": model["revision"],
                    "case": case[0], "batch": case[1],
                    "sequence": case[2], "rows": case[3],
                    "attention_strided_calls": attention_calls,
                    "attention_strided_bytes": attention_bytes,
                    "residual_strided_calls":
                        int(record["strided_copy_calls"]),
                    "residual_strided_bytes":
                        int(record["strided_copy_bytes"]),
                })
                diagnostic_records.append(record)

    comparisons = []
    for model in selected_models:
        for case_name, batch, sequence, rows in CASES:
            reference = outputs[(model["name"], case_name, "baseline", 1)]
            reference_top = row_top(
                reference, batch, model["vocabulary_size"])
            selected = [
                row for row in performance_records
                if row["model"] == model["name"] and
                row["case"] == case_name]
            grouped = {
                policy: [row for row in selected if row["policy"] == policy]
                for policy in POLICIES
            }
            maximum = 0.0
            rms = 0.0
            finite = True
            top_equal = True
            for policy in POLICIES:
                for process_run in range(1, args.runs + 1):
                    actual = outputs[(model["name"], case_name,
                                      policy, process_run)]
                    current = error(reference, actual)
                    maximum = max(maximum, current[0])
                    rms = max(rms, current[1])
                    finite = finite and current[2]
                    top_equal = (
                        top_equal and reference_top ==
                        row_top(actual, batch, model["vocabulary_size"]))
            baseline_tps = median(
                grouped["baseline"], "prefill_tokens_per_second")
            bthd_tps = median(
                grouped["bthd"], "prefill_tokens_per_second")
            baseline_peak = int(median(
                grouped["baseline"], "engine_peak_bytes"))
            bthd_peak = int(median(
                grouped["bthd"], "engine_peak_bytes"))
            comparisons.append({
                "model": model["name"], "revision": model["revision"],
                "case": case_name, "batch": batch,
                "sequence": sequence, "rows": rows,
                "baseline_tokens_per_second": baseline_tps,
                "bthd_tokens_per_second": bthd_tps,
                "speedup": bthd_tps / baseline_tps,
                "baseline_peak_bytes": baseline_peak,
                "bthd_peak_bytes": bthd_peak,
                "peak_ratio": bthd_peak / baseline_peak,
                "peak_bytes_saved": baseline_peak - bthd_peak,
                "bthd_attention_strided_calls": next(
                    int(item["attention_strided_calls"])
                    for item in diagnostic_records
                    if item["model"] == model["name"] and
                    item["case"] == case_name),
                "bthd_residual_strided_calls": next(
                    int(item["residual_strided_calls"])
                    for item in diagnostic_records
                    if item["model"] == model["name"] and
                    item["case"] == case_name),
                "bthd_residual_strided_bytes": next(
                    int(item["residual_strided_bytes"])
                    for item in diagnostic_records
                    if item["model"] == model["name"] and
                    item["case"] == case_name),
                "maximum_absolute_logit_difference": maximum,
                "maximum_rms_logit_difference": rms,
                "finite_complete_logits": finite,
                "top_rows_equal": top_equal,
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
    copies = len(diagnostic_records) == 6 and all(
        int(row["attention_strided_calls"]) == 0 and
        int(row["attention_strided_bytes"]) == 0
        for row in diagnostic_records)
    summary = {
        "schema_version": 1,
        "status": "pass" if correctness and memory and copies else "fail",
        "record_type": "inference_bthd_shape_model_summary",
        "performance_processes": len(performance_records),
        "diagnostic_processes": len(diagnostic_records),
        "correctness_gate": correctness,
        "performance_gate": performance,
        "memory_gate": memory,
        "copy_elimination_gate": copies,
        "comparisons": comparisons,
        "decision": (
            "keep explicit BTHD policy for measured sequence/batch cases"
            if correctness and performance and memory and copies else
            "retain only individually passing BTHD cases"),
    }
    for filename, rows in (
            ("performance-raw.jsonl", performance_records),
            ("diagnostic-raw.jsonl", diagnostic_records)):
        with (args.output_directory / filename).open(
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
