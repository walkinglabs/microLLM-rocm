#!/usr/bin/env python3
"""Measure lazy versus explicit BF16 grouped-QKV first-request setup."""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


POLICIES = ("baseline", "lazy", "prewarm")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--qwen-index", type=int, default=64713)
    parser.add_argument("--deepseek-index", type=int, default=64755)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=0.25)
    parser.add_argument("--rms-tolerance", type=float, default=0.05)
    result = parser.parse_args()
    if (result.runs <= 0 or result.sequence <= 0 or
            result.qwen_index < 0 or result.deepseek_index < 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("prewarm inputs are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    expected = {"qwen2.5-0.5b", "deepseek-r1-distill-qwen-1.5b"}
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != expected:
        raise RuntimeError("prewarm gate requires pinned Qwen and DeepSeek")
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(length)]


def index_for(args: argparse.Namespace, model: dict) -> int:
    return args.qwen_index if model["name"] == "qwen2.5-0.5b" \
        else args.deepseek_index


def command(args: argparse.Namespace, model: dict, policy: str,
            logits: Path) -> list[str]:
    tokens = repeated(model["inference"]["token_ids"], args.sequence)
    result = [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", "1",
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true", "--bf16-ffn-arena-minimum-rows", "512",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-logits", "last", "--logits-output", str(logits),
    ]
    if policy != "baseline":
        result.extend([
            "--bf16-qkv-arena", "true",
            "--bf16-qkv-arena-minimum-rows", "512",
            "--bf16-grouped-qkv-algorithm-index", str(index_for(args, model)),
        ])
    if policy == "prewarm":
        result.extend(["--bf16-grouped-qkv-prewarm", "true"])
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
    if len(reference) != len(actual):
        raise RuntimeError("prewarm complete-logit size changed")
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
    records = []
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="microllm-grouped-prewarm-") as temp:
        temporary = Path(temp)
        for model in selected_models:
            for process_run in range(1, args.runs + 1):
                order = list(POLICIES)
                if process_run % 2 == 0:
                    order.reverse()
                for policy in order:
                    stem = f"{model['name']}-p{process_run}-{policy}"
                    logits = temporary / f"{stem}.bin"
                    completed = subprocess.run(
                        command(args, model, policy, logits),
                        text=True, capture_output=True, check=False)
                    if completed.returncode != 0:
                        raise RuntimeError(completed.stdout + completed.stderr)
                    record = last_json(completed.stdout)
                    if record.get("status") != "pass":
                        raise RuntimeError(f"invalid prewarm record: {stem}")
                    blocks = 24 if model["name"] == "qwen2.5-0.5b" else 28
                    if policy == "lazy" and (
                            int(record["bf16_grouped_qkv_plan_hits"]) != 0 or
                            int(record["bf16_grouped_qkv_dispatches"]) != blocks):
                        raise RuntimeError("lazy policy did not build plans on request")
                    if policy == "prewarm" and (
                            int(record["bf16_grouped_qkv_plan_hits"]) != blocks or
                            int(record["bf16_grouped_qkv_dispatches"]) != 2 * blocks or
                            float(record["bf16_grouped_qkv_prewarm_ms"]) <= 0):
                        raise RuntimeError("explicit prewarm did not move plan setup")
                    record.update({
                        "record_type": "bf16_grouped_qkv_prewarm_measurement",
                        "model": model["name"], "revision": model["revision"],
                        "policy": policy, "process_run": process_run,
                        "process_order": order,
                    })
                    records.append(record)
                    outputs[(model["name"], policy, process_run)] = floats(logits)

    comparisons = []
    for model in selected_models:
        reference = outputs[(model["name"], "baseline", 1)]
        selected = [row for row in records if row["model"] == model["name"]]
        grouped = {policy: [row for row in selected if row["policy"] == policy]
                   for policy in POLICIES}
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
        prewarm_ms = median(grouped["prewarm"], "bf16_grouped_qkv_prewarm_ms")
        first_ms = median(grouped["prewarm"], "forward_ms")
        comparisons.append({
            "model": model["name"], "revision": model["revision"],
            "baseline_first_ms": median(grouped["baseline"], "forward_ms"),
            "lazy_first_ms": median(grouped["lazy"], "forward_ms"),
            "prewarm_ms": prewarm_ms,
            "prewarmed_first_ms": first_ms,
            "prewarm_plus_first_ms": prewarm_ms + first_ms,
            "kernel_setup_ms": median(
                grouped["prewarm"], "bf16_grouped_qkv_prewarm_kernel_ms"),
            "argument_setup_ms": median(
                grouped["prewarm"], "bf16_grouped_qkv_prewarm_arguments_ms"),
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
    moved = all(row["prewarmed_first_ms"] < row["lazy_first_ms"] and
                row["prewarm_plus_first_ms"] >= row["prewarmed_first_ms"]
                for row in comparisons)
    summary = {
        "schema_version": 1, "status": "pass" if correctness and moved else "fail",
        "record_type": "bf16_grouped_qkv_prewarm_summary",
        "raw_processes": len(records), "correctness_gate": correctness,
        "setup_moved_before_request": moved, "comparisons": comparisons,
        "decision": "keep explicit prewarm API; default remains off",
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
