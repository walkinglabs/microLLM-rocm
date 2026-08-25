#!/usr/bin/env python3
"""Audit complete cached logits across batch sizes and decode steps."""

from __future__ import annotations

import argparse
from array import array
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def csv_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("values must be unique nonnegative integers")
    return result


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batches", type=csv_ints, default=[1, 2, 4, 8])
    parser.add_argument("--decode-steps", type=csv_ints, default=[0, 1, 2])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not args.manifest.is_file() or not args.binary.is_file() or
            args.context <= 0 or args.batches != [1, 2, 4, 8] or
            args.decode_steps != [0, 1, 2] or args.runs < 2 or
            args.warmup < 0 or args.timeout_seconds <= 0):
        parser.error("cross-batch audit inputs are outside the fixed contract")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def model_entry(path: Path, name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("models", []) if document.get("schema_version") == 1 else []
    selected = [row for row in rows if row.get("name") == name]
    if len(selected) != 1 or not selected[0].get("inference", {}).get("token_ids"):
        raise ValueError("audit requires one complete pinned model")
    return selected[0]


def expanded(seed: list[int], length: int) -> str:
    return ",".join(str(seed[index % len(seed)]) for index in range(length))


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("inference process emitted no JSON")


def read_logits(path: Path, batch: int, vocabulary: int) -> list[float]:
    values = array("f")
    with path.open("rb") as stream:
        values.frombytes(stream.read())
    expected = batch * vocabulary
    if len(values) != expected or any(not math.isfinite(value) for value in values):
        raise ValueError("cached logits file is incomplete or non-finite")
    return list(values)


def error(left: list[float], right: list[float]) -> tuple[float, float, bool]:
    if len(left) != len(right) or not left:
        raise ValueError("logit comparison shape changed")
    differences = [abs(a - b) for a, b in zip(left, right)]
    return (max(differences),
            math.sqrt(sum(value * value for value in differences) / len(differences)),
            left == right)


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def command(args: argparse.Namespace, model: dict, batch: int, step: int,
            output: Path) -> list[str]:
    new_tokens = step + 1
    return [
        str(args.binary), "--config", model["config"], "--weights", model["weights"],
        "--tokens", expanded(model["inference"]["token_ids"], args.context),
        "--device", "hip", "--top-k", "1", "--batch", str(batch),
        "--use-cache", "true", "--cache-prefill-mode", "full",
        "--decode-mode", "steady", "--batch-argmax-mode", "device",
        "--prefill-logits", "last", "--kv-cache-dtype", args.cache_dtype,
        "--cache-capacity", str(args.context + new_tokens),
        "--new-tokens", str(new_tokens), "--warmup", str(args.warmup),
        "--steps", "1", "--prefill-warmup", str(args.warmup),
        "--prefill-steps", "1", "--bf16-ffn", "true",
        "--bf16-attention", "true", "--workload", "decode",
        "--cache-logits-output", str(output), "--cache-logits-step", str(step),
    ]


def run_one(args: argparse.Namespace, model: dict, vocabulary: int, batch: int,
            step: int, run: int, temporary: Path) -> tuple[dict, list[float]]:
    logits_path = temporary / f"b{batch}-s{step}-r{run}.bin"
    completed = subprocess.run(
        command(args, model, batch, step, logits_path), text=True,
        capture_output=True, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    record = last_json(completed.stdout)
    required = {
        "status": "pass", "batch": batch, "token_count": args.context,
        "decode_tokens": step + 1, "warmup": args.warmup, "steps": 1,
        "use_cache": True, "decode_mode": "steady",
        "decode_step_semantics": "one_model_forward_per_measured_token",
        "kv_cache_dtype": args.cache_dtype, "cache_logits_step": step,
        "cached_attention_materialized_policy": "auto-enabled",
        "cached_attention_materialized_scores": True,
        "cached_attention_pv_splits": 0,
    }
    for name, wanted in required.items():
        if record.get(name) != wanted:
            raise ValueError(f"{name} expected {wanted!r}, got {record.get(name)!r}")
    if len(record.get("generated_tokens", [])) != step + 1:
        raise ValueError("generated-token evidence is incomplete")
    logits = read_logits(logits_path, batch, vocabulary)
    rows = [logits[index * vocabulary:(index + 1) * vocabulary]
            for index in range(batch)]
    within = [error(rows[0], row) for row in rows[1:]]
    host_tokens = [argmax(row) for row in rows]
    device_token = int(record["generated_tokens"][step])
    record.update({
        "schema_version": 1,
        "record_type": "cached_cross_batch_logit_measurement",
        "model": args.model,
        "revision": model["revision"],
        "context": args.context,
        "decode_step": step,
        "process_run": run,
        "complete_logit_elements": len(logits),
        "complete_logit_rows": batch,
        "within_batch_maximum_error": max((item[0] for item in within), default=0.0),
        "within_batch_rms_error": max((item[1] for item in within), default=0.0),
        "within_batch_bitwise_equal": all(item[2] for item in within),
        "host_argmax_tokens": host_tokens,
        "device_argmax_token": device_token,
        "host_device_argmax_equal": all(token == device_token for token in host_tokens),
    })
    return record, logits


def summarize(records: list[tuple[dict, list[float]]], vocabulary: int) -> dict:
    by_key = {(row["batch"], row["decode_step"], row["process_run"]): (row, logits)
              for row, logits in records}
    cases = []
    for step in (0, 1, 2):
        reference = by_key[(1, step, 1)][1]
        for batch in (1, 2, 4, 8):
            rows = [by_key[(batch, step, run)] for run in (1, 2)]
            first_rows = [logits[:vocabulary] for _, logits in rows]
            cross = [error(reference, values) for values in first_rows]
            repeat = error(first_rows[0], first_rows[1])
            cases.append({
                "batch": batch, "decode_step": step, "runs": 2,
                "complete_values_compared_per_run": vocabulary,
                "cross_batch_maximum_error": max(item[0] for item in cross),
                "cross_batch_maximum_rms_error": max(item[1] for item in cross),
                "cross_batch_bitwise_equal": all(item[2] for item in cross),
                "repeat_maximum_error": repeat[0],
                "repeat_rms_error": repeat[1],
                "repeat_bitwise_equal": repeat[2],
                "within_batch_bitwise_equal": all(
                    row["within_batch_bitwise_equal"] for row, _ in rows),
                "host_device_argmax_equal": all(
                    row["host_device_argmax_equal"] for row, _ in rows),
                "device_argmax_tokens": [row["device_argmax_token"] for row, _ in rows],
            })
    return {
        "schema_version": 1,
        "record_type": "cached_cross_batch_logit_audit",
        "status": "pass",
        "process_rows": len(records),
        "case_rows": len(cases),
        "batches": [1, 2, 4, 8],
        "decode_steps": [0, 1, 2],
        "runs_per_case": 2,
        "vocabulary_size": vocabulary,
        "all_repeat_bitwise_equal": all(case["repeat_bitwise_equal"] for case in cases),
        "all_within_batch_bitwise_equal": all(
            case["within_batch_bitwise_equal"] for case in cases),
        "all_host_device_argmax_equal": all(
            case["host_device_argmax_equal"] for case in cases),
        "all_cross_batch_bitwise_equal": all(
            case["cross_batch_bitwise_equal"] for case in cases),
        "maximum_cross_batch_error": max(
            case["cross_batch_maximum_error"] for case in cases),
        "maximum_cross_batch_rms_error": max(
            case["cross_batch_maximum_rms_error"] for case in cases),
        "first_non_bitwise_step": next((
            case["decode_step"] for case in cases
            if not case["cross_batch_bitwise_equal"]), -1),
        "cases": cases,
    }


def render(summary: dict) -> str:
    width, height = 1200, 520
    maximum = max(case["cross_batch_maximum_error"] for case in summary["cases"])
    scale = 650.0 / maximum if maximum else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#e5e7eb}'
        '.title{font-size:22px;font-weight:700}.label{font-size:13px}'
        '.muted{fill:#94a3b8;font-size:12px}</style>',
        '<text x="30" y="38" class="title">DeepSeek microLLM cross-batch complete logits</text>',
        '<text x="30" y="62" class="muted">B1 reference · step 0/1/2 · every row and host/device argmax checked</text>',
    ]
    for index, case in enumerate(summary["cases"]):
        y = 88 + index * 33
        length = max(2.0, case["cross_batch_maximum_error"] * scale)
        color = "#22c55e" if case["cross_batch_bitwise_equal"] else "#ef4444"
        parts.extend((
            f'<text x="30" y="{y + 17}" class="label">S{case["decode_step"]} B{case["batch"]}</text>',
            f'<rect x="120" y="{y}" width="{length:.2f}" height="22" rx="4" fill="{color}"/>',
            f'<text x="{140 + length:.2f}" y="{y + 16}" class="label">'
            f'Max {case["cross_batch_maximum_error"]:.3e} · '
            f'RMS {case["cross_batch_maximum_rms_error"]:.3e}</text>',
        ))
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    args = options()
    model = model_entry(args.manifest, args.model)
    config = json.loads(Path(model["config"]).read_text(encoding="utf-8"))
    vocabulary = int(config["vocab_size"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    measurements = []
    with tempfile.TemporaryDirectory(prefix="microllm-cross-batch-logits-") as directory:
        temporary = Path(directory)
        for run in range(1, args.runs + 1):
            order = args.batches if run % 2 else list(reversed(args.batches))
            for step in args.decode_steps:
                for batch in order:
                    row = run_one(args, model, vocabulary, batch, step, run,
                                  temporary)
                    measurements.append(row)
                    print(json.dumps({
                        "batch": batch, "decode_step": step,
                        "process_run": run,
                        "within_batch_bitwise_equal": row[0][
                            "within_batch_bitwise_equal"],
                        "host_device_argmax_equal": row[0][
                            "host_device_argmax_equal"],
                    }, sort_keys=True), flush=True)
    summary = summarize(measurements, vocabulary)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n"
                for row, _ in measurements), encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_directory / "cross-batch.svg").write_text(
        render(summary), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"audit_cached_cross_batch_logits: {error}", file=sys.stderr)
        raise SystemExit(2) from error
