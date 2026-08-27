#!/usr/bin/env python3
"""Run exact-length Qwen3 natural prompts through the official shape worker."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SHAPE_RUNNER = Path(__file__).with_name("hf_inference_shape_matrix.py")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--micro-binary", type=Path, required=True)
    parser.add_argument("--pytorch-python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-amdsmi-fallback", action="store_true")
    args = parser.parse_args()
    for path in (args.manifest, args.micro_binary, args.pytorch_python, SHAPE_RUNNER):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if (args.warmup < 0 or args.steps <= 0 or args.runs <= 0 or
            args.decode_tokens <= 0 or args.timeout_seconds <= 0):
        parser.error("warmup must be nonnegative and all other counts positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("output directory must be empty")
    return args


def load_prompts(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    models = document.get("models") if document.get("schema_version") == 1 else None
    if not isinstance(models, list) or len(models) != 4:
        raise RuntimeError("exact prompt manifest must contain four models")
    families = set()
    for model in models:
        inference = model.get("inference", {})
        tokens = inference.get("token_ids")
        context = inference.get("exact_context")
        family = inference.get("prompt_family")
        if (not isinstance(tokens, list) or not tokens or context != len(tokens) or
                family not in {"english", "chinese", "code", "chat"} or
                family in families):
            raise RuntimeError("exact prompt metadata is incomplete or inconsistent")
        families.add(family)
    return models


def aggregate(models: list[dict], summaries: list[dict], records: list[dict]) -> dict:
    if len(models) != 4 or len(summaries) != 4:
        raise RuntimeError("exact prompt matrix needs four complete submatrices")
    rows = []
    prompt_rows = []
    for model, summary in zip(models, summaries):
        expected_context = model["inference"]["exact_context"]
        subrows = summary.get("rows", [])
        if (len(subrows) != 4 or
                any(row.get("model") != model["name"] or
                    row.get("context") != expected_context for row in subrows)):
            raise RuntimeError("exact prompt submatrix changed its model/context contract")
        for row in subrows:
            rows.append({**row,
                         "prompt_family": model["inference"]["prompt_family"],
                         "prompt_text": model["inference"]["prompt_text"]})
        prompt_rows.append({
            "model": model["name"],
            "family": model["inference"]["prompt_family"],
            "context": expected_context,
            "pass_rows": sum(row["status"] == "pass" for row in subrows),
            "precision_mismatch_rows": sum(
                row["status"] == "precision_mismatch" for row in subrows),
            "batch_invariance_mismatch_rows": sum(
                row["status"] == "batch_invariance_mismatch" for row in subrows),
            "limited_rows": sum(row["status"] == "limited" for row in subrows),
        })
    worker_passes = sum(record.get("status") == "pass" for record in records)
    return {
        "schema_version": 1,
        "record_type": "qwen3_exact_prompt_matrix",
        "status": ("pass" if all(row["status"] == "pass" for row in rows)
                   else "complete_with_recorded_limits"),
        "candidate": "phase-selective decode-up FP32",
        "worker_passes": worker_passes,
        "worker_count": len(records),
        "aggregate_rows": len(rows),
        "pass_rows": sum(row["status"] == "pass" for row in rows),
        "precision_mismatch_rows": sum(
            row["status"] == "precision_mismatch" for row in rows),
        "batch_invariance_mismatch_rows": sum(
            row["status"] == "batch_invariance_mismatch" for row in rows),
        "limited_rows": sum(row["status"] == "limited" for row in rows),
        "prompts": prompt_rows,
        "rows": rows,
        "boundary": (
            "four exact tokenizer-generated prompts; B1/B2 prefill and cached "
            "decode; performance medians require a separate repeated gate"),
    }


def main() -> int:
    args = options()
    models = load_prompts(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    combined_raw = []
    for model in models:
        family = model["inference"]["prompt_family"]
        context = model["inference"]["exact_context"]
        output = args.output_directory / family
        command = [
            str(args.pytorch_python), str(SHAPE_RUNNER),
            "--manifest", str(args.manifest),
            "--micro-binary", str(args.micro_binary),
            "--pytorch-python", str(args.pytorch_python),
            "--output-directory", str(output),
            "--models", model["name"], "--contexts", str(context),
            "--batches", "1,2", "--decode-lengths", str(args.decode_tokens),
            "--cases", "prefill,cached", "--micro-kv-cache-dtype", "bf16",
            "--micro-cache-capacity", "exact",
            "--micro-bf16-ffn-decode-up-fp32",
            "--warmup", str(args.warmup), "--steps", str(args.steps),
            "--runs", str(args.runs), "--timeout-seconds", str(args.timeout_seconds),
        ]
        if args.allow_amdsmi_fallback:
            command.append("--allow-amdsmi-fallback")
        completed = subprocess.run(
            command, capture_output=True, text=True,
            timeout=args.timeout_seconds * 8)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{family} matrix failed: " +
                (completed.stderr.strip() or completed.stdout.strip()))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["prompt_family"] = family
            record["prompt_text"] = model["inference"]["prompt_text"]
            combined_raw.append(record)
        print(json.dumps({
            "family": family, "context": context, "status": summary["status"],
        }, sort_keys=True), flush=True)
    summary = aggregate(models, summaries, combined_raw)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_raw),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
