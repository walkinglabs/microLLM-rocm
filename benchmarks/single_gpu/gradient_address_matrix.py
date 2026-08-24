#!/usr/bin/env python3
"""Audit gradient Storage identity across steady model backward passes."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys


CASES = (
    ("tiny", "fp32", 8),
    ("tiny", "bf16", 8),
    ("qwen", "bf16", 8),
    ("qwen", "bf16", 512),
    ("deepseek", "bf16", 8),
    ("deepseek", "bf16", 512),
)
EXPECTED = {
    "tiny": (22688, 21),
    "qwen": (494032768, 290),
    "deepseek": (1777088000, 339),
}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--qwen-config", required=True, type=pathlib.Path)
    parser.add_argument("--deepseek-config", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    result = parser.parse_args()
    for path in (result.binary, result.qwen_config, result.deepseek_config):
        if not path.is_file(): parser.error(f"input does not exist: {path}")
    if result.runs < 3 or result.warmup < 1 or result.steps < 2:
        parser.error("gradient address matrix counts are invalid")
    return result


def execute(args: argparse.Namespace, model: str, precision: str,
            context: int) -> dict:
    command = [
        str(args.binary), "--model", model, "--precision", precision,
        "--warmup", str(args.warmup), "--steps", str(args.steps),
        "--context", str(context),
    ]
    if model == "qwen": command += ["--config", str(args.qwen_config)]
    elif model == "deepseek":
        command.extend(["--config", str(args.deepseek_config)])
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    try:
        row = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("gradient address benchmark emitted invalid JSON") from error
    expected_parameters, expected_tensors = EXPECTED[model]
    if row.get("schema_version") != 1 or row.get("status") != "pass" or \
       row.get("record_type") != "gradient_address_stability" or \
       row.get("model") != model or row.get("precision") != precision or \
       row.get("context") != context or row.get("warmup") != args.warmup or \
       row.get("steps") != args.steps or \
       row.get("parameter_count") != expected_parameters or \
       row.get("parameter_tensors") != expected_tensors or \
       len(row.get("records", [])) != expected_tensors or \
       row.get("stable_gradient_tensors", 0) + \
           row.get("changed_gradient_tensors", 0) != expected_tensors or \
       any(record.get("observations") != args.steps or
           record.get("minimum_storage_use_count") != 2 or
           record.get("maximum_storage_use_count") != 2
           for record in row.get("records", [])):
        raise RuntimeError(f"invalid gradient address row: {model}/{precision}/T{context}")
    return row


def category(name: str) -> str:
    if ".attention." in name: return "attention"
    if ".feed_forward." in name: return "ffn"
    if "norm" in name: return "norm"
    if "embedding" in name or "output_head" in name: return "embedding_head"
    return "other"


def summarize(records: list[dict], runs: int) -> dict:
    comparisons = []
    for model, precision, context in CASES:
        rows = [row for row in records
                if row["model"] == model and row["precision"] == precision and
                row["context"] == context]
        if len(rows) != runs:
            raise RuntimeError(f"incomplete gradient address case: {model}/{precision}/T{context}")
        signatures = {
            tuple(record["name"] for record in row["records"]
                  if not record["address_stable"])
            for row in rows
        }
        if len(signatures) != 1:
            raise RuntimeError(f"unstable changed-gradient set: {model}/{precision}/T{context}")
        changed_names = next(iter(signatures))
        categories = {}
        for name in changed_names:
            key = category(name)
            categories[key] = categories.get(key, 0) + 1
        comparisons.append({
            "model": model,
            "precision": precision,
            "context": context,
            "runs": runs,
            "parameter_tensors": rows[0]["parameter_tensors"],
            "stable_gradient_tensors": rows[0]["stable_gradient_tensors"],
            "changed_gradient_tensors": rows[0]["changed_gradient_tensors"],
            "stable_gradient_bytes": rows[0]["stable_gradient_bytes"],
            "changed_gradient_bytes": rows[0]["changed_gradient_bytes"],
            "all_gradient_addresses_stable": rows[0]["all_gradient_addresses_stable"],
            "changed_categories": categories,
            "elapsed_ms_median": statistics.median(
                float(row["elapsed_ms"]) for row in rows),
            "engine_peak_bytes_maximum": max(
                int(row["engine_peak_bytes"]) for row in rows),
        })
    by_key = {(row["model"], row["precision"], row["context"]): row
              for row in comparisons}
    gates = {
        "qwen_t8_t512_all_addresses_stable": all(
            by_key[("qwen", "bf16", context)]["all_gradient_addresses_stable"]
            for context in (8, 512)),
        "deepseek_t8_addresses_stable":
            by_key[("deepseek", "bf16", 8)]["all_gradient_addresses_stable"],
        "deepseek_t512_counterexample_present":
            by_key[("deepseek", "bf16", 512)]["changed_gradient_tensors"] == 198 and
            by_key[("deepseek", "bf16", 512)]["changed_gradient_bytes"] == 7107772416,
        "tiny_gqa_counterexample_present": all(
            by_key[("tiny", precision, 8)]["changed_gradient_tensors"] == 4
            for precision in ("fp32", "bf16")),
        "raw_addresses_not_exported": all(
            all("address" not in record for record in row["records"])
            for row in records),
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(gates.values()) else "fail",
        "experiment": "gradient_storage_address_stability",
        "processes": len(records),
        "runs_per_case": runs,
        "comparisons": comparisons,
        "gates": gates,
        "decision": (
            "allow only snapshot-proven optimizer Graph workspaces; "
            "DeepSeek T512 requires stable gradients or recapture"),
    }


def main() -> int:
    args = options()
    records = []
    for process_run in range(1, args.runs + 1):
        ordered = CASES if process_run % 2 else tuple(reversed(CASES))
        for model, precision, context in ordered:
            row = execute(args, model, precision, context)
            row["process_run"] = process_run
            records.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = summarize(records, args.runs)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"gradient_address_matrix: {error}", file=sys.stderr)
        raise SystemExit(2)
