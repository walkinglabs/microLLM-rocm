#!/usr/bin/env python3
"""Attribute composed-policy T512 strided copies to model source scopes."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


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
    parser.add_argument("--sequence", type=int, default=512)
    result = parser.parse_args()
    if (result.runs <= 0 or result.sequence <= 0 or
            not result.manifest.is_file() or not result.binary.is_file()):
        parser.error("strided source inputs are invalid or unavailable")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    if document.get("schema_version") != 1 or \
            {model.get("name") for model in result} != set(INDICES):
        raise RuntimeError("strided source gate requires pinned official models")
    return result


def repeated(seed: list[int], length: int) -> list[int]:
    return [seed[index % len(seed)] for index in range(length)]


def command(args: argparse.Namespace, model: dict) -> list[str]:
    tokens = repeated(model["inference"]["token_ids"], args.sequence)
    qkv_index, gate_up_index = INDICES[model["name"]]
    return [
        str(args.binary), "--config", model["config"],
        "--weights", model["weights"], "--tokens",
        ",".join(str(token) for token in tokens),
        "--device", "hip", "--top-k", "10", "--batch", "1",
        "--bf16-ffn", "true", "--bf16-attention", "true",
        "--bf16-ffn-arena", "true",
        "--bf16-ffn-arena-minimum-rows", "512",
        "--bf16-qkv-arena", "true",
        "--bf16-qkv-arena-minimum-rows", "512",
        "--bf16-grouped-qkv-algorithm-index", str(qkv_index),
        "--bf16-grouped-gate-up-algorithm-index", str(gate_up_index),
        "--strided-copy-diagnostics", "true",
        "--workload", "prefill", "--new-tokens", "0",
        "--warmup", "0", "--steps", "1",
        "--prefill-warmup", "0", "--prefill-steps", "1",
        "--prefill-logits", "last",
    ]


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("hf_infer emitted no JSON")


def main() -> int:
    args = options()
    selected_models = models(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        order = list(selected_models)
        if process_run % 2 == 0:
            order.reverse()
        for model in order:
            completed = subprocess.run(
                command(args, model), text=True,
                capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stdout + completed.stderr)
            record = last_json(completed.stdout)
            if record.get("status") != "pass" or \
                    record.get("strided_copy_diagnostics") is not True or \
                    int(record.get("bf16_grouped_qkv_dispatches", 0)) <= 0 or \
                    int(record.get(
                        "bf16_grouped_gate_up_dispatches", 0)) <= 0:
                raise RuntimeError("invalid strided source record")
            record.update({
                "record_type": "hf_strided_copy_source_measurement",
                "model": model["name"],
                "revision": model["revision"],
                "process_run": process_run,
                "process_order": [item["name"] for item in order],
            })
            records.append(record)

    comparisons = []
    for model in selected_models:
        selected = [row for row in records
                    if row["model"] == model["name"]]
        canonical = selected[0]["strided_copy_records"]
        if any(row["strided_copy_records"] != canonical
               for row in selected[1:]):
            raise RuntimeError("strided source records changed across processes")
        source_totals = defaultdict(lambda: {"calls": 0, "bytes": 0})
        for record in canonical:
            source_totals[record["source"]]["calls"] += int(record["calls"])
            source_totals[record["source"]]["bytes"] += int(record["bytes"])
        comparisons.append({
            "model": model["name"],
            "revision": model["revision"],
            "calls": int(selected[0]["strided_copy_calls"]),
            "bytes": int(selected[0]["strided_copy_bytes"]),
            "record_count": len(canonical),
            "source_totals": dict(sorted(source_totals.items())),
            "records": canonical,
        })
    expected_sources = {"attention.core", "attention.layout"}
    attribution = all(
        set(row["source_totals"]) == expected_sources and
        sum(item["calls"] for item in row["source_totals"].values()) ==
            row["calls"] and
        sum(item["bytes"] for item in row["source_totals"].values()) ==
            row["bytes"]
        for row in comparisons)
    summary = {
        "schema_version": 1,
        "status": "pass" if attribution else "fail",
        "record_type": "hf_strided_copy_source_summary",
        "raw_processes": len(records),
        "attribution_gate": attribution,
        "comparisons": comparisons,
        "decision": (
            "next candidate is an inference BTHD Attention island"
            if attribution else "strided source attribution incomplete"),
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
