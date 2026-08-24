#!/usr/bin/env python3
"""Collect deterministic model allocation source×size distributions."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--context", type=int, default=512)
    result = parser.parse_args()
    if result.runs <= 0 or result.context <= 0:
        parser.error("runs and context must be positive")
    if not result.manifest.is_file() or not result.binary.is_file():
        parser.error("manifest and binary must exist")
    return result


def models(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = document.get("models", [])
    if document.get("schema_version") != 1 or len(result) != 2:
        raise RuntimeError("allocation attribution requires two schema-v1 models")
    for model in result:
        if not Path(model["config"]).is_file() or not Path(model["weights"]).is_file():
            raise RuntimeError(f"checkpoint unavailable: {model['name']}")
    return result


def repeated(seed: list[int], length: int) -> str:
    return ",".join(str(seed[index % len(seed)]) for index in range(length))


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("binary emitted no JSON object")


def signature(record: dict) -> list[tuple]:
    return sorted((row["source"], row["device"], int(row["allocation_bytes"]),
                   int(row["calls"]), int(row["total_bytes"]))
                  for row in record["allocation_source_records"])


def main() -> int:
    args = options()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    logs = args.output_directory / "logs"
    logs.mkdir(exist_ok=True)
    raw: list[dict] = []
    for model in models(args.manifest):
        for process_run in range(1, args.runs + 1):
            command = [
                str(args.binary), "--config", model["config"],
                "--weights", model["weights"],
                "--tokens", repeated(model["inference"]["token_ids"], args.context),
                "--device", "hip", "--top-k", "10", "--batch", "1",
                "--bf16-ffn", "true", "--bf16-attention", "true",
                "--bf16-ffn-arena", "true",
                "--bf16-ffn-arena-minimum-rows", "512",
                "--bf16-qkv-arena", "false", "--workload", "prefill",
                "--new-tokens", "0", "--warmup", "0", "--steps", "1",
                "--prefill-warmup", "0", "--prefill-steps", "1",
                "--prefill-logits", "last",
                "--allocation-source-diagnostics", "true",
            ]
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False)
            stem = f"{model['name']}-p{process_run}"
            (logs / f"{stem}.stdout.txt").write_text(
                completed.stdout, encoding="utf-8")
            (logs / f"{stem}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"diagnostic run failed for {stem}: {completed.stderr}")
            record = last_json(completed.stdout)
            records = record.get("allocation_source_records", [])
            if (record.get("status") != "pass" or
                    not record.get("allocation_source_diagnostics") or not records or
                    record.get("allocation_source_calls") !=
                        sum(int(row["calls"]) for row in records) or
                    record.get("allocation_source_bytes") !=
                        sum(int(row["total_bytes"]) for row in records)):
                raise RuntimeError(f"invalid allocation diagnostics for {stem}")
            record.update({
                "record_type": "hf_allocation_source_measurement",
                "model": model["name"], "revision": model["revision"],
                "context": args.context, "process_run": process_run,
            })
            raw.append(record)

    summaries = []
    top_sources = []
    for model in models(args.manifest):
        selected = [record for record in raw if record["model"] == model["name"]]
        reference = signature(selected[0])
        if any(signature(record) != reference for record in selected[1:]):
            raise RuntimeError(f"allocation distribution is not deterministic: {model['name']}")
        grouped: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "total_bytes": 0, "sizes": []})
        for source, device, size, calls, total_bytes in reference:
            row = grouped[source]
            row["device"] = device
            row["calls"] += calls
            row["total_bytes"] += total_bytes
            row["sizes"].append({"allocation_bytes": size, "calls": calls,
                                 "total_bytes": total_bytes})
        sources = []
        for source, values in grouped.items():
            values["sizes"].sort(
                key=lambda row: (-row["total_bytes"], -row["calls"],
                                 -row["allocation_bytes"]))
            sources.append({"source": source, **values})
        sources.sort(key=lambda row: (-row["total_bytes"], -row["calls"],
                                     row["source"]))
        top_sources.append(sources[0]["source"])
        summaries.append({
            "model": model["name"], "revision": model["revision"],
            "context": args.context,
            "allocation_calls": selected[0]["allocation_source_calls"],
            "allocation_bytes": selected[0]["allocation_source_bytes"],
            "top_source": sources[0]["source"],
            "sources": sources,
        })
    common = top_sources[0] if len(set(top_sources)) == 1 else "split"
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "hf_allocation_source_summary",
        "raw_processes": len(raw), "context": args.context,
        "deterministic_distributions": True,
        "common_top_source": common,
        "models": summaries,
        "decision": (f"profile and optimize {common}"
                     if common != "split" else
                     "separate model allocation hypotheses"),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for record in raw:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
