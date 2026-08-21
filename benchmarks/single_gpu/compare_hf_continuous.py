#!/usr/bin/env python3
"""Join microLLM continuous-serving evidence with a sequential PyTorch reference.

The output deliberately calls the throughput quotient an observed service ratio:
the two programs use the same requests and generated-token comparison, but they do
not use the same scheduler or the same weight residency policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-summary", required=True, type=Path)
    parser.add_argument("--pytorch-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def pytorch_records(directory: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    for path in sorted(directory.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["model"], path.stem)
        if key in records:
            raise RuntimeError(f"duplicate PyTorch record: {key}")
        if record.get("record_type") != "pytorch_sequential_request_reference" or \
                record.get("status") != "pass":
            raise RuntimeError(f"invalid PyTorch reference: {path}")
        records[key] = record
    return records


def compare(micro: dict, pytorch: dict[tuple[str, str], dict]) -> dict:
    if micro.get("track") != "official_continuous_serving_matrix" or \
            micro.get("status") != "pass":
        raise RuntimeError("microLLM matrix is not a passing official matrix")
    raw = micro.get("rows", [])
    representatives: dict[tuple[str, str], dict] = {}
    for row in raw:
        key = (row["model"], row["case"])
        representatives.setdefault(key, row)
        first = representatives[key]
        stable_fields = (
            "generated_tokens", "token_checksum", "allocated_cache_bytes",
            "peak_active_cache_bytes", "kv_cache_byte_utilization",
            "slot_utilization", "resident_weight_bytes", "engine_peak_bytes",
        )
        if any(row[field] != first[field] for field in stable_fields):
            raise RuntimeError(f"microLLM evidence changed across runs: {key}")

    rows = []
    for aggregate in micro.get("aggregates", []):
        key = (aggregate["model"], aggregate["case"])
        if aggregate.get("status") != "pass" or \
                aggregate.get("successful_runs") != micro.get("runs"):
            raise RuntimeError(f"incomplete microLLM aggregate: {key}")
        current = representatives[key]
        reference = pytorch.get(key)
        if reference is None:
            raise RuntimeError(f"missing PyTorch reference: {key}")
        if current["prompt_lengths"] != reference["prompt_lengths"] or \
                current["new_token_lengths"] != reference["new_token_lengths"]:
            raise RuntimeError(f"request axes differ: {key}")
        exact = current["generated_tokens"] == reference["generated_tokens"]
        micro_tps = float(aggregate["tokens_per_second_p50"])
        pytorch_tps = float(reference["tokens_per_second"])
        rows.append({
            "model": key[0],
            "case": key[1],
            "request_count": current["request_count"],
            "slots": current["continuous_slots"],
            "maximum_prompt_length": max(current["prompt_lengths"]),
            "micro_tokens_per_second_p50": micro_tps,
            "micro_tokens_per_second_min": aggregate["tokens_per_second_min"],
            "micro_tokens_per_second_max": aggregate["tokens_per_second_max"],
            "pytorch_sequential_tokens_per_second": pytorch_tps,
            "observed_service_throughput_ratio": micro_tps / pytorch_tps,
            "exact_generated_tokens": exact,
            "accuracy_status": "pass" if exact else "fail",
            "micro_allocated_cache_bytes": current["allocated_cache_bytes"],
            "micro_peak_active_cache_bytes": current["peak_active_cache_bytes"],
            "micro_kv_cache_byte_utilization": current["kv_cache_byte_utilization"],
            "micro_slot_utilization": current["slot_utilization"],
            "micro_engine_peak_bytes": current["engine_peak_bytes"],
            "micro_resident_weight_bytes": current["resident_weight_bytes"],
            "pytorch_peak_bytes": reference["peak_bytes"],
            "pytorch_resident_weight_bytes": reference["resident_weight_bytes"],
        })
    if len(rows) != len(pytorch):
        raise RuntimeError("microLLM and PyTorch matrices have different sizes")
    return {
        "schema_version": 1,
        "track": "official_continuous_serving_comparison",
        "status": "complete_with_recorded_accuracy_failures"
        if any(row["accuracy_status"] == "fail" for row in rows) else "pass",
        "micro_process_runs_per_case": micro["runs"],
        "comparison_boundary": (
            "microLLM continuous slots versus PyTorch sequential requests; "
            "same request tokens, different scheduler and weight residency policy"
        ),
        "rows": rows,
    }


def main() -> int:
    args = options()
    result = compare(
        json.loads(args.micro_summary.read_text(encoding="utf-8")),
        pytorch_records(args.pytorch_directory))
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
