#!/usr/bin/env python3
"""Gate fused BF16-to-FP32 repeat across official V shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


CASES = (
    ("qwen", 1, 256, 2, 64, 7),
    ("qwen", 1, 512, 2, 64, 7),
    ("qwen", 1, 1024, 2, 64, 7),
    ("qwen", 2, 512, 2, 64, 7),
    ("deepseek", 1, 256, 2, 128, 6),
    ("deepseek", 1, 512, 2, 128, 6),
    ("deepseek", 1, 1024, 2, 128, 6),
    ("deepseek", 2, 512, 2, 128, 6),
)
POLICIES = (("composed", "false"), ("fused", "true"))


def last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("BF16 repeat benchmark emitted no JSON")


def median(rows: list[dict], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    args = parser.parse_args()
    if not args.binary.is_file() or args.runs <= 0 or args.warmup < 0 or \
            args.repetitions <= 0 or args.minimum_speedup <= 1:
        parser.error("BF16 repeat comparison options are invalid")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for process_run in range(1, args.runs + 1):
        cases = list(CASES)
        policies = list(POLICIES)
        if process_run % 2 == 0:
            cases.reverse()
            policies.reverse()
        for family, batch, sequence, kv_heads, width, repeats in cases:
            case = f"b{batch}t{sequence}"
            for policy, fused in policies:
                completed = subprocess.run([
                    str(args.binary), "--batch", str(batch),
                    "--sequence", str(sequence),
                    "--kv-heads", str(kv_heads), "--width", str(width),
                    "--repeats", str(repeats), "--fused", fused,
                    "--warmup", str(args.warmup),
                    "--repetitions", str(args.repetitions),
                ], text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stdout + completed.stderr)
                record = last_json(completed.stdout)
                if record.get("status") != "pass" or \
                        record.get("fused") is not (fused == "true"):
                    raise RuntimeError("invalid BF16 repeat record")
                record.update({
                    "record_type": "bf16_repeat_measurement",
                    "family": family, "case": case, "policy": policy,
                    "process_run": process_run,
                    "policy_order": [item[0] for item in policies],
                })
                records.append(record)
    comparisons = []
    for family, batch, sequence, kv_heads, width, repeats in CASES:
        case = f"b{batch}t{sequence}"
        selected = [row for row in records if row["family"] == family and
                    row["case"] == case]
        grouped = {policy: [row for row in selected if row["policy"] == policy]
                   for policy, _ in POLICIES}
        control = median(grouped["composed"], "event_ms_p50")
        fused = median(grouped["fused"], "event_ms_p50")
        comparisons.append({
            "family": family, "case": case, "batch": batch,
            "sequence": sequence, "kv_heads": kv_heads,
            "width": width, "repeats": repeats,
            "composed_event_ms": control, "fused_event_ms": fused,
            "event_speedup": control / fused,
        })
    performance = all(row["event_speedup"] >= args.minimum_speedup
                      for row in comparisons)
    summary = {
        "schema_version": 1, "status": "pass",
        "record_type": "bf16_repeat_summary", "processes": len(records),
        "correctness_gate": True, "performance_gate": performance,
        "comparisons": comparisons,
        "decision": ("advance fused BF16 V repeat to model gate"
                     if performance else "reject fused BF16 V repeat"),
    }
    with (args.output_directory / "raw.jsonl").open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
