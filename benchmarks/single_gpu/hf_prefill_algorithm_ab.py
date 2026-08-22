#!/usr/bin/env python3
"""Fresh-process performance A/B for a version-local BF16 solution index."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import hf_continuous_matrix as matrix


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--algorithm-index", required=True, type=int)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    result = parser.parse_args()
    if result.algorithm_index < 0 or result.runs <= 0 or result.warmup < 0 or result.steps <= 0:
        parser.error("invalid algorithm or repetition settings")
    return result


def command(binary: Path, model: dict, tokens: list[int], batch: int,
            warmup: int, steps: int, algorithm: int | None) -> list[str]:
    result = [str(binary), "--config", model["config"], "--weights", model["weights"],
              "--tokens", ",".join(map(str, tokens)), "--device", "hip",
              "--top-k", "1", "--new-tokens", "0", "--workload", "prefill",
              "--batch", str(batch), "--bf16-ffn", "true",
              "--bf16-attention", "true", "--prefill-logits", "last",
              "--prefill-warmup", str(warmup), "--prefill-steps", str(steps)]
    if algorithm is not None:
        result.extend(["--bf16-algorithm-index", str(algorithm)])
    return result


def main() -> int:
    args = options()
    model = matrix.load_models(args.manifest, [args.model])[0]
    seed = [int(value) for value in model["inference"]["token_ids"]]
    tokens = [seed[(index + 5) % len(seed)] for index in range(32)]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in (1, 2):
        for policy, algorithm in (("default", None), ("common", args.algorithm_index)):
            for process_run in range(1, args.runs + 1):
                completed = subprocess.run(
                    command(args.binary, model, tokens, batch, args.warmup,
                            args.steps, algorithm), capture_output=True, text=True,
                    timeout=900)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                if len(lines) != 1:
                    raise RuntimeError("prefill A/B worker must emit one JSON line")
                row = json.loads(lines[0])
                row.update({"model": model["name"], "batch": batch,
                            "policy": policy, "process_run": process_run})
                rows.append(row)
                print(json.dumps({"batch": batch, "policy": policy,
                                  "process_run": process_run,
                                  "status": row["status"]}), flush=True)
    raw_path = args.output_directory / "raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    summary_rows = []
    for batch in (1, 2):
        values = {}
        for policy in ("default", "common"):
            selected = [float(row["prefill_tokens_per_second"]) for row in rows
                        if row["batch"] == batch and row["policy"] == policy]
            values[policy] = statistics.median(selected)
        summary_rows.append({"batch": batch,
                             "default_tokens_per_second_p50": values["default"],
                             "common_tokens_per_second_p50": values["common"],
                             "common_over_default": values["common"] / values["default"]})
    summary = {"schema_version": 1, "track": "bf16_algorithm_prefill_ab",
               "status": "pass", "model": model["name"],
               "algorithm_index": args.algorithm_index, "runs": args.runs,
               "warmup": args.warmup, "steps": args.steps,
               "rows": summary_rows,
               "boundary": "no trace; fresh-process prefill timing"}
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
