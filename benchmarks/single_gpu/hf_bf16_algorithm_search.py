#!/usr/bin/env python3
"""Search version-local BF16 solutions using complete B1/B2 logits."""

from __future__ import annotations

import argparse
import array
import json
import math
import subprocess
from pathlib import Path

import hf_continuous_matrix as matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()
    candidates = [int(value) for value in args.candidates.split(",")]
    if not candidates or any(value < 0 for value in candidates):
        parser.error("candidates must be nonnegative integers")
    model = matrix.load_models(args.manifest, [args.model])[0]
    seed = [int(value) for value in model["inference"]["token_ids"]]
    tokens = [seed[(index + 5) % len(seed)] for index in range(32)]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    exact = None
    for candidate in candidates:
        values = {}
        for batch in (1, 2):
            output = args.output_directory / f"candidate-{candidate}-b{batch}.bin"
            command = [str(args.binary), "--config", model["config"], "--weights", model["weights"],
                       "--tokens", ",".join(map(str, tokens)), "--device", "hip",
                       "--top-k", "1", "--new-tokens", "0", "--workload", "prefill",
                       "--batch", str(batch), "--bf16-ffn", "true",
                       "--bf16-attention", "true", "--prefill-logits", "last",
                       "--prefill-warmup", "0", "--prefill-steps", "1",
                       "--bf16-algorithm-index", str(candidate),
                       "--logits-output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
            if completed.returncode != 0:
                rows.append({"candidate": candidate, "status": "unsupported",
                             "batch": batch,
                             "error": completed.stderr.strip() or completed.stdout.strip()})
                break
            data = array.array("f")
            data.frombytes(output.read_bytes())
            output.unlink()
            values[batch] = data.tolist()
        if len(values) != 2:
            continue
        absolute = [abs(left - right) for left, right in zip(values[1], values[2])]
        maximum = max(absolute)
        row = {"candidate": candidate, "status": "pass",
               "elements": len(absolute), "max_abs": maximum,
               "mean_abs": sum(absolute) / len(absolute),
               "rms_abs": math.sqrt(sum(value * value for value in absolute) /
                                    len(absolute)),
               "exact": maximum == 0.0}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["exact"]:
            exact = candidate
            break
    (args.output_directory / "raw.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8")
    summary = {"schema_version": 1, "track": "bf16_algorithm_exact_search",
               "status": "exact_found" if exact is not None else "searched_without_exact",
               "model": model["name"], "requested_candidates": candidates,
               "tested_candidates": len({row["candidate"] for row in rows}),
               "exact_candidate": exact, "rows": rows}
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
