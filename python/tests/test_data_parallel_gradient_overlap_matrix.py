#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_gradient_overlap_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--overlap-gradient-communication", "overlap_communication_performed",
        "overlapped_bucket_count", "median_overlap_finish_ms",
        "total_speedup_vs_synchronous_views", "peak_bytes_added_vs_transient",
        "keep explicit and move to one-process-per-GPU",
        "reject single-process gradient overlap route",
    ):
        assert token in text
    print("data-parallel gradient overlap matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
