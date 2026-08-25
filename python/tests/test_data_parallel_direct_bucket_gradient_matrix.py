#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_direct_bucket_gradient_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--direct-bucket-gradients", "direct_gradient_target_count",
        "pack_copies_removed", "median_forward_backward_ms",
        "forward_backward_speedup_vs_views", "peak_bytes_added_vs_transient",
        "reject direct bucket-gradient model route",
    ):
        assert token in text
    print("data-parallel direct bucket gradient matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
