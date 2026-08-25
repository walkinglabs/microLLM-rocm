#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_gradient_ready_audit.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--record-gradient-ready-order", "gradient_ready_order_rank0",
        "parameter_names", "parameter_elements", "bucket_ranges",
        "ready_order_is_reverse_parameter_order",
        "buckets_ready_before_backward_end", "admit event-based overlap prototype",
    ):
        assert token in text
    print("data-parallel gradient-ready audit contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
