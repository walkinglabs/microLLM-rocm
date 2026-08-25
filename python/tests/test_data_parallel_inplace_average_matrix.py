#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_inplace_average_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--inplace-bucket-average", "average_tensor_count",
        "communication_backend_allocation_calls", "249378816", "374068224",
        "loss_trajectories_exact", "keep in-place bucket average as default",
    ):
        assert token in text
    print("data-parallel in-place average matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

