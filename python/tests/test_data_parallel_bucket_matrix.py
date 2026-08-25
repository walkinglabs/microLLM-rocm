#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_bucket_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        '("4b", 4)', '("64b", 64)', '("4kib", 4096)',
        '"--parameter-check-interval"', "rows[1:]", "rotated bucket order",
        "loss_trajectories_exact", "one_bucket_policies_are_equivalent_workloads",
        "multi_bucket_policies_are_slower", "Model-S multi-bucket workload",
    ):
        assert token in text
    print("data-parallel bucket matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
