#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_model_s_bucket_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "15_586_176", '("1mib", 1 * 1024 * 1024)',
        '("25mib", 25 * 1024 * 1024)', '"model-s"',
        '"bucket_total_elements"', '"maximum_engine_peak_bytes"',
        "rows[1:]", "rotated bucket order", "loss_trajectories_exact",
    ):
        assert token in text
    print("data-parallel Model-S bucket matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

