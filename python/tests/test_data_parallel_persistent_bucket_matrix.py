#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_persistent_bucket_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--persistent-gradient-buckets", "bucket_plan_reused",
        "communication_backend_allocation_calls", "EXPECTED_PLAN_BYTES",
        "maximum_engine_current_bytes", "loss_trajectories_exact",
        "keep explicit and continue to view-backed gradients",
    ):
        assert token in text
    print("data-parallel persistent bucket matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
