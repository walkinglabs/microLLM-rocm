#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_verification_matrix.py"
CLI = ROOT / "apps/distributed_train.cpp"
SOURCE = ROOT / "src/multi_gpu/data_parallel.cpp"


def main() -> int:
    runner = RUNNER.read_text(encoding="utf-8")
    cli = CLI.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(runner)
    for token in (
        "--parameter-check-interval", "every_step", "final_step", "disabled",
        "rows[1:]", "rotated policy order", "loss_trajectories_exact",
    ):
        assert token in runner
    for token in (
        "parameter_check_performed", "verification_ms",
        "parameter_check_interval",
    ):
        assert token in cli
        assert token in source
    print("data-parallel verification matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

