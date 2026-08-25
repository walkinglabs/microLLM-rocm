#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_weight_gradient_workspace_matrix.py"
BENCHMARK = ROOT / "benchmarks/micro/benchmark_bf16_weight_gradient.cpp"


def main() -> int:
    runner = RUNNER.read_text(encoding="utf-8")
    benchmark = BENCHMARK.read_text(encoding="utf-8")
    ast.parse(runner)
    for token in (
        "preallocated_over_allocating_wall_speedup",
        "preallocated_over_allocating_event_speedup",
        "allocating_backend_allocation_calls_per_invocation",
        "wall median >= 1.01", "reject workspace API",
    ):
        assert token in runner
    for token in (
        "enable_hip_caching_allocator", "allocation_before", "allocation_after",
        "allocating_allocation_calls_per_invocation",
        "allocating_cache_reuse_calls_per_invocation",
    ):
        assert token in benchmark
    print("BF16 weight-gradient workspace matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

