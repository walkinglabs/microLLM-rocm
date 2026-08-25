#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/bf16_weight_gradient_matrix.py"
BENCHMARK = ROOT / "benchmarks/micro/benchmark_bf16_weight_gradient.cpp"


def main() -> int:
    runner = RUNNER.read_text(encoding="utf-8")
    benchmark = BENCHMARK.read_text(encoding="utf-8")
    ast.parse(runner)
    assert runner.count("qwen2.5-0.5b") >= 2
    assert runner.count("deepseek-r1-distill-qwen-1.5b") >= 2
    for token in (
        "candidate_includes_input_cast_transpose",
        "candidate_includes_gradient_cast",
        "complete_output_finite", "bf16_reference_sample_max_error",
        "fp32_baseline_max_error", "event_speedup_median",
        "passes_operator_performance_gate", "minimum >= 1.0",
    ):
        assert token in runner
    for token in (
        "cast_transpose_2d_out_", "cast_out_", "bf16_matmul_output_out_",
        "MatmulImplementation::HipBLASLt", "complete_output_elements",
        "bf16_reference_sample_max_error", "fp32_baseline_rms_error",
    ):
        assert token in benchmark
    print("BF16 weight-gradient matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

