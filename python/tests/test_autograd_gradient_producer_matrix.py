#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/micro/autograd_gradient_producer_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "model_s_head_t32", "model_s_ffn_t32", "model_s_attention_t32",
        "model_s_head_t512", "tiny_counterexample", "baseline-first",
        "direct_dispatches_per_invocation", "allocation_calls_removed_per_invocation",
        "passes_1_05_gate", "remove scoped Autograd producer route",
    ):
        assert token in text
    print("Autograd gradient producer matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
