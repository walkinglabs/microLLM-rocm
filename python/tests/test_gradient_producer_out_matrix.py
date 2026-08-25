#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/micro/gradient_producer_out_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "model_s_head_t32", "model_s_ffn_t32", "model_s_attention_t32",
        "model_s_head_t512", "tiny_counterexample", "direct-first",
        "allocating_calls_per_invocation", "direct_calls_per_invocation",
        "passes_1_05_gate", "admit exact producer shapes to Autograd gate",
    ):
        assert token in text
    print("gradient producer out matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
