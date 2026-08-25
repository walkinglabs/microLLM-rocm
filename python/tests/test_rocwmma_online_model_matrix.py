#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/rocwmma_online_model_matrix.py"


def main() -> int:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    text = RUNNER.read_text(encoding="utf-8")
    assert len(tree.body) > 5
    for token in (
        "--inference-bthd-online-attention",
        "rocwmma_online_attention_native_calls",
        "maximum_absolute_logit_difference",
        "prefill_tokens_per_second",
        "engine_peak_bytes",
        "model_route_accepted",
    ):
        assert token in text
    print("rocWMMA online model matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
