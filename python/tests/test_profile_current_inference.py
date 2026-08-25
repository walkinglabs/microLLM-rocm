#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/profile_current_inference.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--kernel-trace", "--stats", "(1, 6)",
        "profile_step_delta.py", "inference_prefill_kernel_phase_delta",
        "--inference-bthd-online-attention", "derived_forwards",
        "--expected-bf16-ffn-norm",
        "--expected-bf16-attention-norm",
    ):
        assert token in text
    print("current inference profile runner contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
