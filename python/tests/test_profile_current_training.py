#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/profile_current_training.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--kernel-trace", "--stats", "(1, 3)",
        "profile_step_delta.py", "training_kernel_phase_delta",
        "CONTEXT = 512", "MOMENT_THRESHOLD = 1_048_576",
        '"--linear-precision", "bf16"',
        '"--adamw-moment-precision", "bf16"',
        '"--tied-embedding-sparse-add", "true"',
        '"--attention-context-layout-fusion", "true"',
        '"--attention-layout-plan-cache", "false"',
        "OPTIMIZER_METADATA_BYTES_PER_STEP",
        '"optimizer_host_to_device_calls"',
        '"current_training_profile_summary"',
    ):
        assert token in text
    print("current training profile runner contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
