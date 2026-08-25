#!/usr/bin/env python3

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    runner = (ROOT / "benchmarks/single_gpu/profile_current_cached_decode.py").read_text(
        encoding="utf-8")
    delta = (ROOT / "benchmarks/single_gpu/profile_step_delta.py").read_text(
        encoding="utf-8")
    ast.parse(runner)
    ast.parse(delta)
    for token in (
        "--context", "2048", "--batch", "--decode-tokens", "64",
        "--warmup", "--many-step-count", "--overwrite",
        "--kernel-trace", "--hip-runtime-trace", "--memory-copy-trace",
        "--memory-allocation-trace", "--stats",
        "one_model_forward_per_measured_token", "kv_cache_utilization",
        "profiled cached decode contract changed", "json.dumps(changed",
        "profile_step_delta.py", "inference_cached_decode_kernel_phase_delta",
        "current_cached_decode_profile_summary", "derived_forward_steps",
    ):
        assert token in runner
    assert "cached_attention" in delta
    assert "cached Attention" in delta
    assert "inference_cached_decode_kernel_phase_delta" in delta
    print("current cached decode profile contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
