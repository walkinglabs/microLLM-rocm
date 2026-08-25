#!/usr/bin/env python3

import ast
import json
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
        "cached_attention_materialized_policy", "auto-enabled",
    ):
        assert token in runner
    assert "cached_attention" in delta
    assert "cached Attention" in delta
    assert "cached Attention scores" in delta
    assert "cached Attention finalize" in delta
    assert "inference_cached_decode_kernel_phase_delta" in delta
    result = (ROOT / "benchmarks/results" /
              "2026-08-25-current-deepseek-t2048-profile")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    analysis = json.loads((result / "analysis.json").read_text(encoding="utf-8"))
    verification = json.loads((result / "verification.json").read_text(
        encoding="utf-8"))
    profile = summary["kernel_profile"]
    categories = {row["category"]: row for row in profile["categories"]}
    assert summary["status"] == "pass"
    assert (summary["context"], summary["batch"], summary["decode_tokens"]) == (
        2048, 2, 64)
    assert summary["derived_forward_steps"] == 128
    assert profile["negative_call_delta_names"] == []
    assert profile["total_kernel_ns_per_step"] == 1051286618.0
    assert categories["cached Attention"] == {
        "calls_per_step": 1792.0,
        "category": "cached Attention",
        "duration_ns_per_step": 647256070.5,
        "kernel_share": 0.6156799291627624,
    }
    assert categories["hipBLASLt GEMM"]["kernel_share"] == \
        0.25721431231991576
    assert analysis["backend_allocation_calls"] == 0
    assert analysis["cache_reuse_calls"] == 36963
    assert analysis["hip_memcpy_duration_is_not_copy_cost"] is True
    assert verification["measurement_commit"] == \
        "90862e88cb7ba61bec3f1f4fc0cb5d37ee499ef3"
    assert verification["current_profile_confirms_cached_attention_hotspot"] is True
    assert verification["allocator_is_current_hotspot"] is False
    for prefix in ("1-step", "3-step"):
        for suffix in ("kernel", "hip-api", "memory-copy", "memory-allocation"):
            assert (result / f"{prefix}-{suffix}-stats.csv").is_file()
    print("current cached decode profile contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
