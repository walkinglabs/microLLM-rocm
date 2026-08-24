#!/usr/bin/env python3
"""Reject a stale hf_infer binary whose embedded CLI contract lags the source."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise RuntimeError("hf_infer binary is missing")
    payload = args.binary.read_bytes()
    required = (
        b"device-tensor-amax",
        b"ffn-outer-row",
        b"--fp8-activation-minimum-scale",
        b"fp8_device_weight_bytes_scanned",
        b"fp8_dynamic_tensor_calls",
        b"--fp8-fp32-layers",
        b"--fp8-diagnostic-mode",
        b"fp8_linears_covered",
        b"both-roundtrip",
        b"fp8_output_column_scale_calls",
        b"fp8_dynamic_column_calls",
        b"output-channel-amax",
        b"fp8_output_column_native_status",
        b"--fp8-weight-scale-scope",
        b"attention-output-only",
        b"fp8_dynamic_clipped_tensor_calls",
        b"--bf16-ffn-arena",
        b"bf16_ffn_arena_capacity_bytes",
        b"--bf16-ffn-arena-minimum-rows",
        b"bf16_ffn_arena_bypassed_calls",
        b"--bf16-qkv-arena",
        b"--bf16-qkv-arena-minimum-rows",
        b"bf16_qkv_arena_capacity_bytes",
        b"--allocation-source-diagnostics",
        b"allocation_source_records",
        b"--attention-core-arena",
        b"--attention-core-arena-minimum-sequence",
        b"attention_core_arena_capacity_bytes",
        b"--bf16-grouped-gate-up-algorithm-index",
        b"bf16_grouped_gate_up_dispatches",
        b"prefill logits shape does not match batch export contract",
        b"--strided-copy-diagnostics",
        b"strided_copy_records",
        b"--inference-bthd-attention",
        b"inference_bthd_attention",
    )
    missing = [value.decode() for value in required if value not in payload]
    if missing:
        raise RuntimeError(f"hf_infer binary has a stale CLI contract: {missing}")
    removed = (b"--fp8-activation-format",)
    retained = [value.decode() for value in removed if value in payload]
    if retained:
        raise RuntimeError(
            f"hf_infer binary retains rejected CLI policies: {retained}")
    print("hf_infer binary contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
