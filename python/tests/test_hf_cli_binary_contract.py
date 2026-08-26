#!/usr/bin/env python3
"""Reject a stale hf_infer binary whose embedded CLI contract lags the source."""

import argparse
import subprocess
import tempfile
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
        b"--bf16-ffn-fp32-layers",
        b"bf16_ffn_fp32_layers",
        b"--bf16-decode-algorithm-index",
        b"bf16_decode_algorithm_index",
        b"bf16_registered_algorithm_count",
        b"--fp32-prefill-q-solution-index",
        b"--fp32-prefill-kv-solution-index",
        b"fp32_prefill_q_solution_index",
        b"--fp32-prefill-attention-qk-solution-index",
        b"--fp32-prefill-attention-pv-solution-index",
        b"--fp32-prefill-attention-o-solution-index",
        b"fp32_prefill_attention_qk_solution_index",
        b"fp32_prefill_attention_pv_solution_index",
        b"fp32_prefill_attention_o_solution_index",
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
        b"--cached-attention-splits",
        b"--cached-attention-minimum-sequence",
        b"cached_attention_splits",
        b"--cached-attention-materialized",
        b"cached_attention_materialized_scores",
        b"cached_attention_materialized_policy",
        b"cached_attention_pv_splits",
        b"cached-attention-pv-splits",
        b"--cache-logits-step",
        b"--prefill-cache-output",
        b"--prefill-cache-layer",
        b"prefill_cache_exported",
        b"decode_tokens",
        b"hf-cached-decode",
        b"inference.cached.blocks.",
        b"auto-enabled",
        b"auto-bypass",
        b"--bf16-grouped-gate-up-algorithm-index",
        b"bf16_grouped_gate_up_dispatches",
        b"prefill logits shape does not match batch export contract",
        b"--strided-copy-diagnostics",
        b"strided_copy_records",
        b"--inference-bthd-attention",
        b"inference_bthd_attention",
        b"--inference-bthd-bf16-qk",
        b"inference_bthd_bf16_qk",
        b"bf16_grouped_qkv_retained_query_key_dispatches",
        b"--int8-linear",
        b"int8_device_weight_bytes_scanned",
        b"single_representation_int8_linear_explicit",
    )
    missing = [value.decode() for value in required if value not in payload]
    if missing:
        raise RuntimeError(f"hf_infer binary has a stale CLI contract: {missing}")
    removed = (b"--fp8-activation-format",)
    retained = [value.decode() for value in removed if value in payload]
    if retained:
        raise RuntimeError(
            f"hf_infer binary retains rejected CLI policies: {retained}")
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "missing"
        rejected = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1",
            "--device", "cpu", "--inference-bthd-bf16-qk", "true",
        ], text=True, capture_output=True, check=False)
        if rejected.returncode == 0 or \
                "requires BTHD Attention, QKV Arena" not in rejected.stderr:
            raise RuntimeError(
                "hf_infer accepted BF16 Q/K without its required exact route")
        rejected_split = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1",
            "--device", "cpu", "--cached-attention-splits", "33",
        ], text=True, capture_output=True, check=False)
        if rejected_split.returncode == 0 or \
                "--cached-attention-splits must be 0..32" not in \
                rejected_split.stderr:
            raise RuntimeError(
                "hf_infer accepted an invalid cached Attention split count")
        rejected_pv_split = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1",
            "--device", "cpu", "--cached-attention-pv-splits", "33",
        ], text=True, capture_output=True, check=False)
        if rejected_pv_split.returncode == 0 or \
                "--cached-attention-pv-splits must be 0..32" not in \
                rejected_pv_split.stderr:
            raise RuntimeError(
                "hf_infer accepted an invalid cached Attention P*V split count")
        rejected_conflict = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1",
            "--device", "cpu", "--cached-attention-splits", "2",
            "--cached-attention-materialized", "true",
        ], text=True, capture_output=True, check=False)
        if rejected_conflict.returncode == 0 or \
                "mutually exclusive" not in rejected_conflict.stderr:
            raise RuntimeError(
                "hf_infer accepted conflicting cached Attention policies")
        rejected_pv_conflict = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1",
            "--device", "cpu", "--cached-attention-pv-splits", "16",
            "--cached-attention-materialized", "true",
        ], text=True, capture_output=True, check=False)
        if rejected_pv_conflict.returncode == 0 or \
                "mutually exclusive" not in rejected_pv_conflict.stderr:
            raise RuntimeError(
                "hf_infer accepted conflicting split-P*V and materialized policies")
        rejected_logit_step = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1",
            "--device", "cpu", "--cache-logits-step", "0",
        ], text=True, capture_output=True, check=False)
        if rejected_logit_step.returncode == 0 or \
                "requires an output" not in rejected_logit_step.stderr:
            raise RuntimeError("hf_infer accepted a logits step without output")
        rejected_cache_layer = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--prefill-cache-layer", "0",
        ], text=True, capture_output=True, check=False)
        if rejected_cache_layer.returncode == 0 or \
                "requires one full cached decode" not in rejected_cache_layer.stderr:
            raise RuntimeError("hf_infer accepted a cache layer without export")
        rejected_bf16_layers = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--bf16-ffn-fp32-layers", "0",
        ], text=True, capture_output=True, check=False)
        if rejected_bf16_layers.returncode == 0 or \
                "requires --bf16-ffn true" not in rejected_bf16_layers.stderr:
            raise RuntimeError(
                "hf_infer accepted selective BF16 layers without BF16 FFN")
        rejected_decode_algorithm = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--workload", "decode", "--new-tokens", "1",
            "--bf16-ffn", "true",
            "--bf16-decode-algorithm-index", "75892",
        ], text=True, capture_output=True, check=False)
        if rejected_decode_algorithm.returncode == 0 or \
                "requires HIP cached decode" not in rejected_decode_algorithm.stderr:
            raise RuntimeError(
                "hf_infer accepted a decode BF16 solution outside HIP cached decode")
        rejected_prefill_projection = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--workload", "decode", "--new-tokens", "1", "--use-cache", "true",
            "--cache-prefill-mode", "full",
            "--fp32-prefill-q-solution-index", "296100",
        ], text=True, capture_output=True, check=False)
        if rejected_prefill_projection.returncode == 0 or \
                "require HIP full cached decode" not in \
                rejected_prefill_projection.stderr:
            raise RuntimeError(
                "hf_infer accepted a prefill projection solution outside HIP")
        rejected_prefill_attention = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--workload", "decode", "--new-tokens", "1", "--use-cache", "true",
            "--cache-prefill-mode", "full",
            "--fp32-prefill-attention-qk-solution-index", "304681",
        ], text=True, capture_output=True, check=False)
        if rejected_prefill_attention.returncode == 0 or \
                "require HIP full cached decode" not in \
                rejected_prefill_attention.stderr:
            raise RuntimeError(
                "hf_infer accepted a cached-prefill Attention solution outside HIP")
        independent_attention = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--bf16-ffn", "false", "--bf16-attention", "true",
        ], text=True, capture_output=True, check=False)
        if independent_attention.returncode == 0 or \
                "requires --bf16-ffn" in independent_attention.stderr:
            raise RuntimeError(
                "hf_infer still couples independent BF16 Attention and FFN islands")
        rejected_decode_trace = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--workload", "decode", "--new-tokens", "1",
            "--trace-output", str(Path(temporary) / "trace.jsonl"),
            "--warmup", "0", "--steps", "1",
        ], text=True, capture_output=True, check=False)
        if rejected_decode_trace.returncode == 0 or \
                "one selected cached decode step" not in rejected_decode_trace.stderr:
            raise RuntimeError(
                "hf_infer accepted an unselected cached decode trace")
        rejected_binary_trace = subprocess.run([
            str(args.binary), "--config", str(missing),
            "--weights", str(missing), "--tokens", "1", "--device", "cpu",
            "--trace-binary-directory", str(Path(temporary) / "binary-values"),
        ], text=True, capture_output=True, check=False)
        if rejected_binary_trace.returncode == 0 or \
                "requires --trace-output and an explicit --trace-value-filter" \
                not in rejected_binary_trace.stderr:
            raise RuntimeError(
                "hf_infer accepted unscoped binary trace output")
    print("hf_infer binary contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
