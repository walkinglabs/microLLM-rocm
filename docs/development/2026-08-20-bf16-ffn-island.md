# 2026-08-20 — continuous BF16 FFN activation island

## Outcome

`ops::bf16_ffn` now keeps gate/up projections, SwiGLU and the down projection in one
continuous BF16 region. It casts the FP32 residual input once and returns FP32 at the
residual boundary. BF16 weights are caller-owned; the operator does not keep a hidden
FP32+BF16 pair.

## Real-shape correction

The first square smoke was incomplete. Qwen decode shapes exposed hipBLASLt status 6 for
direct BF16-input/FP32-output GEMM. A per-shape capability cache now remembers the rejection
and falls back to BF16 output plus a device cast. The exact Qwen decode shape has a dedicated
zero-transfer/reuse test.

## Evidence

- CPU `158/158`, sanitizer `156/156`, HIP `61/61`, Python/PyTorch oracle `4/4` pass;
- 36/36 benchmark records pass error and zero-payload-transfer gates;
- operator speedup versus FP32 ranges from `1.117×` to `1.576×` across the fixed matrix;
- speedup versus the previous per-Linear BF16 composition ranges from `1.067×` to `1.091×`;
- raw JSONL, summary, kernel traces and a generated SVG are under Experiment 030.

## Honest boundary

This is an operator result, not whole-model BF16 inference or training. The next node must
define single-representation BF16 inference weights, official-model logits/tokens and an
end-to-end PyTorch BF16 comparison before any whole-model claim.
