# Qwen3 complete one-step training alignment

This evidence replaces the earlier gate/up-only boundary with every independent runtime
parameter in the pinned official Qwen3-0.6B model.

The checkpoint stores 311 Tensors, but `lm_head.weight` is tied to
`model.embed_tokens.weight`. Both frameworks therefore export 310 independent internal
Tensors once, containing 596,049,920 values. The audit compares every gradient before
AdamW and every parameter after one step: 1,192,099,840 values per precision.

## Decision

| Precision | Gradient Max / aggregate RMS | Parameter Max / aggregate RMS | Decision |
|---|---:|---:|---|
| FP32 | `5.746e-4 / 5.024e-7` | `1.999e-5 / 5.110e-8` | pass |
| BF16 forward, FP32 masters | `3.641e-1 / 4.071e-4` | `2.289e-5 / 2.253e-6` | reject |

FP32 passes every predeclared aggregate gate. Its largest gradient difference is in the
tied embedding. BF16 fails Gradient Max (`0.3641 > 0.05`) and updated-parameter aggregate
RMS (`2.253e-6 > 2e-6`). Its largest gradient difference is also the tied embedding;
Attention QKV, FFN down and gate/up reach `0.2645`, `0.3043` and `0.2536` respectively.

All names, shapes, element counts and finite-value checks pass in both precisions. This
means BF16 executes completely, not that it matches the PyTorch BF16 training formula.

## RMS boundary

The fixed RMS gate is one aggregate over all 596,049,920 values. It is not a promise that
every small Tensor has an RMS below the same number. The raw evidence therefore also keeps
the largest individual-Tensor RMS:

- FP32 gradient: `2.356e-5`, `blocks.5.attention.q_norm.weight`;
- FP32 parameter: `1.206e-6`, `blocks.26.attention_norm.weight`;
- BF16 gradient: `1.993e-2`, `blocks.0.attention.q_norm.weight`;
- BF16 parameter: `5.301e-6`, `blocks.5.attention.q_norm.weight`.

Global Max still bounds every individual value. Per-Tensor RMS is diagnostic in this node
and is not silently substituted for the predeclared aggregate gate.

## Files

- `fp32-raw.jsonl` / `bf16-raw.jsonl`: 620 records each, one for every gradient and parameter Tensor;
- `*-summary.json`: aggregate and nine observed-family attribution;
- `*-workers.json`: C++ and PyTorch worker metadata;
- `summary.json`: one compact final decision.

Four temporary safetensors exports total 9,536,932,560 bytes per precision. They were
verified and deleted after each comparison; no model, gradient or updated-weight payload is
vendored here. Serialization is diagnostic-only and excluded from performance claims.

AdamW moments, multi-step trajectories, SFT and other GPUs remain separate gates.

Repository regression closes at CPU 434/434, ASan/UBSan 431/431 and MI300X HIP
215/215. The coverage inventory reports 199 Tensor operators, 45 graph APIs and 159
registered test files.
