# Qwen3 gate/up training alignment audit

This diagnostic compares every gate/up gradient before AdamW and every gate/up parameter after one
official Qwen3 B1/T32 step. PyTorch weights are transposed into microLLM's internal `[K,N]` layout;
names and shapes must match exactly.

The audit covers 56 tensors and 176,160,768 elements for gradients, plus the same amount for updated
parameters. Temporary safetensors payloads are not committed; the repository keeps every per-tensor
Max/RMS record and worker metadata.

## FP32 — pass

- Gradient Max/RMS: `3.109e-4 / 4.377e-7`;
- updated Parameter Max/RMS: `1.996e-5 / 5.645e-8`;
- all predeclared gates pass;
- worst gradient: `blocks.1.feed_forward.up_proj.weight`.

## BF16 forward with FP32 masters — reject

- Gradient Max/RMS: `0.25356 / 3.597e-4`;
- updated Parameter Max/RMS: `2.003e-5 / 2.626e-6`;
- Gradient Max fails the fixed `0.05` limit;
- Parameter RMS fails the fixed `2e-6` limit;
- worst gradient: `blocks.6.feed_forward.up_proj.weight`.

The export API is diagnostic-only: gradient serialization occurs before optimizer timing and marks
the worker profile `diagnostic`. No performance conclusion uses these runs.

This is a large but still partial model audit. Attention, norms, embeddings/output and optimizer
moment tensors remain for the next complete-family gate.

The subsequent complete-parameter audit now covers the first group above; this directory remains
the narrower historical attribution. Optimizer moments are still separate.

Repository regression closes at CPU 434/434 and ASan/UBSan 431/431. Coverage inventory reports
199 Tensor operators, 45 graph APIs and 159 registered test files.

Files:

- `fp32-raw.jsonl` / `bf16-raw.jsonl`: 112 per-tensor records each;
- `*-summary.json`: aggregate gates;
- `*-workers.json`: C++ and PyTorch worker metadata;
- `summary.json`: final decision.
