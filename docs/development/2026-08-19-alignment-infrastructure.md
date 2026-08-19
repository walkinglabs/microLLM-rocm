# 2026-08-19 — cross-framework alignment infrastructure

## Goal

Run one model in microLLM and PyTorch with identical parameters and inputs, compare
function/layer/model values, and measure both sides without mixing diagnostic copies
into accepted timing.

## Implementation

- public C++ `TraceSession`, `ScopedTraceSession`, `TraceTimer`, records, statistics,
  JSONL export, capture limit, and CPU/HIP synchronization mode;
- automatic forward operator instrumentation in the eager autograd path;
- model checkpoints for input, embedding, each block, final norm, logits, and forward;
- `microllm_alignment` forward values, training loss/gradients, operator timing, layer
  timing, and backward timing runner;
- PyTorch runner rebuilt from the exact microLLM parameter trace;
- comparator for shape/dtype, full-value tolerance, max abs/rel, MSE, cosine, error
  index, timing median/p95, and PyTorch/microLLM ratio;
- orchestrator that preserves commands, tool versions, git state, raw traces, logs,
  comparison JSON, Markdown report, and artifact manifest.

## Failure found during bring-up

The first function-level comparison found gate/up FFN matmul records swapped even while
final logits matched. `gate_.forward(...)` and `up_.forward(...)` were passed as separate
C++ function arguments, whose evaluation order was not a stable trace contract. The
model now evaluates and names gate and up projections explicitly before SwiGLU.

## Measured smoke

- CPU microLLM versus PyTorch CPU: 45/45 checkpoints pass;
- maximum absolute difference: 8.34465e-7;
- 39 operator timing checkpoints;
- 4 layer/model timing checkpoints;
- MI300X microLLM versus PyTorch CPU numerical pass: 58/58 forward/loss/gradient
  checkpoints;
- maximum absolute difference in that extended run: 3.3378601e-6;

The extended CPU training trace passes 58/58 total checkpoints: the original 45
forward checkpoints, one cross-entropy loss, and all 12 named parameter gradients.
The loss is bit-identical for the fixed input; the largest gradient absolute difference
is 1.4305115e-6, below the configured `3e-5 + 3e-5 × |reference|` gate. Backward has a
separate five-repetition timing record so its work is not hidden in forward time.

The HIP/CPU timing ratio is diagnostic only because the two frameworks ran on different
devices. A direct speed claim waits for a working PyTorch ROCm environment and identical
workload/hardware.
