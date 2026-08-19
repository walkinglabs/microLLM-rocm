# 2026-08-19 — reusable RCCL data-parallel trainer

## Goal

Replace hand-written distributed-step test code with a reusable training API that owns
rank models, optimizers, batch validation, gradient reduction, update consistency, and
stage metrics.

## Implementation

- `DataParallelTrainer` with explicit device list, bucket byte limit, seed, model config,
  and AdamW config;
- same-seed rank model construction and rank-local GPU placement;
- one local batch per rank with equal target-count validation;
- rank-local forward/loss/backward;
- explicit backward completion before communication-stream reads;
- bucket pack, RCCL average all-reduce, unpack, and gradient assignment;
- identical rank-local AdamW updates;
- maximum rank parameter difference and stage wall times;
- unified profiling records for forward/backward, all-reduce, optimizer, and total step;
- `microllm_distributed_train` JSON metrics and trace CLI.

## Measured result

Three steps on two MI300X ranks:

```text
mean loss: 2.75452 → 2.05171 → 1.69936
maximum rank parameter difference: 0 on every step
bucket count: 1 for the configured tiny-model 4096-byte bucket limit
```

The dedicated test also compares all parameters with one CPU model trained on the
equivalent global batch for three steps.

## Research boundary

RCCL supplies collectives but not the full DDP reducer. Mature DDP marks gradients ready
from autograd hooks, launches an asynchronous all-reduce when every gradient in a bucket
is ready, and manages bucket views/work completion. The current implementation is the
synchronous correctness baseline; overlap, one-process-per-GPU, unused parameters,
uneven inputs, and distributed failure propagation remain explicit next work.
