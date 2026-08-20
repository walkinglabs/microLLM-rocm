# Experiment 042 — official Qwen BF16 training shape baseline

Status: `baseline` for the new batch/context track

## Why this track exists

The earlier official training comparison used one tiny shape: batch 1, context 3. That
proves a training step can run, but it says almost nothing about ordinary matrix shapes.
Experiment 042 first made batch explicit in both engines, then measured four shapes with
fresh processes and alternating framework order.

Each process runs one warm-up and two measured updates. Each table value is the median of
three microLLM processes and three PyTorch processes. microLLM uses BF16 Linear forward,
FP32 masters and persistent BF16 mirrors; PyTorch uses BF16 autocast with FP32 parameters.

## Result

| Shape B×T | microLLM | PyTorch | Throughput ratio | Peak-memory ratio |
|---|---:|---:|---:|---:|
| 1×3 | 18.79 tok/s | 45.46 tok/s | 0.413× | 0.967× |
| 2×3 | 30.39 tok/s | 89.07 tok/s | 0.341× | 0.968× |
| 1×32 | 60.02 tok/s | 457.69 tok/s | 0.131× | 0.984× |
| 1×128 | 654.78 tok/s | 1861.96 tok/s | 0.352× | 1.053× |

![Official training shape baseline](../assets/bf16-training-shape-matrix.svg)

All 24 raw rows have finite losses, changed parameters and the exact
`batch × context × steps` trained-token count. Every microLLM optimizer window reports zero
host-to-device and device-to-host Tensor payload calls.

## What the curve says

The gap is not monotonic in context. microLLM `1×32` takes about 534 ms per step, while
`1×128` takes only about 195 ms even though it has four times more tokens. The time charged
to the optimizer synchronization boundary is about 488 ms at context 32 and 125 ms at
context 128. That boundary includes earlier asynchronous work, so it does not prove AdamW
itself changes cost with context.

This curve rejects a simple “longer Attention alone explains the gap” story. The next
experiment must profile context 32 and 128 and classify GEMM, reduction, cast, allocation
and synchronization time. No performance optimization is retained by this experiment; it
creates the baseline against which the next candidate must be compared.

DeepSeek and batch>1 at longer contexts remain separate matrix rows, not implied by this
Qwen result.
