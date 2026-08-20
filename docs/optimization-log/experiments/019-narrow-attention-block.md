# Experiment 019 — narrow cached-Attention blocks

Status: `discard`

## Hypothesis

The fused cached-Attention Kernel always used 256 threads even when Qwen head width is 64
and DeepSeek head width is 128. Choosing 64/128/256 threads by head width should reduce
idle lanes and reduction synchronization.

## Candidate

- template the fused Kernel by block size;
- use 64 threads for width <=64;
- use 128 threads for width <=128;
- retain 256 threads for wider heads;
- no equation, cache, model or measurement change.

Focused cached-Attention, GQA, generation and exact-token tests passed.

## Three-process result

| Model | Retained median | Candidate samples | Candidate median | Change |
|---|---:|---|---:|---:|
| Qwen generate | 154.60 | 155.06 / 144.35 / 135.13 | 144.35 | -6.6% |
| DeepSeek generate | 58.32 | 58.15 / 55.49 / 53.38 | 55.49 | -4.9% |

Training is unchanged. With the fixed PyTorch reference, the candidate score is
`1.791371`, below the retained `1.845199`.

## What was falsified

Fewer idle threads are not enough to predict a faster Attention Kernel. The larger block
may help occupancy, memory access scheduling or hide the serial dot-product loop. A
thread-count argument without measured throughput is not an optimization.

## Decision

`discard`. Qwen crosses the 5% regression gate, both model medians decline and the score
falls. Candidate code is removed; raw JSONL remains in [019-data](019-data/README.md).
