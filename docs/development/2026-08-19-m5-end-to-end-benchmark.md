# 2026-08-19 — M5 end-to-end benchmark and GPU-slower failure

## Contract

Measure synchronized train and cache-backed generation throughput separately from
operator Event time. Report warm-up, measured and setup-inclusive wall time, tokens,
loss, engine peak allocation, device/software metadata, and an output guard. Validate
emitted JSON with a real parser.

## Tiny-model results

| mode | CPU tokens/s | gfx942 HIP tokens/s | CPU/HIP result evidence |
|---|---:|---:|---|
| generate, B1, 1+8 tokens | 15,654.8 | 840.6 | identical output guard 550 |
| train, B1, context 8 | 11,991.2 | 1,017.2 | final loss 1.95585215 / 1.95585191 |

Setup-inclusive generation is 10,491.6 tokens/s CPU versus 223.5 HIP. Engine peak
allocation is 26,180 bytes for both generation paths. Training peak is 150,856 CPU
bytes and 150,792 HIP bytes, excluding external/runtime allocations.

## Stable failure

The AMD GPU path is much slower for this tiny workload. Two current explanations are:

1. many tiny kernel launches dominate computation;
2. cache growth, nonlinear backward, and AdamW still cross the host boundary.

A rebuttal experiment must remove one cause at a time: preallocate device KV cache
without changing kernels, then replace host AdamW/backward while preserving shapes.
Larger Model-S projection shapes must be measured separately; this tiny result cannot
be generalized to large GEMM.

## Evidence integrity failure found

The first end-to-end JSON string contained an extra quote after the driver version.
The executable still exited successfully. A Python `json.loads` schema smoke now runs
the micro and model benchmark executables and checks required fields, preventing an
invalid record from being accepted merely because the benchmark calculation ran.

Raw valid JSONL is committed as `benchmarks/results/2026-08-19-tiny-e2e.jsonl`.
