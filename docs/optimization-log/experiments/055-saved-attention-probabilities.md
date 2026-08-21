# Experiment 055 — saved long-sequence Attention probabilities

## Hypothesis

Experiment 054 still recomputes QK scores, max, exp and denominator during backward. At
T=512 the row stage costs 473.91 ms per traced process. Saving forward probabilities may
trade memory for less repeated compute.

## Design

- only autograd T≥256 with hipBLASLt saves `[B,H,T,T]` FP32 probabilities;
- forward zeroes the upper triangle and writes normalized lower-triangle rows;
- backward computes probability gradients/softmax backward from saved values, then reuses
  Experiment 054 batched GEMMs for K/V;
- standalone operator calls, short sequences and no-library builds retain recomputation.

The first T=256 test exposed uninitialized future probabilities. Both saved matrices are
now explicitly zero-filled before causal row writes; Q/K/V gradients then match CPU.

## Official speed/memory result

| Model | Before | After | Speedup | Measured peak cost | PyTorch ratio |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1103.05 | 1248.17 tok/s | 1.132× | +336 MiB (+2.78%) | 0.160× |
| DeepSeek Distill 1.5B | 546.07 | 627.83 tok/s | 1.150× | +336 MiB (+0.95%) | 0.128× |

![Saved Attention probabilities](../assets/saved-attention-probabilities.svg)

T=128 stays on the old path: one process is `0.991×` with identical peak, inside the 5%
no-regression gate.

## Retained profile

Qwen saved-row backward falls `473.91→305.15 ms` (`1.553×`). Forward rises only
`269.69→272.52 ms` from global probability writes. Total Kernel time falls
`1.442→1.284 s` (`1.123×`); dispatch count is unchanged and HIP API calls change by 0.08%.

## Decision

Keep as an explicit long-sequence speed/memory trade-off. The fixed 336 MiB cost must stay
visible in model reports. Forward and saved-row kernels are now similar 272/305 ms hotspots;
parity still needs tiled score/context computation rather than another full-row scalar pass.
