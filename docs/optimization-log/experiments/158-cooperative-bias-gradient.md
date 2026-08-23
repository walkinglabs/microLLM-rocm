# Experiment 158 — cooperative bias-gradient rows

## Problem

The latest retained Qwen T512 profile left bias gradient at 216 calls and 118.18 ms.
The old Kernel assigned one thread to each output column and made that thread scan every
row serially. A naive one-block-per-column rewrite would expose row parallelism but turn
adjacent wave lanes into wide-stride loads.

## Candidate and rebuttal

Each 256-thread block covers 32 contiguous columns and eight row lanes. A lane accumulates
every eighth row for one column, so the 32 threads in each row lane still load contiguous
values. Shared memory stores an `8×33` tile; row lane zero combines eight partial sums.

`Auto` selects it for `rows >= 32`. Scalar remains below that boundary. Reject if complete
outputs exceed Max `3e-5` or RMS `1e-5`, either official model improves under 1.05×,
final loss differs over 0.5%, the observed parameter guard changes, or peak memory grows.

## Operator matrix

The fresh-process matrix contains 13 shapes × two implementations × three processes.
All 78 rows pass complete-output finite/Max/RMS gates.

| Shape | Cooperative speedup | Max error |
|---|---:|---:|
| 16×896 counterexample | 1.005× | 1.49e-7 |
| 32×128 boundary | 1.106× | 8.94e-8 |
| 512×128 | 3.260× | 2.09e-6 |
| 512×256 | 3.214× | 3.58e-7 |
| 512×896 | 3.231× | 2.12e-6 |
| 512×1536 | 3.274× | 8.59e-6 |
| 1024×256 | 4.224× | 2.38e-7 |

The 16-row case shows why the small fallback remains. Width 128/256/896/1536 all clear
the gate at 32 rows, so the threshold is measured rather than copied from another reduction.

## Same-revision official A/B

Candidate and Scalar-only baseline each use three fresh processes, BF16 Linear forward,
FP32 masters, T512/B1, one warm-up and two measured steps. The baseline temporarily raises
the Auto threshold beyond the executed domain, then the final worktree restores 32.

| Model | Scalar | Cooperative | Speedup | Optimizer speedup | Peak ratio |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 11,688.35 | 14,282.79 tok/s | 1.222× | 1.567× | 1.000× |
| DeepSeek Distill 1.5B | 5,525.16 | 6,140.58 tok/s | 1.111× | 1.210× | 1.000× |

Changed reduction order is not bit-exact. Worst complete operator Max/RMS are
`8.59e-6/3.95e-6`; worst final-loss relative difference is Qwen `0.442%`. The fixed
observed parameter after three updates remains equal for both models.

Same-workload rocprofv3 records 216 calls on both sides: target Kernel total falls
`26.00→4.01 ms` (`6.49×`) and its share falls `18.74%→3.44%`.

![Cooperative bias gradient](../assets/cooperative-bias-gradient.svg)

## Decision

Keep the 2D cooperative Kernel and `rows >= 32` Auto threshold. It passes complete output,
model loss/parameter, memory and both official speed gates. Keep the Scalar implementation
as the short-row reference and explicit diagnosis path.

Raw evidence is in
[`benchmarks/results/2026-08-23-cooperative-bias-gradient/`](../../../benchmarks/results/2026-08-23-cooperative-bias-gradient/).
