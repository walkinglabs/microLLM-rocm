# Experiment 160 — BF16 training solution indices remain diagnostic

## Hypothesis

Experiment 159 attributes 53.47% of per-step Kernel time to hipBLASLt GEMMs. The default
heuristic may not choose the fastest solution for the eight unique BF16 Linear forward
shapes in Qwen/DeepSeek T512 training.

## Correctness-before-timing tuner

`microllm_tune_bf16_algorithms` queries up to 64 heuristic solution indices for one exact
M/K/N/output/workspace problem. For each index it clears the process-local registry,
registers only that exact shape, compares the complete output against default hipBLASLt,
and records default-Stream Event plus wall P50/P95 only after finite Max/RMS pass.

The formal runner executes eight shapes × three fresh processes. All 1,536 candidate
evaluations pass Max `1e-4` and RMS `1e-5`; observed worst errors are `2.801e-6/7.44e-7`.
Screening ends with zero registered solutions.

| Shape | Best common median speedup | Per-process best index |
|---|---:|---|
| Qwen Q/O | 1.189× | 98683 / 98676 / 98695 |
| Qwen K/V | 1.186× | 98587 / 98590 / 98591 |
| Qwen gate/up | 1.031× | 98864 / 98876 / 98887 |
| Qwen down | 1.099× | 98724 / 98724 / 98724 |
| DeepSeek Q/O | 1.083× | 98721 / 98710 / 98707 |
| DeepSeek K/V | 1.156× | 98606 / 98607 / 98607 |
| DeepSeek gate/up | 1.044× | 98919 / 98919 / 98932 |
| DeepSeek down | 1.112× | 98732 / 98769 / 98860 |

Only Qwen down selects the same best index in all three processes. The formal summary
chooses the lowest median common-passing index, not a single fastest sample.

## Same-revision model rebuttal

The training CLI accepts explicit process-local `M:K:N:index` records. It does not change
the default registry or write a cache. Three policies run three fresh processes per model:

| Policy | Qwen speedup | DeepSeek speedup | Max peak ratio |
|---|---:|---:|---:|
| all four shapes | 0.995× | 1.005× | 1.00028× |
| remove gate/up | 1.020× | 1.007× | 1.00028× |

Neither passes 1.05 on either model. Final-loss relative differences stay below 0.328%,
and the observed parameter guard remains equal, so this is a performance rejection rather
than a correctness failure.

![BF16 training solution discard](../assets/bf16-training-solution-discard.svg)

## Decision

Keep the complete-output solution tuner, fresh-process matrix and explicit research CLI.
Do not register a model default and do not persist these indices. Version-local indices,
unstable per-process winners and an absent end-to-end win are insufficient for a mature
automatic policy.

Raw evidence is in
[`benchmarks/results/2026-08-23-bf16-training-solutions/`](../../../benchmarks/results/2026-08-23-bf16-training-solutions/).
