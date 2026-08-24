# Experiment 195 — grouped gate/up capability before model routing

Status: keep benchmark capability; production unchanged

## Question

The retained BF16 FFN Arena casts its input once, then submits gate and up as two separate GEMMs.
They have the same rows, inner dimension, output width and input pointer. Does hipBLASLt support
them as a two-operation GroupedGemm, and is the pointer-stable path fast enough to justify a
complete-model experiment?

The capability gate requires:

- exact Qwen and DeepSeek T512 BF16-output shapes;
- three fresh processes per model;
- 64 complete-output candidate checks before timing;
- bit-exact outputs;
- at least 1.05× Event speedup with device user arguments on both models;
- an explicit per-call reinitialization counterexample.

## Result

| Model | Algorithms | Passing | Stable Event | User arguments | Reinitialized | Setup |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 10227 | 64/64 | 1.203× | 1.188× | 0.823× | 0.0537 ms |
| DeepSeek | 10227 | 64/64 | 1.139× | 1.155× | 0.940× | 0.0524 ms |

All six processes are bit-exact. Qwen selects indices 65168 or 65198; DeepSeek selects 65200.
The small argument setup shown here excludes the shared kernel initialization that a production
plan must report separately.

![Grouped gate/up capability](../assets/bf16-grouped-gate-up.svg)

## Decision

Keep the benchmark mode and proceed to a separate pointer-stable FFN Arena integration. Do not
route the model yet, and do not hard-code a model name. The next node must register an exact
environment/shape solution, bind each block's persistent weight pointers, and pass complete logits,
steady throughput, setup and peak-memory gates.

Raw evidence:
[benchmarks/results/2026-08-24-bf16-grouped-gate-up/](../../../benchmarks/results/2026-08-24-bf16-grouped-gate-up/).
