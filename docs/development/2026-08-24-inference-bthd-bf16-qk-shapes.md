# Direct BF16 Q/K shape expansion

## Scope

Expand the exact BTHD BF16 Q/K path from B1/T512 to B1/T256, B1/T1024 and
B2/T512 without changing the default policy or tolerance.

## Evidence design

- two pinned official checkpoints;
- full last-row vocabulary for every batch row;
- exact retained-dispatch count per process;
- alternating case/policy order;
- three-process pilot retained, then five-process formal matrix;
- complete-logit, per-row top-1, throughput and engine-peak gates.

The pilot failed only Qwen B2/T512 performance at 1.0091x. The five-process
matrix passes all six cases at 1.0128x to 1.0244x with bit-exact logits and
unchanged peak. See [Experiment 206](../optimization-log/experiments/206-inference-bthd-bf16-qk-shapes.md).
