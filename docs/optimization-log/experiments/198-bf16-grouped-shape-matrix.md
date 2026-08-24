# Experiment 198 — grouped capability at rows 256 and 1024

Status: keep capability evidence; production policy not broadened yet

## Question

The retained policies are exact T512 entries. Do QKV and gate/up remain supported and faster at
flattened rows 256/1024, or was the grouped win a single-shape accident?

Every one of eight cases runs three fresh processes and screens 64 complete-output candidates.

| Rows | Model | QKV user args | Gate/up user args | QKV winners | Gate/up winners |
|---:|---|---:|---:|---|---|
| 256 | Qwen | 1.695× | 1.339× | 64713/64752 | 65197 |
| 256 | DeepSeek | 1.604× | 1.236× | 64699/64713 | 65168 |
| 1024 | Qwen | 1.389× | 1.124× | 64713/64754/64755 | 65168/65200 |
| 1024 | DeepSeek | 1.397× | 1.225× | 64754/64755 | 65183/65212 |

![Grouped shape capability](../assets/bf16-grouped-shape-matrix.svg)

All cases pass numerical gates. The slowest user-arguments Event ratio is still 1.124×. Per-call
reinitialization is below 1.0 in all eight three-process medians.

## Counterexample correction

The initial one-process DeepSeek rows256 QKV pilot reported reinitialization at 1.051×. Formal
repetition gives 0.964×. This is why a single process cannot redefine the lifetime design.

## Decision

Keep the cross-shape benchmark evidence. Do not broaden model policy from operator results alone.
The next node must test B1/T256, B1/T1024 and B2/T512 complete models, because rows1024 shares a
projection shape across two different Attention/batch workloads.

Raw evidence:
[benchmarks/results/2026-08-24-bf16-grouped-shape-matrix/](../../../benchmarks/results/2026-08-24-bf16-grouped-shape-matrix/).
