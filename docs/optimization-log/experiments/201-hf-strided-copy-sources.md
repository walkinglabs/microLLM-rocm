# Experiment 201 — every remaining layout copy belongs to Attention

Status: keep diagnostic; select inference BTHD Attention island

## Diagnostic contract

StridedCopyRecord now includes the existing AllocationSource. Source participates in aggregation,
so equal shape/stride copies from different model regions cannot merge. ScopedAllocationSource
activates when allocation diagnostics, strided diagnostics, or both are enabled; both disabled
retains the old no-op path.

hf_infer accepts an explicit one-prefill/zero-warmup flag and emits source, device, element size,
shape, strides, calls, elements and bytes.

## Formal attribution

| Model | Total | Attention layout | Attention core |
|---|---:|---:|---:|
| Qwen | 96 calls / 100.7 MB | 72 / 56.6 MB | 24 / 44.0 MB |
| DeepSeek | 112 / 205.5 MB | 84 / 117.4 MB | 28 / 88.1 MB |

All six processes produce identical three-record sets.

![Strided-copy source attribution](../assets/hf-strided-copy-sources.svg)

The exact layouts prove three BTHD→BHTD copies for Q/K/V and one BHTD→BTHD context copy per
block. No remaining copy belongs to FFN, embedding or output.

## Decision

The next candidate is an inference BTHD Attention island spanning RoPE input layout through context
output. Do not optimize the generic copy Kernel: that would move the same avoidable bytes faster
instead of removing them.

Raw evidence:
[benchmarks/results/2026-08-24-hf-strided-copy-sources/](../../../benchmarks/results/2026-08-24-hf-strided-copy-sources/).
