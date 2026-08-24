# Experiment 171 — forward-only Value broadcast closes zero-stride routing

## Candidate

The final distinct variant uses zero-stride Value only for D>=128 forward P×V. Backward dP
returns to expanded V plus one H-batched GEMM. K and dV remain unchanged. Qwen D64 must not
route.

The T256/D128 HIP test compares three paths against one control: expanded, forward-only,
and full forward+dP broadcast. All outputs, probabilities and Q/K/V gradients pass.

## Same-binary official result

| Model | Disabled | Forward-only | Speedup | Allocations saved | Peak ratio | Parameter equal |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B D64 | 15,223.30 | 14,951.71 tok/s | 0.9822× | 0 | 1.000 | yes |
| DeepSeek Distill D128 | 6,334.44 | 6,339.86 tok/s | 1.0009× | 56 | 1.000 | no |

Qwen executes the same route; its difference is process drift. DeepSeek's P×V result uses a
different GEMM grouping and changes the fixed parameter from `2.124970913` to
`2.124971151`. Loss relative differences remain under 0.246%.

## Profile closure

DeepSeek across three executed steps:

```text
repeat forward calls 336 → 252
repeat backward calls 168 → 168
all dispatches       8,058 → 8,058
all Kernel time      261.730 → 264.039 ms
```

The one isolated P×V benchmark favored broadcast, but the full process replaces each of 84
removed repeat launches with an extra KV-group GEMM launch. It neither lowers peak nor
survives the full schedule.

![Forward-only GQA Value broadcast discarded](../assets/forward-only-gqa-value-broadcast-discard.svg)

## Decision

Default false and close zero-stride model routing. The P×V/dP primitives remain valid
backend capability tests, but universal, width-selective full, and width-selective
forward-only policies all have explicit counterexamples. The next training work must leave
this search family.

Raw evidence is in
[`benchmarks/results/2026-08-23-forward-only-gqa-value-broadcast/`](../../../benchmarks/results/2026-08-23-forward-only-gqa-value-broadcast/).
