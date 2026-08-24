# Experiment 172 — real allocation savings do not remove device work

Status: `discard`

## Observed bottleneck

Experiment 165 attributes `0.859 ms/step` and 168 calls to gradient/elementwise add on
Qwen T512. The older generic copy-on-write experiment did not know which targets were
actually exclusive and reduced no measured allocation. The retained graph now has
source-aware Autograd diagnostics, so ownership can be measured instead of guessed.

One diagnostic step finds:

| Model | Dense adds | Exclusive contiguous candidates | Elements | Exact target |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 121 | 72 | 33,030,144 | `reshape [512,896]` |
| DeepSeek Distill 1.5B | 140 | 84 | 66,060,288 | `reshape [512,1536]` |

Residual `[1,T,D]`, embedding and tied-leaf destinations are shared and are correctly
excluded. This is a new, narrower source boundary rather than a retry of “mutate every
gradient if it looks writable.”

## Hypothesis and rejection rule

If those 72/84 destinations receive their second contribution in place, each measured
step avoids that many Tensor allocations. Keep only if both official B1 T512 models reach
at least `1.01×`, loss differs by at most 0.5%, the fixed parameter is equal, peak does not
increase, and diagnostics prove that only eligible destinations execute.

Any model below `1.01×` rejects the default even when allocation counters improve.

## Safety contract

- Autograd gradients remain FP32 and device-local;
- the destination and source must be contiguous and shape/device compatible;
- one temporary `Storage` owner plus one node owner is the only mutable case;
- a source alias or any other graph/view owner raises the count and falls back;
- the public primitive rejects partially overlapping views and permits an exact self-add;
- no synchronization, host payload copy, tolerance change or reference deletion is allowed.

CPU tests prove exact values, one saved allocation and the shared `add(x,x)` fallback. HIP
tests preserve the destination address and report zero H2D/D2H/D2D payload transfers. The
installed-package consumer links and executes the new public symbol.

The numerical function is the existing `add`, whose FP32/FP16/BF16 shape matrix and PyTorch
oracle pass in the 247-test PyTorch-enabled build. `add_in_place_` is registered as operator
infrastructure because its new contract is ownership, aliasing and address preservation—not a
different mathematical result. Those state properties are checked directly on CPU and HIP.

## Same-binary official result

Both layout fusions and tied-embedding sparse accumulation stay enabled. Rejected plan,
alpha, paired-repeat and zero-stride policies stay disabled. Only
`--unique-gradient-inplace-add false/true` changes. Each policy/model uses three fresh
processes, alternating order, one warm-up and two measured steps.

| Model | Allocating | In-place | Speedup | Saved allocations | Peak ratio | Loss relative diff | Parameter equal |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 14,969.66 | 15,032.11 tok/s | 1.0042× | 144 | 1.000 | 0.0279% | yes |
| DeepSeek Distill 1.5B | 6,267.24 | 6,236.90 tok/s | 0.9952× | 168 | 1.000 | 0 | yes |

Separate one-step diagnostics report `72/72` and `84/84` eligible/executed for the enabled
policy and zero executed for the disabled policy. Strided-copy counters remain zero.

## Profiler rebuttal

The Qwen profile runs the same binary and same load plus three training steps:

| Counter | Allocating | In-place | Delta |
|---|---:|---:|---:|
| Engine allocation calls | 5,010 | 4,794 | -216 |
| Engine cache reuse calls | 3,983 | 3,767 | -216 |
| Backend allocation calls | 1,027 | 1,027 | 0 |
| HIP allocation/free calls | 2,071 / 452 | 2,071 / 452 | 0 / 0 |
| All Kernel calls | 6,905 | 6,905 | 0 |
| FP32 add Kernel calls | 504 | 504 | 0 |
| Add Kernel time | 2.292 ms | 2.374 ms | +0.082 ms |
| Peak engine bytes | 11,824,424,968 | 11,824,424,968 | 0 |

The candidate reuses the destination but still launches the same add Kernel. The removed
output blocks were already served by the exact-size engine cache, so no backend allocation
or HIP API call disappears. This directly explains the end-to-end plateau.

![Unique-gradient in-place accumulation discarded](../assets/unique-gradient-inplace-add-discard.svg)

## Decision

Reject as the production default. Retain `add_in_place_`, the exclusive-owner guard,
diagnostic candidate/executed counters and an explicit same-binary switch because they are
tested infrastructure for a future liveness planner. Do not retry another local owner-count
predicate: a useful next proposal must remove add Kernel work, shorten gradient lifetime, or
plan buffers across multiple graph nodes.

Raw evidence is in
[`benchmarks/results/2026-08-24-unique-gradient-inplace-add/`](../../../benchmarks/results/2026-08-24-unique-gradient-inplace-add/).
