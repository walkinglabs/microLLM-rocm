# Experiment 010 — copy-on-write in-place gradient accumulation

Status: `discard`

## Ownership finding

Autograd's first gradient assignment can share Storage with an upstream gradient or
another leaf. Blind in-place accumulation would mutate both aliases. Storage already
exposes `use_count`, so uniqueness can be checked without a global liveness planner.

## Hypothesis

When a gradient buffer is uniquely owned, device-native `add_` can replace allocation
plus out-of-place add. Aliased cases retain the old behavior. Training logical/backend
allocation counts should fall without changing repeated-backward semantics.

## Scope

- add validated FP32 contiguous `add_`;
- use it only when the existing gradient Storage has one owner;
- aliases, first assignment and non-contiguous preparation keep existing semantics;
- no zero-grad mode change, arena planner, optimizer or model change.

## Required gates

- [x] shared leaves and graph branches
- [x] same parameter used repeatedly
- [x] repeated backward and zero_grad
- [x] no parameter/gradient alias
- [x] CPU/HIP add_ and full Transformer gradients
- [x] measured allocator counts

## Candidate

- added validated contiguous FP32 `add_` using the existing alias-safe elementwise Kernel;
- first gradient kept existing assignment behavior;
- a defined gradient used `add_` only when its Storage use count was one;
- aliased Storage retained the old out-of-place `add`;
- focused tests covered two leaves sharing their first gradient, repeated backward,
  self-use, address stability and complete HIP Transformer gradients.

## Falsifying measurement

The primary metric was allocation reduction. It did not move at all:

| Workload, five measured steps | Baseline logical allocations | Candidate |
|---|---:|---:|
| Qwen train | 9,200 | 9,200 |
| DeepSeek train | 10,715 | 10,715 |

Backend allocation counts also stayed in the same normal range. The single candidate
timing was 118.28/68.79 token/s, but Experiment 008 already proved this magnitude of
cross-process training variation can occur with no training-code change. With zero
allocation reduction, it is not evidence for the candidate.

## Interpretation

Backward-produced gradients usually remain aliased to an upstream node while
accumulation happens, so the uniqueness condition rarely opens the in-place path. A
useful next design would require explicit contribution counts or liveness/ownership
transfer, not another local `use_count` condition.

## Correctness and cleanup

Focused CPU/HIP/autograd/Transformer tests passed. The allocation hypothesis failed
before full regression or three-process performance was warranted. Candidate source and
tests were removed, returning framework code exactly to Experiment 009.

## Results

Falsified: copy-on-write uniqueness did not reduce any measured Tensor allocations.

## Decision

`discard`. No `add_` API or autograd behavior change remains.
