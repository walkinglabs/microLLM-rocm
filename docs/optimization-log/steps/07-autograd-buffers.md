# Step 07 — in-place gradient accumulation and buffer reuse

Status: `in progress` — local copy-on-write candidate discarded in Experiment 010

## Hypothesis

Allocating a new Tensor for every gradient branch accumulation contributes to training
allocation churn and launch count.

## Design

- add validated `add_` for same-shape/device/dtype Tensor;
- allocate leaf grad buffers lazily once;
- distinguish set-to-none from zero-in-place;
- release intermediate gradients after their backward closure;
- optionally assign reusable arena slots using liveness;
- preserve repeated-backward semantics.

## Required tests

- shared branch accumulation;
- same parameter used many times;
- repeated backward;
- view/transpose gradient order;
- zero_grad modes;
- full named Transformer gradients;
- no alias between parameter data and gradient.

## Falsification

If allocations fall but training time does not, serial CE/Norm or transpose copies still
dominate. Do not add a complex planner before simple in-place accumulation is measured.

## Keep gate

Correctness unchanged, allocation/launch count reduced, training rows improve or memory
drops materially without unexplained throughput regression.

## Experiment 010 result

A local `Storage::use_count()==1` condition was correct but reduced zero measured
allocations: Qwen/DeepSeek remained 9,200/10,715. Backward gradients commonly still
alias upstream nodes. The candidate was removed. Future work must explicitly model
contribution count/liveness; this step is not complete.
