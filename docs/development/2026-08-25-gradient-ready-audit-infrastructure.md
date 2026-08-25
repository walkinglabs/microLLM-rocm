# 2026-08-25 — gradient-ready order audit infrastructure

## Why this diagnostic exists

Real overlap is possible only if complete buckets become ready before backward ends. Parameter
order, reverse graph traversal, tied/shared leaves and bucket limits determine that condition; a
synthetic overlap benchmark cannot prove it.

## Contract

- hooks are accepted only on differentiable leaves;
- backward counts every incoming leaf contribution before traversal;
- a hook runs after accumulation of the final contribution, exactly once per backward;
- repeated backward recomputes counts and clear removes the callback;
- DataParallel audit requires a complete parameter permutation on every rank;
- both rank orders must match or the step fails;
- ordinary training installs no hooks and constructs no ready-state table.

The CLI exports 57 Model-S parameter names, element counts and two rank orders. The runner rebuilds
the same 25 MiB ranges as the C++ reducer and computes each bucket's completion position.

## Smoke evidence

CPU shared/repeated/clear/error tests pass. The two-rank tiny test produces a complete stable
permutation on both steps. Model-S produces exact reverse parameter order: bucket 2 completes at
1/57, bucket 1 at 35/57, and bucket 0 at 57/57. The synchronization path remains unchanged until
the formal three-process audit confirms this structure.

Infrastructure gates pass: focused CPU hook semantics, RCCL-labelled `32/32`, 42 graph API entries
and 120 registered native/Python test sources.
