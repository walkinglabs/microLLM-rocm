# Contributor task contract

Copy this template into an issue or development record before implementation.

```text
ID:
Observed failure:
Goal — what changes in this task only:
Allowed files and interfaces:
Forbidden dependencies or changes:
Public interface:
State and ownership changes:
Invariants:
CPU/reference oracle:
Required positive tests:
Required negative/boundary test:
HIP hardware/software matrix, if relevant:
Benchmark method, if relevant:
Commands to run:
Evidence to retain:
Review assumptions:
Decision — accept, modify, or reject:
```

## Example: Tensor transpose view

```text
Observed failure: a raw pointer cannot represent a transposed matrix.
Goal: add a zero-copy transpose view.
Allowed files: core Tensor implementation and core tests.
Forbidden changes: HIP, autograd, public dtype semantics.
Invariants: Storage is shared; logical values follow swapped shape/strides.
Oracle: hand-calculated 2x3 matrix.
Negative test: out-of-range dimension.
Evidence: deterministic values, aliasing test, sanitizer run.
```
