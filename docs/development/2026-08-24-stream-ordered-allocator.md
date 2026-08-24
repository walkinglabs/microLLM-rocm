# HIP Stream ordered allocator probe

## Scope

This node adds an explicit Beta HIP allocation primitive and measures it against the retained
deferred-lifetime control. It does not replace ordinary Storage or enable a model policy.

## Implementation

- capability query via `hipDeviceAttributeMemoryPoolsSupported`;
- default-pool release threshold, trim and reserved/used current/high diagnostics;
- move-only `StreamOrderedHipBuffer` using async allocation/release on one Stream;
- capture test proving allocation/free Graph nodes execute safely;
- three-policy benchmark, 72-process matrix and profiler attribution.

## Decision

Correctness and address reuse pass, performance does not. Eager async reaches 0.619×–0.709× of
deferred. Graph allocation-node replay reaches 0.036×–0.048× and owns N distinct addresses. The
primitive remains explicit for future experiments; Tensor/model routing remains unchanged.

Full report: [Experiment 179](../optimization-log/experiments/179-stream-ordered-allocator.md).
