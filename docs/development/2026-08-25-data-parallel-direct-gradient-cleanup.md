# 2026-08-25 — rejected direct-gradient route cleanup

Experiment 259 proved that pre-seeding bucket views removes all reducer copies but regresses the
retained view path end to end. The rejected model route is therefore removed from:

- `DataParallelConfig` and trainer control flow;
- the distributed CLI and recorded-run wrapper;
- gradient-bucket stats and reducer parameters;
- candidate benchmark runner and its schema test;
- route-specific RCCL tests and test-file registration.

The generic leaf-only `Value::set_grad_accumulation_target` API remains. Its CPU tests cover
preset-value accumulation, branching, repeated backward, shared-Storage embedding, and rejection
boundaries. It is not wired into DataParallel and exists only as a producer out-kernel foundation.

Post-cleanup gates pass: CPU `361/361`, ASan/UBSan `359/359`, RCCL-labelled `30/30`, optimization
log validation, and 118 registered native/Python test sources. The log validator also fails if the
rejected config or CLI flag reappears.
