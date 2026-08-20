# 2026-08-20 — Experiment 013 unavailable GroupedGemm QKV

The hipBLASLt extension candidate kept Q/K/V weights and outputs independent and shared
only the decode input pointer. MI300X FP32 M=1 probes returned no grouped heuristic for
both `N={128,64,64}` and equal-width `N={128,128,128}` controls.

Both fallback outputs were numerically correct, but no grouped Kernel launched. The API,
model branch and extension dependency were removed before documentation was committed.

Decision: `discard`; revisit only for BF16 or larger-M prefill shapes.
