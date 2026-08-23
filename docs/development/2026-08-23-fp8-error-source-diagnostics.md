# FP8 weight/activation error-source diagnostics

Exp140 rejected a critical-block FP32 policy: the selected block amplified error but was not a
stable primary source. The next boundary therefore isolates each Linear operand without changing
the model, prompt or FP32 reference.

`Fp8DiagnosticMode::{Full,WeightOnly,ActivationOnly}` is part of `ModelConfig`. `Full` preserves
the native FP8 GEMM path. `WeightOnly` stores prepared FP8 weights, dequantizes each weight for an
FP32 GEMM and never quantizes activations. `ActivationOnly` retains FP32 weights, quantizes and
dequantizes activations, then uses FP32 GEMM. The latter two are inference-only counterfactuals and
must not be used for performance claims.

Preparation reports distinguish `linears_covered` from `converted_tensors`, so activation-only can
prove full model coverage while truthfully reporting zero converted weights. CLI JSON additionally
records the diagnostic mode, compute dtype, storage policy and machine counters.

The first focused run exposed an old programmatic runner fixture without the new field. The runner
now defaults such calls to `full`, while CLI parsing remains strict. Final Release/MI300 regression:
356/356 pass, CPU label 248, HIP label 108, two intentional environment skips. The HIP diagnostic
test proves 0 payload H2D/D2H after preparation and zero native FP8 GEMM dispatch for both isolated
modes.
