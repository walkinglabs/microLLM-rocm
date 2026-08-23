# FP8 both-roundtrip diagnostic

Exp141 proves that isolated weight and activation rounding both matter, with different dominance
across Qwen, DeepSeek, context and metric. It cannot distinguish joint operand rounding from native
FP8 GEMM execution because neither isolated mode quantizes both operands.

`Fp8DiagnosticMode::BothRoundtrip` closes that causal boundary. It uses the exact existing weight
preparation and dynamic activation quantization, dequantizes both operands to FP32, and executes the
same `MatmulImplementation::Auto` FP32 boundary as the one-sided diagnostics. It is intentionally
slow and never enters `ops::fp8_matmul`.

The CLI spelling is `--fp8-diagnostic-mode both-roundtrip`. JSON reports
`fp32_gemm_with_fp8_roundtrip_both_operands`; the precision policy includes both selected scale
modes. This prevents the diagnostic from being confused with a native FP8 performance run.

The existing CPU/HIP diagnostic tests now cover all three counterfactuals. Before and after
preparation are value-identical, prepared weights and dynamic-call counts match the selected
operands, the HIP hot path has zero payload transfers, and native/fallback FP8 GEMM counters remain
zero. The CLI and matrix contracts also reject a stale binary or unsupported spelling.
