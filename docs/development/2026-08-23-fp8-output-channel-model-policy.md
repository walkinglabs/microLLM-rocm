# FP8 output-channel model policy

`Fp8WeightScaleMode::OutputChannelAmax` connects the proven column-scale operator to every FP8
Linear. The CLI spelling is `--fp8-weight-scale-mode output-channel-amax`; it remains opt-in.

Preparation converts each `[K,N]` FP32 Linear weight once, retains one byte per element plus `N`
FP32 device scales, and releases the FP32 source. `Fp8WeightPreparationReport::linears_covered`
remains the model coverage count, while `scale_bytes_retained` now uses the actual scale Tensor size
instead of assuming one scalar per Linear.

Prepared `Linear` state preserves `Fp8ScaleMode::OuterColumn`. Hot inference reuses it without
rescanning weights; native shapes execute scalar-scale FP8 GEMM and one output-column correction,
while unsupported shapes dequantize the full per-column representation in the existing fallback.

Tiny CPU evidence covers all eight Linear weights and 80 retained column scales. The HIP model gate
proves eight device-only preparations, no weight payload D2H, zero hot-path H2D/D2H, no repeated
column quantization and unchanged dynamic activation-call semantics. CLI JSON exposes preparation
column calls/elements and native post-scale calls.

This is an experiment policy, not a new default. Official Qwen/DeepSeek complete-logit, memory and
throughput gates decide whether it is kept, rejected, or narrowed to selected Linear families.

The official matrix schema retains FP8 weight bytes and scale bytes as separate fields. This is
required because a lower total can otherwise hide an unexpectedly large vector-scale allocation.
