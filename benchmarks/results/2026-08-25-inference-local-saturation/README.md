# Current inference local-saturation audit

This audit combines the retained post-Norm profile with the measured rejection
gates since Experiment 232. Remaining casts occupy 2.694%/1.841% of Qwen/
DeepSeek Kernel time, so even free deletion would cap Kernel-only speedup at
1.0277x/1.0188x. Both direct hipBLASLt mixed-dtype routes are unsupported.

Softmax thread tuning, online rocWMMA model routing, exact Attention solutions,
vectorized SwiGLU and grouped Swish all have full-model counterexamples. GEMM is
61.5%/68.8%, but another exact-index search is also closed. The next inference
work must introduce a different graph-wide/custom-kernel architecture or a new
backend/hardware matrix; local default-policy knob search is saturated.
