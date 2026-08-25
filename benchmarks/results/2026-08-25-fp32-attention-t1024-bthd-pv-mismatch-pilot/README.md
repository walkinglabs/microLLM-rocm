# BTHD PV descriptor mismatch pilot

The current BTHD model path does not consume the generic PV solution key emitted
by `microllm_tune_fp32_attention_algorithms`. Qwen baseline and QK-only processes
completed, but the first PV-only process reported one registered entry, 175
registry misses and zero dispatches. The runner stopped before producing a false
performance comparison.

The reason is structural: current `attention_probability_value_bthd` uses an
interleaved-V hipBLASLt descriptor, while the standalone PV tuner screens the
standard BHTD descriptor. The pilot is retained to prevent reusing index 294867
as if it applied to the model path.
