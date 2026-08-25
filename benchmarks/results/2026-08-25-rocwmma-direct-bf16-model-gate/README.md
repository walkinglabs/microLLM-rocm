# Direct-BF16 online-Attention model rebuttal

This directory repeats the Experiment 230 full-model matrix after eliminating
all three Attention-core casts:

- grouped QKV retains BF16 query, key and value outputs;
- BF16 value bias writes BF16 directly;
- fused bias + split-half RoPE writes BF16 Q/K directly;
- the online operator consumes those tensors without a cast.

The same two pinned models, B1T256/B1T1024/B2T512 cases, three fresh processes,
2 warm-up and 5 measured prefills are used. All 36 processes hit exactly 168 or
196 native calls and zero fallbacks.

The hypothesis is rejected. Direct BF16 improves every prior three-cast ratio
slightly, but full-model throughput remains only 0.777×–0.906× current. Peak
memory still falls by 3.5–57.0 MiB and top tokens stay equal; Qwen complete-logit
Max/RMS remains as high as 0.485/0.110. The public primitives are retained and the
model route remains disabled.
