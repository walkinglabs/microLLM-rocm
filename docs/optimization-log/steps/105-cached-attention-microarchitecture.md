# Step 105 — Cached Attention score/context microarchitecture

Status: planned

Experiment 281证明当前DeepSeek T2048/B2/N64的cached Attention占Kernel时间61.57%，单次约
361.2us；GEMM第二，KV store和allocator不是主因。

下一步先拆算子，不接模型：

1. `cached_gqa_attention_scores`测Q·K；
2. softmax测score→probability；
3. readable context测P·V；
4. fused current作为端到端operator baseline；
5. 候选只改变一个score或context微架构。

固定DeepSeek H12/KV2/D128，T512/T2048，B1/B2，FP32/BF16 cache。每个候选必须提交完整
score、probability、context Max/RMS，CPU/HIP/PyTorch门，warm-up 3 + measured 20 Event/wall，
以及T2048反例。算子至少1.05x且完整context门通过，才允许进入官方模型；否则拒绝。
