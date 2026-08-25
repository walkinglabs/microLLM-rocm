# Step 105 — Cached Attention score/context microarchitecture

Status: scoped gfx942/BF16/uniform T>=2048 auto policy implemented; default-path check pending

Experiment 281证明当前DeepSeek T2048/B2/N64的cached Attention占Kernel时间61.57%，单次约
361.2us；GEMM第二，KV store和allocator不是主因。

第一阶段先拆算子，不接模型：

1. `cached_gqa_attention_scores`测Q·K；
2. softmax测score→probability；
3. `cached_gqa_attention_context`测P·V；
4. fused current作为端到端operator baseline；
5. 候选只改变一个score或context微架构。

诊断接口已经完成。它们像把一台封闭机器拆成三个透明盒子：先看每个Q·K分数，再看softmax
概率，最后看P·V context。CPU手算、PyTorch oracle和HIP均比较完整输出，不用几个抽样点代替
整张Tensor。HIP矩阵覆盖DeepSeek H12/KV2/D128、B1/B2、FP32/BF16 cache，以及
T31/32/33、T511/512/513和T2048，共16个case；运行期间没有payload H2D/D2H。

24个新进程已经完成矩阵。透明softmax占65.46%–73.56%，fused比透明pipeline快2.72x–4.16x，
BF16 fused比FP32快1.313x–1.534x。透明比例不冒充fused内部归因。

144进程搜索中，八个winner Event为2.381x–8.096x，wall为2.084x–6.988x；S1全失败、S2全
过门，T512选S16，T2048选S16/S32。完整精度和资源门通过，因此只准入显式官方DeepSeek模型A/B。
模型三对速度达到2.2223x且token/peak/KV通过，但完整logits Max/RMS为0.05691/0.01370，精度拒绝。
下一候选并行物化逐position score，再用保留原归约/P·V顺序的finalize Kernel；仍不改默认。
