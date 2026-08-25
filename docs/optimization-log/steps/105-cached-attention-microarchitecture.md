# Step 105 — Cached Attention score/context microarchitecture

Status: split-sequence candidate and S1/2/4/8/16 matrix runner implemented; full timing pending

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

候选已经只改变sequence并行分解：每个head使用多个partial blocks，再用log-sum-exp合并。它必须
提交完整context Max/RMS、warm-up 3 + measured 20 Event/wall、0 payload transfer/热backend
allocation，以及T512/B2反例。至少1.05x才允许进入官方模型；否则拒绝并关闭这一候选。
