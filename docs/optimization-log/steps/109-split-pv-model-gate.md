# Step 109 — Split-P×V official-model gate

Status: planned

Experiment 291的16个operator case全部选择S16并通过性能门。现在固定DeepSeek
T2048/B2/BF16/N64，三对fresh process：

1. current显式materialized=true、split-PV=0；
2. candidate显式materialized=false、split-PV=16；
3. 保存每步完整cached logits，共303,872值/进程；
4. 比较Max/RMS、64 token、generation/cache-prepare、peak与KV bytes；
5. 记录逻辑/backend allocation和cache reuse；
6. 三组leave-one throughput都需过1.05；
7. logits必须通过现有严格门；top-1相同不能替代完整分布。

若精度失败，Step 108到此关闭并说明P×V重排也会放大；若通过，才扩展Qwen/DeepSeek的T512/T2048
边界，不能直接加入Auto。
