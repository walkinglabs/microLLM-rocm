# 2026-08-25 — current DeepSeek T2048 profile

![Current DeepSeek T2048 profile](../optimization-log/assets/current-deepseek-t2048-profile.svg)

当前T2048/B2/N64 profile使用1/3-step process delta去掉共同load/warm-up。cached Attention为
647.3ms/61.57%，GEMM为270.4ms/25.72%；1,792次Attention正好等于28层×64 token。

allocator增量为0，KV store只占0.65%。下一节点只进入score/context微架构矩阵，不直接改模型。
