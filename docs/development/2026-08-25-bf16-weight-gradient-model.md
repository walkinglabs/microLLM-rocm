# 2026-08-25 — BF16 gate/up gradient 穿过短模型门

算子快不代表训练快。本轮用同一二进制只切换一个开关，交替进程顺序，结果是 Qwen
1.0213×、DeepSeek 1.0638×，峰值显存不变。

![BF16 weight-gradient model gate](../optimization-log/assets/bf16-weight-gradient-model.svg)

诊断证明每个step只改变 gate/up：两模型分别命中48/56次，QKV、down和O projection仍是
FP32 weight gradient。首个measured loss发生在一次warm-up更新之后，因此使用相对误差门；
两种loss门都低于0.5%。

当前仍不设默认。两步轨迹太短，而且候选增加192/224次逻辑分配。下一节点要保存逐步loss，
并比较完整gate/up参数摘要；这比再跑一组单点tokens/s更能发现慢性数值漂移。

后续结果：20-step门仅1/5通过，模型路由已删除。本文保留为“短门为什么不等于默认”的记录。
