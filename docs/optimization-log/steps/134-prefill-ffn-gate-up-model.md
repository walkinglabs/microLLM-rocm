# Step 134 — Batch-selective prefill FFN gate/up model gate

Status: planned

Experiment 317的唯一exact index 296100在B1/B2/B8分别为1.040/0.951/0.995×，但B4只有0.941×。
下一节点在测量前固定策略：

- B1/B2/B8：cached-prefill的gate和up两个Linear使用296100；
- B4：保持upstream default；
- Attention scopes保持当前真实upstream，不叠加已拒绝的exact诊断stack；
- B1/2/4/8、两个fresh precision进程和反向排序performance进程；
- 完整BF16 cache、151,936 logits、prefill、peak、allocation和registry计数。

必须相对真实upstream同时改善全局Max/RMS至少10%，每个batch prefill≥0.95×。失败则删除模型路由并
关闭vendor FFN solution线；成功也只保留当前gfx942/backend的显式策略。
