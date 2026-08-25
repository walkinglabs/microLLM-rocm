# 2026-08-25 — next long-context profile audit

ranked reducer局部优化关闭后，全仓状态表中最大的明确性能失败是DeepSeek T2048：最新serving
检查约为PyTorch的0.868x，历史B1/B8为0.866x/0.671x。

旧Experiment 086的rocprof证据显示cached Attention约占decode wall 60%，B8还曾有allocator
相位问题。allocator问题后来已经修复；模型执行路径也经过多轮BTHD/BF16/Arena优化。因此下一
节点必须先建立当前revision的可解释门，不能直接复用旧比例。

审计发现底层已有三条实现：

- T≤4096默认`cached_attention_fused_kernel`；
- 可读的`cached_attention_scores_kernel`；
- 可读的`cached_attention_context_kernel`。

但是公共API只暴露最终context，没有逐位置score。仓库未完成清单也明确要求“per-position
dot/codegen gate”和“score-level oracle”。Step 104因此先公开只读score诊断、覆盖DeepSeek
H12/KV2/D128和T2048，再重跑当前profile。

这个节点只确定实验合同，不声称旧60%仍成立，也不选择新Kernel。
