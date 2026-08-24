# Experiment 209 — 推理微融合局部饱和审计

Status: measured micro-fusion track closed

## 当前最佳路径

Exp205保留直接BF16 Q/K后，T512 phase profile中Qwen/DeepSeek总Kernel为4.754/8.928ms。
GEMM占57.4%/66.8%，softmax占10.0%/5.8%，cast占7.0%/5.8%，repeat占4.4%/3.3%。

![Post BF16 Q/K saturation](../assets/post-bf16-qk-saturation.svg)

即使某一类别被“完美删除”，Kernel时间理论上限也只有：

| Category | Qwen | DeepSeek |
|---|---:|---:|
| softmax | 1.111× | 1.062× |
| cast | 1.076× | 1.061× |
| repeat | 1.046× | 1.035× |

## 最近的反驳链

- 直接BF16 Q/K跨模型与shape稳定通过，keep；
- softmax 128线程只有4/6算子case通过，模型策略拒绝；
- BF16 V cast+repeat只有3/8通过，两条B2失败，模型策略拒绝；
- 既有可读fused Attention虽不物化T²，但只有library路径约0.36×，证明缺少MFMA tile和online
  数据复用时，省显存不能抵消矩阵计算效率损失。

## 饱和结论

继续扫64/128/256线程、再融合一个cast或再少一次launch，没有足够理论上限和跨模型证据。
当前推理微融合track关闭。下一次进入Attention必须是独立大型节点：MFMA/rocWMMA tile、online
max/sum、causal边界、GQA共享、完整logits与显存公式，而不是给可读Kernel改名。

原始汇总：[saturation evidence](../../../benchmarks/results/2026-08-24-post-bf16-qk-saturation/)。
