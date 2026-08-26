# 为什么核心完全一致，模型还是不一致

## 简单解释

我们把每一层Attention中间的“打分、比例、取内容”三步固定成跨batch完全相同。Block 0确实成功。

但是一层Transformer后面还有输出投影、残差、FFN，下一层又读取已经变化的输入。因此修好第一段
不保证整条流水线都相同。实测完整logits的最大差异反而大了约24.6%。

速度也没有过门：B1为0.94954×，比0.95要求只差一点，但数值门已经明确失败，所以不能用四舍五入
把它接受。

![Complete model rejection](../../benchmarks/results/2026-08-26-fp32-prefill-attention-model-gate/model-gate.svg)

下一步尝试每个batch自己的候选，不要求所有descriptor共用同一个index；仍然先完整测量再决定。
