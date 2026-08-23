# FP8 all-layer activation range evidence

新增显式`--trace-all-layer-details true`，默认trace不变。T8 FP32基线对每层四个Linear输入计算
完整Tensor统计，只序列化一个样例值。

首次Qwen trace因FP32 FFN缺少activated观测点按72/96合同停止；修复和96行pilot后，正式得到
Qwen96 + DeepSeek112行。固定scale0.2的±48范围有16个潜在饱和边界，全部位于FFN；最大
Qwen/DeepSeek activated分别超过范围35.9×/64.2×。Attention context的P50 amax仅2.59/2.97，
同一scale又明显过粗。

下一实现从device per-input-Tensor amax开始；当前证据还不足以直接要求per-row/per-token。

详见[Experiment 128](../optimization-log/experiments/128-fp8-activation-range.md)。
