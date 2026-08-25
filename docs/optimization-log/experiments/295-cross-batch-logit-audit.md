# Experiment 295：DeepSeek分叉是Batch行写错，还是数学路径变了

Status: batch-shape numerical drift confirmed

## 审计方法

只比较microLLM自身。固定T2048、BF16 KV、no-flag Auto，B1/B2/B4/B8在decode step0/1/2分别
导出完整`[B,151936]` logits，每格两个fresh process。

![Cross-batch logits](../../../benchmarks/results/2026-08-25-deepseek-cross-batch-logits/cross-batch.svg)

## 排除了什么

- 24/24进程两次运行位级确定；
- 每个batch内部所有相同行位级相同；
- 每行host完整logits argmax与device token相同；
- 24/24都实际为`auto-enabled`。

所以不是某一batch row被写坏，也不是device argmax选错。

## 从哪里开始不同

跨batch在step0已经不位级相同：B2/B4/B8相对B1 Max分别0.04968/0.06757/0.05165。step1最大RMS
达到0.04613；step2全局Max达到0.19780。

step2 top1恰好分成两组：B1/B8为151643，B2/B4为3555。这与Experiment 294的64-token分叉完全
一致，说明序列差异来自更早的完整logits数值，而不是后处理。

## 决定

不设置scheduler默认。当前证据支持“batch shape选择了不同数值路径”，但尚未定位是BF16 FFN、
BF16 Attention还是FP32部分。Step 113只测step0，比较四种精度策略；FP32若也漂移，查通用batch
GEMM/模型路径，若只有某个BF16 island漂移，再进入对应层trace。

证据：[`cross-batch audit`](../../../benchmarks/results/2026-08-25-deepseek-cross-batch-logits/)
