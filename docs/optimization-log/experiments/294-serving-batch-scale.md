# Experiment 294：不改数学，增大Batch能填满GPU吗

Status: measured; scheduler default withheld

## 固定合同

Qwen/DeepSeek、T2048、N64、BF16 KV、exact capacity、B1/B2/B4/B8。microLLM完全不传Attention
开关，24个进程必须报告`auto-enabled`；PyTorch为full BF16。每框架三进程、顺序交错。

![Serving batch scale](../../../benchmarks/results/2026-08-25-serving-batch-scale/batch-scale.svg)

| 模型 | B | micro tok/s | 扩展/效率 | PyTorch tok/s | micro/torch | token |
|---|---:|---:|---:|---:|---:|---|
| Qwen | 1 | 133.70 | 1.000x/100% | 96.07 | 1.392x | 相同 |
| Qwen | 2 | 266.11 | 1.990x/99.5% | 185.96 | 1.431x | 相同 |
| Qwen | 4 | 496.31 | 3.712x/92.8% | 394.61 | 1.258x | 相同 |
| Qwen | 8 | 880.42 | 6.585x/82.3% | 727.37 | 1.210x | 相同 |
| DeepSeek | 1 | 91.90 | 1.000x/100% | 85.43 | 1.076x | 第2索引分叉 |
| DeepSeek | 2 | 178.65 | 1.944x/97.2% | 163.19 | 1.095x | 相同 |
| DeepSeek | 4 | 327.03 | 3.558x/89.0% | 318.16 | 1.028x | 相同 |
| DeepSeek | 8 | 577.38 | 6.282x/78.5% | 672.16 | 0.859x | 第2索引分叉 |

batch确实是有效并行轴，但不是线性免费收益。Qwen B8仍保留82.3%效率；DeepSeek B8降到78.5%，
并被PyTorch反超。microLLM每请求peak随batch下降，KV按batch线性且两框架公式完全相同。

## 环境失败没有删除

第一轮PyTorch因容器AMDSMI报告0设备而失败。正式轮显式使用runner已有的
`amdsmi_zero_fallback_to_hip_runtime`；24条PyTorch raw都记录该字段并执行可见HIP设备。

## 为什么不改scheduler默认

DeepSeek B1/B8从token index 2跨框架分叉，B2/B4却相同。两框架精度政策不同，所以当前证据不能
直接判断谁错；更不能挑B2/B4宣布所有batch正确。Step 112先比较microLLM自身B1与各batch每一行的
完整logits，再决定是内部batch路径错误、argmax边界，还是仅跨精度政策差异。

证据：[`serving batch matrix`](../../../benchmarks/results/2026-08-25-serving-batch-scale/)
