# Experiment 297：Batch漂移从第0个Block边界开始

Status: block 0 selected for detail trace

## 先看完整Block，不先猜某个算子

DeepSeek T2048 step0固定相同prompt，比较B1与B2第0行。FP32 Linear和BF16 FFN-only各跑两个
fresh process，记录embedding、28个block、final norm与完整151,936 logits。

![Cached block drift](../../../benchmarks/results/2026-08-25-deepseek-cached-block-drift/block-drift.svg)

| 边界 | FP32 Max/RMS | BF16 FFN Max/RMS | BF16/FP32 Max |
|---|---:|---:|---:|
| Embedding | 0 / 0 | 0 / 0 | — |
| Block 0 | 7.62e-6 / 1.91e-6 | 0.003909 / 0.000348 | 512.88x |
| Block 27 | 0.008301 / 0.001453 | 0.582840 / 0.054506 | 70.22x |
| Logits | 0.001354 / 0.000218 | 0.062985 / 0.025171 | 46.54x |

四个进程的层级误差完全重复。Embedding位级相同，说明输入和查表不是漂移来源。FP32通用Linear
在Block 0已经出现很小的batch-shape数值差；BF16 FFN-only在同一个边界把Max放大到512.88倍。
误差继续穿过28层增长，最终再被final norm压缩，所以不能只看最终logits反推中间峰值。

## 决定

这只定位“第一处可见边界”，没有证明Block 0内哪个算子是根因，也没有证明只改第一层就能解决
最终生成分叉。因此不改precision或scheduler默认。下一节点只打开Block 0的attention residual、FFN
norm、BF16 input、gate、up、activated、down和block output，寻找第一处真正的放大操作。

证据：[`cached block drift`](../../../benchmarks/results/2026-08-25-deepseek-cached-block-drift/)
