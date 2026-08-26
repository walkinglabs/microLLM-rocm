# Experiment 303：第一处差异是大M的FP32 Q投影

Status: large-M FP32 QKV solution audit selected

## 前两个Batch Row，十个边界

固定DeepSeek T2048、FP32 Linear、BF16 KV，B1/2/4/8各两个fresh process。full-prefill trace只保存
前两个batch row，覆盖embedding、norm、Q/K/V projection、RoPE/value和BF16 cache。

![Prefill block-0 trace](../../../benchmarks/results/2026-08-26-deepseek-prefill-block0-trace/prefill-trace.svg)

Embedding和Attention Norm全部exact。第一处非零是FP32 Q projection。B8时：

| 边界 | Max | RMS |
|---|---:|---:|
| Q projection | 9.155e-5 | 2.914e-6 |
| K projection | 3.052e-5 | 1.481e-6 |
| V projection | 5.007e-6 | 2.632e-7 |
| BF16 cache key | 0.03125 | 8.653e-5 |
| BF16 cache value | 0.0009765625 | 1.497e-5 |

RoPE只继承projection差异。BF16存储让Key Max变成K RoPE的1024倍，让Value Max变成FP32 value的
195.05倍；这是不同FP32输入落入不同BF16舍入桶的放大，不是相同输入的cast错误。

B2内部K/V projection和cache仍exact，但Q row已经不同；B4/B8三路projection和cache内部行都不同。
两次process的全部指标一致。

## 决定

不改BF16 Cache。下一步把Q/K/V的M固定为2048/4096/8192/16384，分别对N1536与N256枚举FP32
hipBLASLt solution并检查相同输入row的完整输出。若存在跨M exact候选，再进完整prefill模型；否则
需要保序FP32 projection Kernel或接受容差语义。

证据：[`prefill block-0 trace`](../../../benchmarks/results/2026-08-26-deepseek-prefill-block0-trace/)
