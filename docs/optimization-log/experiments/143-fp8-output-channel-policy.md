# Experiment 143：DeepSeek变准，Qwen变差，速度都慢13%

## 为什么做

旧策略给整张权重表一把尺子；新策略给每个输出列一把尺子。理论上，较小的列不再被大列挤掉
刻度。算子仍用原生FP8 GEMM，随后在GPU上把每列scale补回。

## 结果

| 模型/上下文 | Max变化 | RMS变化 | T512 TPS变化 | scale显存 |
|---|---:|---:|---:|---:|
| Qwen T8 | +10.20% | **+28.77%** | 不作代际归因 | 1.16 MiB |
| Qwen T512 | +11.42% | **+27.79%** | **-13.09%** | 1.16 MiB |
| DeepSeek T8 | -30.13% | **-59.01%** | 不作代际归因 | 3.04 MiB |
| DeepSeek T512 | -22.59% | **-33.49%** | **-12.86%** | 3.04 MiB |

![Output-channel model policy](../assets/fp8-output-channel-policy.svg)

DeepSeek的数值改善很大，但四个完整精度门仍0/4通过；Qwen两种误差都变差。T512两模型速度
门也0/2通过。top token全部相同不改变这个结论。

## 调用和内存能解释什么

Qwen每个worker有672次post-scale，正好是168个Linear×4次forward。DeepSeek T512为
197×4=788。新增常驻字节约1.2/3.2 MB，与scale向量一致；性能损失主要不是显存容量，而是
每个native Linear多一次Kernel launch。DeepSeek T8有112次旧BF16 fallback，因此只执行676次
post-scale。

## 决定

拒绝设为跨模型默认；保留已验证的算子和显式opt-in模型策略。下一步先真实探测权重侧
hipBLASLt outer-vector：如果MI300支持，就能去掉post-scale launch；如果仍返回不支持，则关闭
这条直接库路径，再研究DeepSeek定向的较小Linear范围或融合补偿。任何后续候选仍需完整logits，
不能因为DeepSeek比例改善或top token相同就宣布可用。
