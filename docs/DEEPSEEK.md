# DeepSeek 支持边界

## 已实测：DeepSeek-R1-Distill-Qwen-1.5B

这个模型把 DeepSeek-R1 的推理数据蒸馏进 Qwen2.5 的密集 Decoder 架构。它使用
Qwen2 权重结构，因此可以复用已经验证的 GQA、Q/K/V bias、split-half RoPE、
RMSNorm 和 SwiGLU。

MI300X 实测证据：

```text
参数                    1,777,088,000
严格加载 Tensor         339
完整 logits             151,936
最大绝对误差            6.409e-5
MSE                     1.169e-10
greedy token 对齐        8/8
当前模型显存             7.108 GB
引擎峰值显存             14.217 GB
```

固定问题 `What is 2+2?` 的 reasoning prompt 和 Transformers 使用相同 12 个
token。两边随后都生成：

```text
[40,1184,311,8253,279,2629,315,220]
I need to determine the sum of 
```

## 没有声称支持：旗舰 DeepSeek-R1/V3

旗舰模型不是“更大的 Qwen”，还需要：

- MLA 和压缩 KV cache；
- routed/shared MoE experts；
- FP8 scale 与专家路由；
- expert/tensor/data parallel；
- 多节点通信与故障处理。

因此 README 和报告必须始终带上 `Distill-Qwen-1.5B` 全名。
