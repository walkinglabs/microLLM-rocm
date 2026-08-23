# Experiment 148：只改O projection，八项不退化并保留Deep收益

O-only只给每层Attention输出投影逐列scale。Q/K/V、FFN和LM head全部保持device Tensor-amax。
候选与control使用同一个`799da5e` binary，各36个worker。

| 模型/上下文 | Max变化 | RMS变化 | TPS变化 |
|---|---:|---:|---:|
| Qwen T8 | 0 | 0 | +1.27% |
| Qwen T512 | 0 | 0 | -3.74% |
| Deep T8 | -8.70% | -7.75% | -0.59% |
| Deep T512 | -16.26% | -14.32% | -1.50% |

![Attention output-only FP8 weights](../assets/fp8-attention-output-only.svg)

八个Max/RMS全部不变差，Deep四项严格改善，两项T512速度都过5%门，因此targeted keep=true。
额外常驻仅Qwen 85,920B、Deep 171,920B；每worker post为24/28层×4 forward，即96/112。

完整FP8门仍0/4，所以这个keep不能改写为“FP8推理已对齐”。它只证明：在当前device-Tensor
基线上，O projection逐列scale是一个数值更稳、代价受控的积木。

更宽Attention-only对Deep的改善与O-only完全相同，却让Qwen T512 RMS恶化8.91%，并多做三倍
post。证据支持Q/K/V造成Qwen长上下文回归、且没有提供额外Deep收益。下一源码节点删除更宽scope，
保留O-only。权重范围已经收敛，下一主线回到activation量化。
