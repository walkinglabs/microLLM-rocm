# Experiment 247 — 两步通过，二十步为什么仍然拒绝

Status: `model route rejected and removed`

短模型门曾达到1.0213×/1.0638×。本轮扩展到20 step、三组交替顺序进程，并在run 1后
完整比较全部gate/up FP32 master参数。

| Model | Throughput | Loss max relative | Parameter Max | Parameter RMS |
|---|---:|---:|---:|---:|
| Qwen | 1.0006× | 15.705× | 1.407e-4 | 1.218e-5 |
| DeepSeek | 1.0528× | 7.70e-4 | 6.235e-5 | 8.299e-7 |

![BF16 weight-gradient trajectory discard](../assets/bf16-weight-gradient-trajectory-discard.svg)

Qwen loss接近零时会放大相对比例，但最大绝对差也达到1.242e-3。更直接的拒绝证据是：
Qwen没有通过1.01吞吐门，两模型Parameter Max都超过5e-5，Qwen Parameter RMS也超过1e-6。
五个聚合门只有峰值显存通过。

完整比较覆盖Qwen 209,190,912个值和DeepSeek 770,703,360个值。临时多GB快照在比较后
删除，240个逐步loss和全部汇总保留。

## 决定

- 删除Autograd/CLI gate/up模型路由；
- 删除只服务这个候选的两个runner与测试；
- 保留CPU/HIP/PyTorch对齐的独立 `bf16_weight_gradient` API；
- 保留六shape benchmark、逐步loss输出与通用完整safetensors比较工具；
- 后续训练优化必须解决分配/workspace或选择另一种架构，不能恢复这个默认候选。

证据：[`20-step trajectory`](../../../benchmarks/results/2026-08-25-bf16-weight-gradient-trajectory/)

