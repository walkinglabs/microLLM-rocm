# 用 BF16 保存 AdamW 的“记忆”

这篇文档只解释一个问题：训练模型时，AdamW 为什么要多保存两份数字，以及怎样用
BF16 把这两份数字缩小一半。

## 1. 先把 AdamW 想成一个带记忆的调参员

模型里每个参数都有一个梯度。梯度告诉我们：“这次应该往哪边改。”如果只看当前一次
梯度，更新很容易忽左忽右。AdamW 因此为每个参数保存两个历史量：

- first moment：最近梯度的大致方向；
- second moment：最近梯度平方的大致大小。

可以把它们想成两本和模型一样大的笔记。若模型有 `N` 个参数，普通 FP32 AdamW 的
两本笔记需要：

```text
N × 4 字节 × 2 = N × 8 字节
```

BF16 每个数只占 2 字节，所以两本笔记变成：

```text
N × 2 字节 × 2 = N × 4 字节
```

这里没有把 FP32 主权重或梯度降成 BF16。改变的只有 AdamW 的两份历史状态。

## 2. 一步更新到底怎样算

每一步都按这个顺序进行：

```text
旧 BF16 moment
→ 转成 FP32 做公式
→ 新结果舍入成 BF16 并保存
→ 再用已经舍入的 moment 更新 FP32 主权重
→ 可选：同时刷新 BF16 推理/训练权重镜像
```

“先舍入，再更新参数”是公开契约。CPU reference、HIP Kernel 和 PyTorch 参考都使用
同一顺序。若先用未舍入值更新参数、最后才存 BF16，短测试也许看不出问题，训练很多步后
却会得到另一条轨迹。

## 3. 为什么不是默认开启

BF16 能表示的细节少于 FP32。它明显节省显存，也可能减少显存带宽，但不保证每个模型、
数据集和训练长度都保持相同质量。因此：

- 默认值仍是 `fp32`；
- `bf16` 必须由用户明确选择；
- checkpoint 会记录选择，恢复时不允许悄悄换策略；
- 当前 32 步 PyTorch 对齐和 100 步手算轨迹通过，但这不等于已完成长周期预训练证明。

## 4. 怎样使用

普通训练程序：

```bash
build/hip-release/apps/microllm_train \
  --data data.txt --model s --device hip \
  --adamw-moment-precision bf16
```

官方权重训练测量程序：

```bash
build/hip-release/apps/microllm_hf_train_step \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokens 1,2,3,4 \
  --device hip --linear-precision bf16 \
  --adamw-moment-precision bf16 \
  --warmup 1 --steps 2 --batch 1
```

输出中的关键字段是：

| 字段 | 含义 |
|---|---|
| `adamw_moment_precision` | 实际使用 `fp32` 还是 `bf16` |
| `adamw_moment_state_bytes` | 两份 moment 实际占用字节数 |
| `mean_optimizer_ms` | backward 完成以后，纯 optimizer 的平均时间 |
| `optimizer_timing_boundary` | 必须是 `post_backward_sync`，防止把 backward 尾部算进来 |
| `optimizer_*_bytes` | 检查 optimizer 是否偷偷搬运 Tensor payload |

## 5. checkpoint 怎样保持可复现

当前保存格式版本为 2：

- 保存 FP32 主权重；
- moment 统一导出成 CPU FP32，便于搬到另一台机器；
- 配置中保存 moment 精度策略；
- 恢复到 BF16 策略时，再明确舍入成 BF16 内部状态；
- 旧版本 1 被解释为 FP32 moment，并有真实二进制兼容测试。

因此，文件里的 canonical state 和 GPU 里节省显存的 working state 是两件事。

## 6. 目前 MI300X 证据

Qwen2.5-0.5B 与 DeepSeek Distill 1.5B 都使用 batch 1、context 512、一次热身、
两步测量和每策略五个新进程：

| 模型 | 训练吞吐 | optimizer | 峰值显存 | moment 状态 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.0226× | 1.0687× | 0.8329× | 3,952,262,144→1,976,131,072 B |
| DeepSeek Distill 1.5B | 1.0356× | 1.1964× | 0.8084× | 14,216,704,000→7,108,352,000 B |

Qwen 没有达到预先设置的 `1.10×` optimizer 进阶目标，所以结论是
`partial_keep`，不是“两个模型的 optimizer 都快 10% 以上”。

完整过程、失败的 multi-tensor 尝试和原始数据见
[Experiment 214](../optimization-log/experiments/214-bf16-adamw-moments-partial.md)。

## 7. 测试从哪里抓错

- CPU：两步逐项参考、100 步变化梯度轨迹、状态大小和错误 dtype；
- HIP：尾部 shape、完整参数/moment/mirror、零 payload 传输；
- multi-tensor 反例：FP32/BF16 完整状态和 metadata copy；
- PyTorch：第 2 步与第 32 步的参数、两份 moment 和 BF16 mirror；
- checkpoint：BF16 恢复后轨迹、策略不匹配拒绝、版本 1 兼容；
- 模型：Qwen/DeepSeek 五进程的 loss、吞吐和显存。

这些证据说明“实现按约定工作”。它们还不能证明任意规模、任意学习率的长训练质量。
