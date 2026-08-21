# Experiment 065 — BF16 KV Cache：显存减半不等于可以默认开启

## 问题

Experiment 064让cached batch可以工作，但microLLM Cache仍是FP32。相同shape下，它每个
数字4字节，而PyTorch参考Cache是BF16、每个数字2字节。问题是：只缩窄K/V Storage，能否
严格减半Cache，同时守住完整logits、token和速度？

## 单变量设计

```text
不变：FP32 Query、softmax、accumulator、output
改变：K/V Storage FP32 → BF16
```

`KVCache(layers, capacity, batch, dtype)`保存策略。full prefill只cast一次；step store在
HIP Kernel中写BF16；fused与T>4096 fallback都按dtype读取并转FP32累加。默认仍是FP32。

测试还发现旧T>4096 batch fallback把所有batch的probability都从batch0读取。B1和
sequence≤4096 fused测试无法发现它；新的B2 T4097反例先修复这个正确性缺口。

## 固定精度门

在看长context结果前固定：`max_abs≤0.25`、`RMSE≤0.05`、top-1相同、四个greedy token
完全相同、全部finite、Cache字节精确2×减少。

| 模型 | Release通过shape | 最大误差范围 | RMSE范围 | 16-token suffix |
|---|---:|---:|---:|---|
| Qwen | 6/6 | 0.103–0.158 | 0.020–0.036 | 6/6一致 |
| DeepSeek | 5/6 | 0.203–0.236 | 0.035–0.059 | 6/6一致 |

唯一Release失败是DeepSeek T512 B1：最大误差0.225仍过门，但RMSE 0.0586超过0.05。
普通构建的独立诊断曾在DeepSeek T2048把第四个token从`151643`改成`3555`；Release没有
复现。它被保留为build-sensitive反例，不能覆盖Release结论，也不能被删除。

## 显存、速度与profile

Release配对的12个shape中11个提高，Qwen T32 B8轻微`-0.43%`；最大收益是DeepSeek
T2048 B8 `+24.78%`。Cache字节每行精确减半，峰值比在`0.950–1.000`之间。

| 模型 | T | B | FP32→BF16 tok/s | 比值 | FP32→BF16 Cache MiB | peak比 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 32 | 1 | 320.83→322.25 | 1.004× | 1.125→0.563 | 0.9995 |
| Qwen | 32 | 8 | 2482.15→2471.41 | 0.996× | 9.0→4.5 | 0.9963 |
| Qwen | 512 | 1 | 221.95→229.79 | 1.035× | 12.375→6.188 | 0.9951 |
| Qwen | 512 | 8 | 1659.97→1775.16 | 1.069× | 99.0→49.5 | 0.9699 |
| Qwen | 2048 | 1 | 111.70→119.75 | 1.072× | 48.375→24.188 | 0.9862 |
| Qwen | 2048 | 8 | 858.73→892.53 | 1.039× | 387.0→193.5 | 0.9658 |
| DeepSeek | 32 | 1 | 190.25→192.50 | 1.012× | 2.625→1.313 | 0.9997 |
| DeepSeek | 32 | 8 | 1497.02→1498.01 | 1.001× | 21.0→10.5 | 0.9976 |
| DeepSeek | 512 | 1 | 123.96→136.47 | 1.101× | 28.875→14.438 | 0.9967 |
| DeepSeek | 512 | 8 | 833.34→995.30 | 1.194× | 231.0→115.5 | 0.9765 |
| DeepSeek | 2048 | 1 | 57.04→71.16 | 1.248× | 112.875→56.438 | 0.9884 |
| DeepSeek | 2048 | 8 | 424.74→530.01 | 1.248× | 903.0→451.5 | 0.9503 |

![BF16 KV Cache](../assets/bf16-kv-cache.svg)

同一formal中的PyTorch BF16参考：Qwen六点为`1.097×–3.347×`，DeepSeek T32/T512为
`1.446×–2.418×`，但DeepSeek T2048 B1/B8仍只有`0.840×/0.794×`。DeepSeek三条
跨框架token差异在冻结FP32 baseline已经存在，不能算作BF16 Cache新回归。

Qwen T2048 B8 profile中cached Attention `41.095→35.686ms`（`1.1516×`），全部Kernel
`322.321→316.195ms`（`1.0194×`）。BF16 prefix新增96次cast，因此下一节点可单独研究
一次Kernel完成cast+capacity-strided写入。非Release current对Release baseline的首轮结果
标为invalid，不进入速度结论。

## 被拒绝的向量化

第一版尝试用`bfloat162`同时读取两个K/V列。Qwen T512 B1/B8的BF16/FP32吞吐比降到
`0.470/0.372`，DeepSeek约`0.373/0.379`。原因不是字节更多，而是context阶段活跃线程
减半。只保留key-dot成对读取后，Qwen T512 B1仍只有`0.697`。代码恢复为标量读取，失败
raw完整保留。

## 决定

能力保留、默认不改：BF16 Cache是显式实验选项，FP32仍是安全默认。下一步不是放宽
DeepSeek RMSE阈值，而是定位T512 B1误差的layer/head；性能侧先反驳能否删除prefix的
96次cast与per-head copy。
