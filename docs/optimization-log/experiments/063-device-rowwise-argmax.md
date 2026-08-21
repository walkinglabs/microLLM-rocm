# Experiment 063 — device row-wise argmax

## 问题

Experiment 060的uncached batch reference每个新token把`[B,V]` FP32 logits搬回CPU，再逐行
argmax。B越大，micro/PyTorch比值越低。目标不是隐藏uncached路径，而是让reference也遵守
“只传必要token”的系统合同。

## 实现与正确性

新增`argmax_last_dim`：任意leading shape，最后一维每行一个256-thread block，输出保留
leading shape的Int32。tie取最小index；某行出现NaN/Inf则该行输出-1。CPU/HIP测试覆盖
shape、tie、非有限值和“Kernel后零D2H，读取时仅B×4字节”。

CLI默认`--batch-argmax-mode device`，显式`host`只做控制。

## 同卡shape矩阵

| 模型 | batch | host | device | 自身加速 | device/PyTorch |
|---|---:|---:|---:|---:|---:|
| Qwen | 1 / 2 / 4 / 8 | 49.8 / 79.3 / 99.3 / 116.8 | 59.1 / 116.2 / 175.7 / 251.3 | 1.19× / 1.47× / 1.77× / 2.15× | 0.62× / 0.64× / 0.45× / 0.32× |
| DeepSeek | 1 / 2 / 4 / 8 | 38.9 / 59.3 / 76.9 / 92.7 | 43.9 / 78.6 / 114.3 / 155.8 | 1.13× / 1.32× / 1.49× / 1.68× | 0.51× / 0.47× / 0.35× / 0.26× |

![Device row-wise argmax](../assets/device-rowwise-argmax.svg)

八点peak完全不变，跨框架token全部一致。B8效率提高但仍只有Qwen53.5%、DeepSeek44.5%。

## Transfer与profile

同一Qwen B8直接控制：

```text
D2H calls       8 → 8
D2H bytes       38,895,616 → 256    151,936× fewer bytes
throughput      115.2 → 252.0       2.19×
```

rocprof下device新增12次argmax Kernel，共20.4ms；总Kernel反而`403.9→424.2ms`，但端到端
`121.1→249.4 tok/s`。这证明收益来自删除大D2H与CPU同步，不是Kernel时间变少。

## 决定

保留。它是batch-aware KV cache的必要采样原语。下一节点扩展Cache/Attention batch维；
否则cached B2/B4/B8仍是unsupported。
