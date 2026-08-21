# Experiment 090 — allocator稳定以后，同一个D2H候选还会失败吗？

Experiment 086曾把token history做对：D2H 24→3，但旧allocator因为少了21次小分配而在B8触发
13,863次backend allocation，吞吐只有0.861×。Experiment 087移除了这个相位混杂变量，现在用
完全相同的目标重试。

## 实现边界

- `argmax_out_`与`argmax_last_dim_out_`写入caller-owned连续Int32 Tensor；
- HIP greedy且没有stop token时，预分配`history[N,B]`；
- 每一步argmax直接写对应view，下一次forward继续使用这个device view；
- generation结束后一次性把history带回host；
- sampling、temperature/top-k随机路径和stop-token提前结束路径完全不变；
- benchmark steady decode使用同一机制。

## Transfer合同

N8、measured steps 3：

| Batch | Exp087 D2H calls | Exp090 D2H calls | bytes before/after |
|---:|---:|---:|---:|
| 1 | 24 | 3 | 96 / 96 |
| 8 | 24 | 3 | 768 / 768 |

调用减少87.5%，数据量完全相同。它减少的是host/device同步边界，不是假装少生成token。

## DeepSeek T2048三对交替

| Batch | baseline tok/s | candidate tok/s | 速度比 | backend alloc |
|---:|---:|---:|---:|---:|
| 1 | 67.29 | 67.41 | 1.002× | 94 / 94 |
| 8 | 512.06 | 513.71 | 1.003× | 94 / 94 |

Experiment 086的allocator爆炸没有重现。peak、reserved、KV和三对token全部相同；candidate只比
baseline少21个logical allocation/reuse，正好对应24→3的history copy边界。

## Qwen T512 B8反驳实验

宽矩阵单进程再次出现噪声回退，所以仍跑三对交替：

```text
baseline:  1714.68 / 1712.83 / 1696.16 tok/s
candidate: 1710.62 / 1707.27 / 1705.79 tok/s
median ratio = 0.99675×
```

0.325%的差异在噪声门内。D2H稳定24→3，双方backend allocation固定86，peak/KV/token完全相同。

## 六shape survey与公共API门

Qwen/DeepSeek T8/T512/T2048、B1/B8共24个candidate/PyTorch进程全部成功。Qwen 6/6 token一致；
DeepSeek T8/T512一致，T2048两点保留既有分叉。所有microLLM记录只有3次D2H、81–94次backend
allocation和0次backend deallocation。

公共`generate()`/`generate_batch()`测试额外证明：

- B1 N4 D2H calls从4降到1；
- B3 N4只做1次D2H、48 bytes；
- 不同batch row仍逐行等于CPU；
- stop-token路径仍使用原逐步host决策并保持不同完成长度。

![Device token history after allocator stabilization](../assets/device-token-history.svg)

## 决定

保留candidate。它是服务边界优化，不是设备Kernel加速：当前长context计算占主导，所以吞吐中性；
但同步次数确定下降、公共greedy API得到同样能力，且allocator、数值和内存门全部通过。

下一步回到cached Attention前，先增加逐position dot中间结果门，避免Experiment 088/089那种“小测试
通过、官方长规约失败”。

数据见[`090-data`](090-data/)。
