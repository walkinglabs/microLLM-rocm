# Experiment 097 — 不再让空slot跑完整模型

Experiment 096的continuous状态机正确，但每次仍把空slot作为dummy row送进模型，再清空整行Cache。
本节点只改变这一件事：gather真实survivor rows，计算后scatter logits；请求状态和Cache容量不变。

## 实现合同

`forward_cached_active_rows(tokens[A,1], cache[B], rows[A])`要求rows严格递增、不重复且在范围内。
每个active row共享原Storage、使用自己的position，只推进自己。inactive row的完整capacity、position和
地址都不能变化。全部slot活跃且position相同时仍走原batch fast path。

```text
旧：survivor B + dummy → 两行模型 → dummy整行reset
新：rows=[B]           → 一行模型 → 只scatter B logits
```

![Active row compaction](../assets/active-row-compaction.svg)

## 正确性

- FP32/BF16 active logits逐row等于独立B1；
- CPU/HIP对齐，HIP active forward区间0次payload D2H；
- inactive row完整capacity逐项不变；
- shared Storage地址稳定；
- 空、重复、逆序、越界row列表明确失败；
- continuous A/B/C结果、随机、stop和cancel合同继续通过；
- 旧dummy rows全部变成`inactive_rows_skipped`，logical rows不变。

## Release全矩阵

| requests/slots | Exp096 tok/s | candidate tok/s | candidate/old | continuous/reference |
|---:|---:|---:|---:|---:|
| 2/2 | 1912.10 | 2167.54 | 1.134× | 0.935× |
| 4/2 | 1879.86 | 2296.98 | 1.222× | 0.960× |
| 8/2 | 1767.71 | 2307.63 | 1.305× | 0.959× |
| 4/4 | 1683.76 | 2268.93 | 1.348× | 0.962× |
| 8/4 | 1911.95 | 2369.27 | 1.239× | 0.985× |

五个divergent shape全部加速，dummy降为0，skipped分别为旧dummy的1/3/9/5/9。Cache allocated、
active peak、slot utilization和输出完全不变。

## 交替A/B反驳

uniform控制路径没执行candidate却波动5%–7%，因此不能只用前后两个进程。冻结Experiment 096
baseline binary后，对两个重点shape运行三对交替进程：

| shape | baseline median | candidate median | candidate/baseline | reference drift |
|---|---:|---:|---:|---:|
| R4/S4 | 1813.76 | 2343.39 | 1.292× | -0.10% |
| R8/S2 | 1868.04 | 2290.68 | 1.226× | -0.72% |

每shape六次checksum唯一且输出全通过。reference几乎不变，支持“收益来自跳过dummy”而不是机器变快。

## 仍然没有解决什么

continuous仍比串行reference慢1.5%–6.5%，更远低于static batch。active compaction当前仍逐row执行
B1，并做logits gather/scatter；KV仍按固定最大capacity预留。下一步需要让多个不同position的真实
row在RoPE、store和Attention中直接并行。

原始数据见 [`097-data`](097-data/)。
