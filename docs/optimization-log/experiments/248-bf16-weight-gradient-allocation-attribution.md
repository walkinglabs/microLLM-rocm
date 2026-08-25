# Experiment 248 — 多出来的分配到底来自哪里

Status: `attributed; workspace speedup not yet proven`

不恢复Experiment 247已经删除的模型路由，只读取保留的20-step记录并与算子Storage合同核对。

| Model | Routes | Extra allocations | Per route | Bytes/route | Cast bytes |
|---|---:|---:|---:|---:|---:|
| Qwen | 960 | 1,920 | 2 | 5,898,240 | 5,898,240 |
| DeepSeek | 1,120 | 2,240 | 2 | 10,747,904 | 10,747,904 |

![BF16 weight-gradient allocation attribution](../assets/bf16-weight-gradient-allocation-attribution.svg)

两次额外逻辑分配恰好是BF16 input cast+transpose和BF16 dY cast。每次route的字节数也逐字节
相等。两模型backend allocation增量、peak增量、cached bytes增量均为0；所有额外逻辑分配
都由cache reuse吸收。

这完成了来源证明，但没有证明workspace值得实现。caller-owned workspace只能省掉逻辑对象与
cache lookup，不能删掉两个cast Kernel或GEMM。下一门必须直接比较allocating与preallocated
的wall/Event；不过门就不增加workspace API。

证据：[`allocation attribution`](../../../benchmarks/results/2026-08-25-bf16-weight-gradient-allocation-attribution/)

