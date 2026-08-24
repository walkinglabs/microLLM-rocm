# Experiment 213 — 训练微融合到达局部饱和

Status: `measured; local track closed`

## 为什么要做差分

完整程序profile包含权重加载、转置、BF16 mirror准备和第一次library setup。例如DeepSeek的197次
`cast_transpose_2d`全部属于加载，不属于每步训练。如果直接用整份profile占比，会选错热点。

本实验对每个模型运行两个相同revision的profile：

```text
load + 1 training step
load + 3 training steps
-------------------------
差值 = 2个完整训练step
```

所有Kernel类别的调用差都非负，说明两份trace可以做结构差分。

## 每步真正花在哪里

| Category | Qwen two-step | Share | DeepSeek two-step | Share |
|---|---:|---:|---:|---:|
| GEMM | 37.479 ms | 55.87% | 96.024 ms | 62.25% |
| AdamW | 11.301 ms | 16.85% | 33.192 ms | 21.52% |
| bias gradient | 2.933 ms | 4.37% | 3.371 ms | 2.19% |
| cross entropy | 2.769 ms | 4.13% | 2.543 ms | 1.65% |
| FP32→BF16 cast | 2.009 ms | 3.00% | 3.595 ms | 2.33% |
| softmax | 1.821 ms | 2.71% | 1.940 ms | 1.26% |
| add | 1.725 ms | 2.57% | 2.222 ms | 1.44% |
| repeat | 1.359 ms | 2.03% | 2.056 ms | 1.33% |

GEMM和AdamW合计占`72.71%/83.77%`。这不是说其余代码不重要，而是说明单独删除一个小Kernel
类别的上限已经很低。

## 完美删除也只是上限

| Perfectly removed category | Qwen upper bound | DeepSeek upper bound |
|---|---:|---:|
| bias gradient | 1.0457× | 1.0223× |
| cast | 1.0309× | 1.0239× |
| add | 1.0264× | 1.0146× |

现实候选还要执行同样的数据读写或改变调度，所以实际收益远低于“完美删除”。最近三条证据正好
给出反驳：

- residual add + RMSNorm：`0.9785×/0.9980×`；
- multi-tensor AdamW：`1.0573×/1.0094×`，DeepSeek未过门；
- shared BF16 activation：三条策略均至少一个模型失败。

![Post training micro saturation](../assets/post-training-micro-saturation.svg)

## 下一阶段合同

训练微融合track关闭。下一项工作必须属于下面至少一类：

1. **GEMM级**：新的精确shape算法、有效的grouped backward或改变中间布局；
2. **带宽级**：AdamW减少parameter/gradient/moment实际流量，而不是只减少launch；
3. **图级**：完整liveness plan、稳定地址和异构HIP Graph，不再靠局部`use_count`猜测。

原始汇总在
[`benchmarks/results/2026-08-24-post-training-micro-saturation/`](../../../benchmarks/results/2026-08-24-post-training-micro-saturation/)。
