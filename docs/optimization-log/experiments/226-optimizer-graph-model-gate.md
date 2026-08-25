# Experiment 226 — micro快36倍，为什么模型optimizer反而更慢

Status: `reject model route; close optimizer-only Graph track`

## 最后一个模型门

前四个实验依次解决了动态Storage、device step、稳定descriptor和Stream/allocator阶段交接。现在
只对snapshot安全的三个case真正launch：Qwen T8、Qwen T512、DeepSeek T8。DeepSeek T512每次
仍只执行preflight并保持零launch。

每个measured step固定为：

```text
device-wide quiescent handoff
→ default-Stream forward/backward
→ 逐项验证gradient snapshot
→ eager Hybrid AdamW 或两节点multi Graph
→ device synchronize
```

Graph preparation不计入steady step，但单独报告；handoff同步计入完整step。

## 21进程结果

| Model | Context | Optimizer Graph/eager | Full step Graph/eager | Setup | Optimizer H2D |
|---|---:|---:|---:|---:|---:|
| Qwen 0.5B config | 8 | 0.798× | 1.050× | 1.70ms | 2→0 |
| Qwen 0.5B config | 512 | 0.807× | 0.929× | 1.94ms | 2→0 |
| DeepSeek-Distill 1.5B config | 8 | 0.656× | 0.875× | 2.97ms | 2→0 |

![Model optimizer Graph gate](../assets/optimizer-graph-model-gate.svg)

三次新进程的每个eager/Graph配对都满足：

- 两步loss数组完全相同；
- 观察参数完全相同；
- optimizer step为2；
- snapshot每步匹配；
- Graph节点数为2；
- Graph optimizer H2D/D2H/D2D为0；
- DeepSeek T512三次preflight仍是零launch。

所以回退不是数值错误、descriptor未命中或metadata仍在复制。

## 为什么micro与模型相反

micro的36×来自256个1K Tensor合成一个grid。真实模型含少量非常大的projection/embedding Tensor；
现有Hybrid eager路径已把小Tensor合并，大Tensor走高效单Tensor Kernel。把全部block塞进一个通用
descriptor Kernel会损失大Tensor执行效率，省下的提交/H2D不足以抵消。

Qwen T8完整step偶然达到1.050×，但它的目标optimizer本身只有0.798×，且Qwen T512和DeepSeek
完整step均回退。不能用一个短case的phase噪声覆盖两个反例。

## 决定

- 不增加模型/CLI Graph开关，不设默认；
- 保留底层Graph step、multi descriptor、snapshot safety与quiescent API作为研究原语；
- 关闭optimizer-only Graph优化track，不再扫描block size、Tensor阈值或更多context；
- 若未来完整forward/backward Graph需要这些原语，可复用但必须重新做整步门；
- 当前训练下一方向回到更大架构问题：online/tiled Attention或真正的graph-wide liveness，而不是
  单独包装optimizer；
- DeepSeek T512的7.108GB地址反例永久保留。

原始证据位于
[`benchmarks/results/2026-08-24-optimizer-graph-model-gate/`](../../../benchmarks/results/2026-08-24-optimizer-graph-model-gate/)。
