# Experiment 129 data

官方Qwen/DeepSeek的weight tensor-amax + device activation Tensor amax矩阵。

```text
contexts          8,512
runs              3 fresh processes, Latin rotation
weight scale      per-Linear host amax preparation
activation scale  per-Linear-input device amax, minimum 0.0001
oracle            FP32 complete last-token vocabulary logits
```

- `raw.jsonl`/`summary.json`：正式36条、12个aggregate和4个FP8精度失败；
- `pilot-*`：正式运行前3-worker Qwen T8合同；
- `gpu2-preflight.jsonl`：三次0/0；
- `gates.json`：完整回归、数值/性能双门和结论。

TPS不包含一次性weight准备，但包含每个forward中的device amax和动态量化。当前amax Kernel只有
一个256-thread block，长context结果是必须保留的性能失败。
