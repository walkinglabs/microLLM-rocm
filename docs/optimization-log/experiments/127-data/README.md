# Experiment 127 data

官方Qwen/DeepSeek的FP8 per-Tensor weight amax实验。

```text
contexts           8,512
policies           FP32, BF16 FFN+Attention, FP8 tensor-amax weight
runs               3 fresh processes, Latin rotation
activation scale   0.2 fixed
weight scale       each Linear max(abs(weight))/240
oracle             internal FP32 complete last-token vocabulary logits
```

- `raw.jsonl`/`summary.json`：正式v2的36条和12个aggregate；
- `gpu2-preflight.jsonl`：正式运行前三次0/0；
- `pilot-*`：计时字段修复后的3-worker Qwen T8 pilot；
- `rejected-*`：第一轮15条部分数据。该轮程序与精度门正常，但runner把准备时间错误记录为0，
  因此停止且无summary；
- `gates.json`：回归、正式矩阵和决策。

正式v2的FP8准备时间不是热路径时间：它包含一次性逐Linear D2H amax扫描。prepared forward的
HIP测试单独证明payload H2D/D2H为0。
