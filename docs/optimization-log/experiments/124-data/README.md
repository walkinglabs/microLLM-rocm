# Experiment 124 data

Experiment 123的Qwen最佳点落在activation scale上边界0.05。本轮只把该维度扩到0.1和0.2，
weight scale保持原来的四个值；两个官方模型各有1个FP32 reference和8个FP8候选。

```text
context           8
activation scales 0.1, 0.2
weight scales     0.00125, 0.0025, 0.005, 0.01
workers           18 fresh processes
comparison        完整151936维last-token logits
```

- `command.txt`：实际命令；
- `gpu2-preflight.jsonl`：三次0/0物理GPU预检；
- `raw.jsonl`：18条原始记录；
- `summary.json`：16个候选、固定选择规则和逐模型最佳点；
- `gates.json`：实验门和下一步；
- `stderr.log`：空错误输出。
