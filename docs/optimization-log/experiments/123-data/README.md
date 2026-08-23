# Experiment 123 data

这是官方Qwen/DeepSeek的FP8全局scale筛选数据。候选值在运行前固定，没有根据最终token反向挑选。

```text
context           8
activation scales 0.00625, 0.0125, 0.025, 0.05
weight scales     0.00125, 0.0025, 0.005, 0.01
workers           每个模型1个FP32 reference + 16个FP8候选
comparison        最后位置的完整151936维FP32 logits
warmup/steps      1/3
```

- `command.txt`：原始命令；
- `gpu2-preflight.jsonl`：运行前连续三次物理GPU状态；
- `raw.jsonl`：34条fresh-process原始记录；
- `summary.json`：固定选择规则、逐模型最佳候选和全部记录；
- `gates.json`：回归、执行和结论门；
- `stderr.log`：空文件，证明正式runner没有写错误输出。

本轮只筛选短上下文。Qwen最优点落在activation网格上边界，因此不能把“当前32个失败”外推成
“所有全局scale都失败”。
