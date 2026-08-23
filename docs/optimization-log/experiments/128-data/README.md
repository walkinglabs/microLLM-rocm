# Experiment 128 data

官方Qwen/DeepSeek FP32 T8全层Linear输入范围trace，以固定FP8 activation scale 0.2的
`±48`范围做反事实分析。

- `raw.jsonl`：Qwen96 + DeepSeek112 = 208个层/边界；
- `workers.jsonl`：两个正式worker和GPU前后状态；
- `qwen-trace.jsonl`/`deepseek-trace.jsonl`：完整Tensor统计，每条只序列化1个样例值；
- `summary.json`：8个model/boundary aggregate与16个潜在饱和边界；
- `pilot-*`：修复后Qwen 96行合同pilot；
- `rejected-*`：第一次Qwen trace有317条记录，但FP32 FFN没有记录activated，runner按72/96
  合同停止且未生成raw/summary；
- `gates.json`：测试、正式矩阵与下一设计门。

同步trace包含大量诊断D2H，只用于数值范围，不用于速度比较。
