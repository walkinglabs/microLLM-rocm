# Experiment 115 data

正式数据来自自动 idle gate 保护的 18 个 Release fresh process：

```text
--suite bucket-sweep
--warmup 1 --steps 3 --runs 3
--physical-gpu-index 3
--max-idle-vram-percent 5
```

每条 `raw.jsonl` 都保存 `pre_run_gpu_state` 和 `post_run_gpu_state`。18 条 pre VRAM 均为 0%，
post 为 0%–2%。

文件：

- `raw.jsonl`：正式 1/2/4 桶矩阵；
- `summary.json`：中位数、token difference 与 Pareto 行；
- `rejected-partial-raw.jsonl`：上一轮程序成功但环境受污染的数据；
- `rejected-gpu3-telemetry.log`：上一轮从 17:23:53Z 开始出现 60%–96% 外部 VRAM 占用；
- `gates.json`：测试、正式矩阵和拒绝窗口合同。

拒绝文件只证明为什么必须重跑，不参与性能表格。上一轮 Qwen 虽在干净窗口完成，也没有与新
DeepSeek 数据拼接；正式矩阵完整重跑了两个模型。
