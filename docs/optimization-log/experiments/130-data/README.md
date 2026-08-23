# Experiment 130 data

官方Qwen/DeepSeek FP32 T8 Linear输入的token-row amax分布。

- `raw.jsonl`：208个Tensor，每个保留8个row amax和完整派生指标；
- `workers.jsonl`/`summary.json`：两个worker与8个boundary aggregate；
- `command.txt`/`gpu2-preflight.jsonl`：复现命令和三次0/0设备门；
- `trace-manifest.json`：完整filtered trace的记录数与字节数。

完整值trace约95MB、压缩后仍约41MB，不放入源码历史；`raw.jsonl`保留本实验需要的全部逐row
amax，命令可重新生成full trace。同步trace不是性能证据。
