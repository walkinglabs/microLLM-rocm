# Experiment 137 data

Qwen/DeepSeek T8 FP32与shared dynamic FP8逐block完整值对照。

- `raw.jsonl`/`summary.json`：runner原始结果；
- `analysis.json`：完整性、最大跳变和关键阶段；
- `per-stage.tsv`：56个阶段全部max/RMS/relative-L2；
- `trace-manifest.json`：4份未入Git的大型trace尺寸与完整性统计。

完整trace共约53MB，可由`command.txt`重建；仓库保留全部派生数值，诊断同步不是性能证据。
