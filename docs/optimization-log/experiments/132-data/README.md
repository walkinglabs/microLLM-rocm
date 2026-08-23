# Experiment 132 data

官方host/device weight tensor-amax冷启动对照。正式36条使用device模式、固定activation0.2。

- `raw.jsonl`/`summary.json`：正式矩阵；
- `pilot-*`：fresh binary Qwen T8；
- `rejected-stale-*`：旧二进制0-row合同失败；
- `rejected-build.log`：fresh build暴露C++字符串拼接错误；
- `fresh-configure.log`/`fresh-build.log`：修复后独立Release Ninja 34/34 build证据。

没有生成或记录文件摘要值。host基线来自Experiment127同机器、同模型、同context和相同
activation策略；结果不bit-exact，因此只比较完整logits误差、top token和聚合性能。
