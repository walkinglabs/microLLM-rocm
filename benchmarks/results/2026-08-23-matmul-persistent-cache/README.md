# Persistent matmul tuning cache evidence

- `cpu-roundtrip.log`：确定性JSONL round-trip、原子覆盖、stale architecture、schema和duplicate回滚；
- `hip-version-filter.log`：真实MI300环境恢复，以及runtime version变化后的stale过滤；
- `verification.json`：机器可读合同摘要。

这个目录证明save/load与环境失效，不证明候选本身更快。候选产生和correctness-before-timing仍由
下一独立节点负责。
