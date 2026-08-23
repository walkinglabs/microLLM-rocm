# Experiment 156 data

- `baseline-token-failure.log`：旧revision 20进程中的一次token失败；
- `before-numeric-failure.log`：修复前完整logits Max/RMS失败；
- `before-direct-determinism.log`：固定Q/K/V连续20次，20个非零最大差；
- `fixed-shape-runs/`：修复后20个独立完整shape进程，全部通过；
- `after-direct-and-shape.log`：最终直接Attention与shape门共同通过；
- `before/after-performance.jsonl`：T128/B8 tiny training各3个fresh process；
- `verification.json`与`gates.json`：机器可读结论。

性能只用于拒绝回退，不把correctness修复计入running-best。旧token失败来自detached
`59ffa5f`构建，证明问题早于exact registry节点。
