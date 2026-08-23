# Experiment 117 data

本节点比较三条路径：统一 B8、固定两个 B4、允许短请求进入兼容大桶的两个 B4。

```text
models       Qwen2.5-0.5B, DeepSeek-R1-Distill-Qwen-1.5B
traffic      short-heavy, long-heavy, delayed
policies     uniform, fixed, compatible-overflow
processes    2 × 3 × 3 × 3 = 54
measurement  warmup 1, measured workloads 3
GPU          physical GPU2, pre/post VRAM/use gates
```

- `raw.jsonl`：54 条正式记录和实际 bucket routes；
- `summary.json`：6 组 overflow 对 fixed/uniform 的 token、吞吐和 focus P95；
- `gpu2-preflight.jsonl`：正式 v2 前连续三次 0/0；
- `rejected-routing-raw.jsonl`：第一次只完成 uniform/fixed 的 6 条记录；
- `rejected-routing-preflight.jsonl`：第一次环境同样干净，证明失败来自软件合同；
- `gates.json`：完整回归、正式矩阵和决策。

第一次 overflow record 因实际 route 与预期不同，在写 raw 前被拒绝。原因是 pending 在 bucket load
中被计算两次。修复、增加 4-slot 阈值测试并通过官方 route smoke 后，才从新目录完整重跑。
