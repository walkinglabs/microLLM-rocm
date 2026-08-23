# Experiment 118 data

固定 short-heavy/long-heavy 请求，比较统一 B8 与三种小桶:大桶 slot 配方：`2:6 / 4:4 / 6:2`。

```text
models       Qwen2.5-0.5B, DeepSeek-R1-Distill-Qwen-1.5B
processes    2 models × 2 traffic × 4 policies × 3 = 48
warmup       1 complete workload
measurement  3 workloads per process
GPU          physical GPU2, idle-gated
```

- `raw.jsonl`：48 条实际 route、吞吐、延迟、Cache 与 GPU boundary；
- `summary.json`：4 组 slot-ratio sweep；
- `gpu2-preflight.jsonl`：启动前连续三次 0/0；
- `gates.json`：测试、环境、数值与决策合同。
