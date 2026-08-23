# Experiment 116 data

正式矩阵使用 physical GPU 2。启动前连续三次、每次间隔 10 秒确认 `use=0% / VRAM=0%`。

```text
suite        traffic-skew
models       Qwen2.5-0.5B, DeepSeek-R1-Distill-Qwen-1.5B
workloads    short-heavy, long-heavy, delayed-arrival
policies     uniform B8, two B4 buckets
warmup       1 complete workload
measurement  3 workloads per process
processes    3 per model/workload/policy
```

文件：

- `raw.jsonl`：36 条 idle-gated 正式记录；
- `summary.json`：6 组 policy comparison 和 focus-request P50/P95；
- `gpu2-preflight.jsonl`：正式运行前的三次空闲确认；
- `rejected-post-gate-selection.jsonl`：第一次选择空闲 GPU 后被外部任务中途抢占；
- `rejected-monitor-timeout.jsonl`：第二次等待 30 分钟仍无空闲设备的 180 次采样；
- `gates.json`：测试、环境和决策合同。

第一次 post gate 发现 61% VRAM 后 raw 保持 0 行；第二次没有满足连续空闲条件，因此没有启动
benchmark。这两组只解释为何等待和重跑，不参与正式性能表。
