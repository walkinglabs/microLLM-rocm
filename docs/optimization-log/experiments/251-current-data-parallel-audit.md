# Experiment 251 — 当前双卡训练离 production reducer 多远

Status: `current baseline measured; first production contract selected`

当前RCCL配置14/14通过。20-step双卡、4MiB bucket限制下，loss从2.75降到0.55，rank参数
最大差始终为0。

| Stage | Median | Share |
|---|---:|---:|
| forward + backward | 1.565 ms | 68.34% |
| communication | 0.350 ms | 15.28% |
| optimizer | 0.070 ms | 3.06% |
| 未单列的参数一致性审计 | 0.305 ms | 13.32% |
| total | 2.290 ms | 100% |

![Current data parallel audit](../assets/current-data-parallel-audit.svg)

当前tiny模型只有一个bucket，所以不能证明真实overlap。源码仍在完整backward后同步每卡，再
pack/all-reduce/unpack；每步optimizer后还把两卡全部参数带回host比较，这段没有单独计时。

第一个production合同不是立刻写readiness state machine，而是先把参数审计从hot path时间中
分离：默认仍每步检查以保持兼容，增加显式interval和verification_ms。随后才能公平测bucket
与overlap。

四卡仍受当前容器64MiB `/dev/shm`限制，不能把四张可见GPU写成四卡可用。

证据：[`current data parallel`](../../../benchmarks/results/2026-08-25-current-data-parallel/)

