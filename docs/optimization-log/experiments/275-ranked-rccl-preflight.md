# Experiment 275 — 把`system error`还原成哪一项资源不足

Status: `diagnosis kept; four-rank execution still unavailable`

依据AMD官方[RCCL环境变量](https://rocm.docs.amd.com/projects/rccl/en/7.13.0-preview/api-reference/env-variables.html)
和[troubleshooting流程](https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/troubleshooting-rccl.html)，
launcher为每个rank写独立debug日志并提取诊断。

| Evidence | Value |
|---|---:|
| visible GPUs | 4 |
| world size | 4 |
| `/dev/shm` total | 67,108,864 bytes |
| `/dev/shm` free before launch | 43,724,800 bytes |
| logs with no-space error | 4/4 |
| failed segment | 21,823,872 bytes |
| RCCL | 2.28.3-HEAD:3309c61 |
| raw debug logs | 507,069 bytes, deleted |

![Ranked RCCL preflight](../assets/ranked-rccl-preflight.svg)

四份日志都明确`No space left on device (28)`，诊断从不透明system error收窄为
`shared-memory-capacity-exhausted`。失败segment不是总需求，因此required total保持null/unknown。

同一preflight下world2完整训练/CPU门继续通过，说明诊断不会误拒绝可用配置。原始verbose日志
删除，只保留结构化summary。当前world4仍不声明成功，等待容器共享内存资源变化后重跑完整门。

证据：[`ranked RCCL preflight`](../../../benchmarks/results/2026-08-25-ranked-rccl-preflight/)
