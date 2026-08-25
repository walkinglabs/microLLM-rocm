# Experiment 274 — 看见4张GPU，为什么还不能说4卡可用

Status: `general interface kept; current four-rank environment unavailable`

同一干净revision运行tiny一步：world1/2走完整rank+CPU reference，world4走有限时group-init探测。

| World | Result | Rank diff | CPU diff | Group time |
|---:|---|---:|---:|---:|
| 1 | pass | 0 | 1.4e-8 | 5.227 s |
| 2 | pass | 0 | 6.0e-8 | 5.328 s |
| 4 | init fail | — | — | 2.756 s |

![Ranked world-size boundary](../assets/ranked-world-size-boundary.svg)

world4四个进程均返回1，stderr为`ncclCommInitRank: unhandled system error`；没有挂死或强制终止。
机器确实暴露4个MI300X VF，但容器`/dev/shm`只有67,108,864 bytes（64MiB），与历史失败一致。

代码现在可以构造N个rank、N份local batch和CPU global batch，并比较全部rank。接口泛化被保留；
当前环境4卡能力明确为未通过。不能把“代码接受world-size=4”写成“4卡训练跑通”。

下一节点开启RCCL debug/preflight，给出比`unhandled system error`更可操作的共享内存诊断；资源真正
变化后再重跑同一完整门。

证据：[`ranked world size`](../../../benchmarks/results/2026-08-25-ranked-world-size/)
