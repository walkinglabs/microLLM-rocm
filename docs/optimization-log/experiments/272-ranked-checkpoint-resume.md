# Experiment 272 — 两个rank恢复以后，还是同一次训练吗

Status: `rank0 ownership and tiny resume kept`

固定tiny、两rank、per-parameter reducer。中断路径跑2步并由rank0写checkpoint，新进程恢复再跑
3步；控制路径从相同seed不中断跑5步。

| Gate | Result |
|---|---:|
| resumed vs uninterrupted parameters | max 0 |
| rank vs rank parameters | max 0 |
| complete checkpoint bytes | equal |
| each checkpoint size | 10,796 bytes |
| rank0 writes | 3 |
| nonzero-rank writes | 0 |
| injected failure | peer −15 / rank0 1 |

![Ranked checkpoint resume](../assets/ranked-checkpoint-resume.svg)

逐字节相等覆盖模型、AdamW first/second moments、optimizer step、global step、data cursor、seed与
配置，不是只比较一个权重文件。rank0在optimizer+barrier后原子写checkpoint，再发布step marker；
rank1只等待与读取。

写失败不发布marker，launcher在rank0返回1后终止等待peer。所有checkpoint/ready/tmp/ID验证后
删除。该节点是可靠性证据，无性能结论；准入Model-S checkpoint大小、耗时与一小步恢复smoke。

证据：[`ranked checkpoint resume`](../../../benchmarks/results/2026-08-25-ranked-checkpoint-resume/)
