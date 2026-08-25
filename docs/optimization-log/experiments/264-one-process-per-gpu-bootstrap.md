# Experiment 264 — 两个独立进程能否学习同一个模型并一起失败

Status: `bootstrap kept`

每次fresh launch先启动rank1等待ID，rank0原子发布opaque RCCL ID；两个进程分别持有tiny模型、
AdamW和GPU0/GPU1。三个run各3step，逐参数all-reduce average后与CPU global batch逐项比较。

![One process per GPU bootstrap](../assets/one-process-per-gpu-bootstrap.svg)

- 6个rank进程、18个rank-step完成；
- 12个Tensor/728个值跨rank位级一致；
- 相对CPU reference最大差1.19e-7；
- rank-group初始化+训练median 5.27s；
- 坏rank返回1，launcher终止等待中的peer，返回-15。

bootstrap保留。当前仍逐parameter collective，没有bucket/overlap性能意义；下一节点先做rank-local
bucket reducer与同步等价，再迁移gradient-ready Event。

证据：[`ranked bootstrap`](../../../benchmarks/results/2026-08-25-ranked-training-bootstrap/)
