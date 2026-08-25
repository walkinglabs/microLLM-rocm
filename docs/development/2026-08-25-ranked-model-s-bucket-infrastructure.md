# 2026-08-25 — ranked Model-S bucket measurement infrastructure

tiny模型只能证明两个独立进程能够通信，不能代表真实Reducer负载。它的12个小参数会落入一个
bucket，而且进程启动时间远大于通信时间。本节点把同一条one-process-per-GPU路径扩展到
Model-S的`B1×T32`一步训练，但仍不提前加入persistent bucket或ready overlap。

## 实现边界

- worker新增显式`tiny|model-s`选择；两个rank使用不同的确定性`B1×T32`输入；
- CPU reference把两份输入拼成`B2×T32`，保持相同global batch；
- 57个、15,586,176个元素的完整参数不写进JSON，而是分别保存为临时safetensors；
- 独立比较程序检查名称、shape、dtype、全部元素、有限性、最大误差和RMS误差；
- rank/rank必须Max和RMS都为零；rank/CPU必须同时通过Max `1e-2`和RMS `1e-5`；
- 每一步两rank loss的均值与CPU global-batch loss最大差必须不超过`1e-4`；
- 比较成功后删除三份临时权重，结果目录只保留小型JSON/stdout/stderr证据；
- `rank_group_ms`继续保留，但不再冒充Reducer性能。

## 可测时间边界

worker在参数落盘前分别记录：

```text
forward + backward
-> synchronous reducer
-> optimizer
-> complete training interval
```

两rank并发时，以较慢rank的时间作为进程组该阶段耗时。模型构造、RCCL bootstrap、参数保存和
CPU reference均留在训练区间之外。这样后续图可以区分“collective减少”与“训练真正变快”。

## 基础设施冒烟

在MI300X/gfx942、两rank、25 MiB、Model-S一步上：

- 57个参数形成3个自然bucket，collective/rank为3；
- 15,586,176个值跨rank Max/RMS均为0；
- CPU global-batch Max为`0.00627380143851`，RMS为`3.48336491075e-06`；
- 三份临时safetensors均在比较后删除；
- `DistributedRank.*`为5/5，包括普通、bucket和peer-failure真实进程测试。
- 完整RCCL标签回归为43/43；coverage audit为175个算子、42个graph API、123个测试文件。

这是基础设施冒烟，不是稳定性能结果。下一提交必须在干净revision上轮换
`per-parameter|bucket`，保存原始记录，并把collective、训练时间和Reducer时间画进同一张图。
