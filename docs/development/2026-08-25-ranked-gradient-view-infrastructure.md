# 2026-08-25 — ranked gradient-as-bucket view infrastructure

Experiment 268的persistent-copy plan仍为bucket和57个输出gradient分别保留Storage，并执行57次
unpack。本节点只改变输出Tensor解释：每个参数gradient成为对应bucket Storage上的连续view。

## Contract

`RankGradientBucketPlan`把view policy纳入不可变合同。第一次按parameter顺序计算每个view的
shape、contiguous stride与storage offset；后续step必须复用相同parameter身份、shape、bucket
limit和view policy。切换copy/view必须先`clear()`。

world-one测试检查：

- 两个gradient共享bucket Storage；offset精确为0和2；
- 第一次logical allocation只有1个bucket，不再为输出gradient分配Storage；
- 第二次allocation为0，两个data address跨step稳定；
- unpack copy为0，数值不变；
- 无persistent plan使用views和对同一plan切换policy都会被拒绝。

worker新增显式`bucket-views`，matrix扩展为四策略并报告views相对逐参数、transient和
persistent-copy的steady时间与显存。

## Pilot

Model-S、两rank、三步、25 MiB：

- backend allocation `[3,0,0]`，plan reuse `[0,1,1]`；
- 57 views/step、57 pack/step、0 unpack/step；
- plan容量62,344,704 bytes/rank，是persistent-copy的一半；
- final current 249,378,816 bytes，与逐参数/transient相同；
- peak 324,929,288 bytes，比逐参数多62,344,704，比persistent-copy少62,344,704；
- 单次相对persistent-copy Reducer/total约1.043×/1.074×；
- 完整参数、CPU、loss与故障门通过。

完整RCCL标签45/45，`DistributedRank.*` 5/5，测试文件仍为123。pilot不形成性能结论；下一
提交从干净revision跑四策略各三次，再决定是否迁移gradient-ready overlap。
