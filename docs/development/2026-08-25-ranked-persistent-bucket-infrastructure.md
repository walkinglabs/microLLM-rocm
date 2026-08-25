# 2026-08-25 — ranked persistent bucket infrastructure

Experiment 267把transient bucket的steady回退定位到每步60次backend allocation和114次copy。
本节点只改变Storage生命周期，不改变bucket范围、collective、pack/unpack、optimizer或同步时机。

## Public contract

新增move-only `RankGradientBucketPlan`。它绑定：

- rank、world size和本地device；
- parameter指针身份、顺序与gradient shape；
- bucket byte limit和由此得到的range。

第一次调用为3个bucket和57个unpacked gradient分配Storage；后续调用必须复用同一容量与地址。
任何合同变化都明确报错，调用者必须先`clear()`，不能静默重建。

worker新增显式`persistent-bucket`策略，逐step报告plan reuse、容量、allocation/copy以及engine
current/peak/cached/reserved。matrix可用`--policies per-parameter bucket persistent-bucket`运行
三策略，并分别报告persistent相对逐参数和transient的steady Reducer、完整step与显存差。

## Pilot evidence

Model-S、两rank、`B1×T32`、三步、25 MiB单次pilot：

- plan reuse为`[0,1,1]`；logical/backend allocation为`[60,0,0]`；
- total allocated bytes为`[124689408,0,0]`；
- collective仍为`[3,3,3]`，pack/unpack仍为每步57/57；
- plan容量为124,689,408 bytes/rank；
- steady Reducer约2.78ms，transient pilot约4.25ms，逐参数pilot约2.88ms；
- 常驻比逐参数/transient多62,344,704 bytes；
- 峰值比逐参数多124,689,408 bytes，比transient多72,376,320 bytes；
- 15,586,176个参数值rank exact，CPU/loss与peer-failure门不变。

pilot不能形成性能结论。基础设施完整RCCL标签44/44，新增world-one地址复用/合同拒绝测试通过；
`DistributedRank.*` 5/5，测试文件审计仍为123。下一提交必须从干净revision跑三策略各三次。
