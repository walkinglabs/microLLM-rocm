# Step 102 — Ranked ready-bucket weighting

Status: implemented; formal Model-S measurement pending

Step 101证明RCCL重叠确实发生：finish快1.930x；失败来自每步57次leaf scale带来的1.520ms。
最后一个最小反驳实验不改变数学公式、bucket边界或通信时机，只改变scale的位置：

```text
leaf ready
→ Event
→ RCCL Stream pack一个完整bucket
→ 整个bucket乘local token weight
→ all-reduce average
```

Model-S每步scale应从57降到3。必须先证明同步与overlap最终15,586,176个参数逐项一致、CPU门
不退化、rank一致、later allocation仍为0，再测T128三轮。整步达到1.01x才保留性能路由；否则
关闭weighted overlap优化track，继续使用同步weighted正确性路径。

实现新增显式`bucket-weighted-overlap`路由。`RankCommunicator`可以在自己的Stream上先乘
local scale、再all-reduce、最后除world size；`RankGradientBucketPlan`逐bucket记录scale调用。
旧leaf路线没有被替换。

Tiny `[B1,B2]`三步得到leaf scale `[0,0,0]`、bucket scale `[1,1,1]`、rank/CPU门通过。
Model-S T32一次pilot中，策略参数逐项完全一致，steady step约1.080x；这是dirty pilot，只证明
测量器和路线可用。完整RCCL标签51/51、`DistributedRank.*` 9/9。下一提交必须从干净revision
运行T128三轮。
