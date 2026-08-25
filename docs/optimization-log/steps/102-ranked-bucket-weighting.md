# Step 102 — Ranked ready-bucket weighting

Status: planned

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
