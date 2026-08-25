# 2026-08-25 — rank-local同步bucket结果

![Ranked gradient buckets](../optimization-log/assets/ranked-gradient-buckets.svg)

tiny三步collective/rank从36降到3，12个rank进程参数exact、CPU最大差1.19e-7、故障门不退化。
但wall仅1.0037×，启动成本占主导，因此只保留正确性baseline。

下一节点切换Model-S one-step自然多bucket，不提前迁移persistent/overlap。
