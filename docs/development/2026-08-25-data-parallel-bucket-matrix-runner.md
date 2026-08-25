# 2026-08-25 — 先让真实 bucket 数量变化

tiny双卡在4MiB限制下只有一个bucket，无法测试readiness overlap。新增矩阵扫描4B、64B、4KiB、
4MiB，每种三个新进程并轮换顺序。

所有运行只在最终step做参数一致性检查，step 2–20进入steady median；20步loss必须逐项相同。
结果同时报告bucket count、communication、optimizer和total。

如果4MiB单bucket最快，下一步不是强行overlap，而是增加Model-S多bucket workload。

