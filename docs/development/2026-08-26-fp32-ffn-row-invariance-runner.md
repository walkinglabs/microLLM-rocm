# 2026-08-26 — FP32 FFN gate/up solution矩阵runner

现有C++ row-invariance工具已能表达M2048–16384、K1536、N8960。这个节点补充同进程default Event
基线，并让每个候选输出四个M的speedup。候选仍严格按“CPU sentinel、完整重复block位级一致、再计时”
的顺序执行。

Python runner只接受四个M各64个inventory、共同候选、四项default/candidate时间和完整正确性字段。
绿色候选必须同时block exact且每个M不低于0.95×；推荐index只从绿色集合按总Event时间选择。所有index
只对当前gfx942、ROCm和hipBLASLt版本有效，不写通用默认。
