# 2026-08-25 — tiny 只有一个自然 bucket

![Data parallel bucket matrix](../optimization-log/assets/data-parallel-bucket-matrix.svg)

4KiB/4MiB限制都产生一个bucket，通信0.34/0.39ms。4B/64B人为切出12个bucket，通信升到
1.26/1.18ms，total也变慢。240个loss完全相同，末步参数差为0。

因此tiny不能作为readiness overlap验收。下一步增加Model-S，让正常bucket限制自然产生多个
bucket，再测正确性、显存和阶段时间。

