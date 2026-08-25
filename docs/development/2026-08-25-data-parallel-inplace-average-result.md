# 2026-08-25 — in-place bucket average保留

![Data parallel in-place average](../optimization-log/assets/data-parallel-inplace-average.svg)

Model-S同二进制A/B中，communication从6.60降到5.20ms，total从19.21降到17.35ms；
peak不变，loss和参数门通过，RCCL22/22。

默认改为原地average，allocating control仍可显式复测。剩余120次backend allocation来自6个
bucket和114个unpacked gradient，下一步做persistent plan。

