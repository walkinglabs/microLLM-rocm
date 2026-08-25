# 2026-08-25 — ranked ready-bucket weighting result

![Ranked bucket weighting](../optimization-log/assets/ranked-bucket-weighting.svg)

Model-S `[B1,B2]` T128把每步57次leaf scale降为3次bucket scale后，steady step从9.262ms
降到8.687ms，达到1.0661x；finish为2.035x。三轮完整参数逐项相同，CPU门和显存门通过。

该路由以显式候选保留，不设为默认：一轮仍为0.951x，leave-one敏感性最低1.0027x。下一节点
只尝试把57次pack copy和3次scale融合成3次持久gather-scale Kernel。
