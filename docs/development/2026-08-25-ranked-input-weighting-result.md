# 2026-08-25 — ranked uneven-input weighting result

![Ranked input weighting](../optimization-log/assets/ranked-input-weighting.svg)

tiny `[B1,B2]` equal-only在参数通信前两rank共同失败；token-weighted使用0.6667/1.3333 scale，
三步rank exact，CPU Max/RMS `8.18e-8/8.79e-9`，loss差`1.94e-7`。

显式weighted模式保留；weighted ready overlap仍拒绝。下一节点Model-S one-step smoke。
