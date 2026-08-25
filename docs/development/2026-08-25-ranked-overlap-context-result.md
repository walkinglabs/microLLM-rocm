# 2026-08-25 — ranked overlap context-scale result

![Ranked overlap context scale](../optimization-log/assets/ranked-overlap-context-scale.svg)

T32/T128正式矩阵证明overlap是尺度策略：T32 total `0.9995×`，T128 `1.0923×`；finish分别
2.022×/2.235×，显存增量均0。删除最慢run后T128仍1.069×。

完整参数、CPU、loss和peer-failure门通过。当前Model-S/two-MI300X/25MiB轨道显式选择
T32同步、T128 overlap；不推广为一般默认。下一主线进入rank0 checkpoint ownership/resume。
