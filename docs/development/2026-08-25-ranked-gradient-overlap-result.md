# 2026-08-25 — ranked gradient-ready overlap result

![Ranked gradient overlap](../optimization-log/assets/ranked-gradient-overlap-discard.svg)

五策略正式矩阵中，overlap把finish wait从3.080降到1.413ms（2.180×），但backward/enqueue
增加1.199ms，完整step仅`8.195→8.152ms`（1.0052×），低于1.01门。

later allocation、current/peak、完整参数、CPU、loss和故障门均通过。实现显式保留、不默认；
Model-S T32 ranked reducer局部路线关闭，后续只能以新context/模型/拓扑建立独立track。
