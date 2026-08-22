# Experiment 112 — 56个Qwen共同候选，0个exact

按M32 heuristic顺序扫描全部56个共同solution，每个候选运行完整151936维B1/B2 logits门。56/56
受支持，0失败/OOM，0 exact。

![Qwen algorithm search](../assets/qwen-algorithm-search.svg)

最佳候选族以`75886`为首：max0.077602、mean0.011887、RMS0.015268，略优于75789但仍非exact。
56个index聚成少量相同数值signature，继续枚举同类候选边际很低。Qwen默认生成token已经跨slot
稳定，因此关闭当前heuristic集合的strict-exact搜索，不回退FP32。

数据见[`112-data`](112-data/)。
