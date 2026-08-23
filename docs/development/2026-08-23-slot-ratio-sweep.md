# 2026-08-23：slot比例随流量翻转

## 固定轴

同一short-heavy/long-heavy请求，比较uniform和小桶:大桶`2:6/4:4/6:2`。总slot始终8，
capacity为520/2064，BF16 Cache和模型权重不变。

## 证据

- 48/48 fresh process；
- preflight三次0/0，pre/post设备门均通过；
- 两模型、两类流量、三种ratio全部token exact；
- short-heavy 6:2：吞吐保留84%–85%，KV少56%，TTFT P95低约59%；
- long-heavy 2:6：吞吐保留约87%，KV少19%，TTFT P95只高6%–7%；
- 配置方向相反时TTFT P95放大3×–5×。

## 决定

显式slot配方有价值，但不存在按模型固定的最优比例。uniform保持默认。下一候选只能在相关桶
空闲时切换配方，并必须测量Cache重分配、allocator reserved bytes和请求生命周期；若切换成本
过高，再进入paged Cache。

详见[Experiment 118](../optimization-log/experiments/118-slot-ratio-sweep.md)。
