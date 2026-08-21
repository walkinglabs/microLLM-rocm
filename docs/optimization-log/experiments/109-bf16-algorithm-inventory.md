# Experiment 109 — M32/M64有53个共同solution

当前默认BF16 matmul传`algo=nullptr`，没有可观察ID。本节点只增加heuristic inventory，不改变执行。

![BF16 algorithm inventory](../assets/bf16-algorithm-inventory.svg)

DeepSeek gate/up shape的M32和M64各请求64个候选，均返回64个；交集为53。workspace从0到约31.5MiB。
这证明same-algorithm实验可做，也证明不能以“shape不同所以没有共同kernel”结束调查。

下一节点从交集选择首个零workspace或共同高排名候选，增加版本局部的实验registry，并用官方P5
完整值与性能门决定是否保留；solution index不得成为默认硬编码。

数据见[`109-data`](109-data/)。
