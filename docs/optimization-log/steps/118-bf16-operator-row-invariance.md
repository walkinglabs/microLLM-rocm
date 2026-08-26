# Step 118 — BF16 operator row-invariance search

Status: complete; gate/up solution search closed

Experiment 300证明64个solution都共同支持M1/2/4/8，但75892完整模型不对齐。这个节点把其他模型
组件全部拿走，只检查gate/up形状本身。

构造一行确定性BF16输入`[1,1536]`和确定性BF16权重`[1536,8960]`，将输入重复成M2/4/8。对
inventory交集的64个version-local index逐个：

1. 注册精确M shape；
2. 与readable CPU/BF16 reference检查完整输出；
3. 比较M1第0行与M2/4/8第0行的8960个值；
4. 报告bitwise、Max/RMS、workspace和Event时间。

候选只有在四个shape都support、CPU门通过且row0位级相同时才进入完整模型。若多个通过，优先选择
最小workspace，再比较速度；若0个通过，固定vendor solution路线关闭。solution index仍不进入默认。

结果：64/64 support、64/64 CPU reference exact、64/64 M1/2/4/8 row-invariant，最大误差0。
75892算子本身也exact，因此完整模型失败来自进入FFN前已经不同的输入。Step 119转向Block 0 prefill
K/V cache前缀审计；不选择任何默认solution。
