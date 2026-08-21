# M32和M64能不能使用同一个hipBLASLt算法

当前引擎调用hipBLASLt时把algorithm传成`nullptr`，库会做隐式heuristic选择。为了验证FFN的B1/B2
差异，先不能假装已经知道它选了哪个算法；我们需要显式查询候选。

新CLI按与引擎相同的row-major转置描述查询：

```bash
microllm_bench_bf16_algorithms \
  --rows 32,64 --inner 1536 --columns 8960 \
  --max-algorithms 64 --workspace-bytes 33554432
```

MI300X结果：

| shape | 返回候选 | workspace范围 |
|---|---:|---:|
| M32×1536×8960 | 64 | 0–33,030,144 bytes |
| M64×1536×8960 | 64 | 0–33,030,144 bytes |
| 共同solution index | **53** | — |

所以“不同shape没有共同算法”被推翻。下一步可以选择共同index，分别注册给M32/M64，再运行同一个
P5完整值runner。solution index只对当前hipBLASLt版本有效，不能写成跨版本常量。

![BF16 algorithm inventory](../optimization-log/assets/bf16-algorithm-inventory.svg)

完整记录见[Experiment 109](../optimization-log/experiments/109-bf16-algorithm-inventory.md)。
