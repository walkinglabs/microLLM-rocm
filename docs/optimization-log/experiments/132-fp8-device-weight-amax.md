# Experiment 132：权重准备快5.8倍，host payload D2H归零

## 两次基础设施失败

第一次pilot被旧Release二进制拒绝，0 raw。独立fresh build随后在CLI字符串拼接处编译失败；旧对象
让此前CTest没有暴露它。修复后新增binary contract，fresh Ninja 34/34 steps完成，再启动实验。

## 正式结果

| 模型/T | device prep | host prep | 降幅 | device/host TPS | device max/RMS |
|---|---:|---:|---:|---:|---:|
| Qwen T8 | 501ms | 2886ms | 82.6% | 1.023× | 4.488/0.664 |
| Qwen T512 | 501ms | 2825ms | 82.3% | 0.998× | 6.406/1.231 |
| Deep T8 | 2112ms | 12163ms | 82.6% | 0.997× | 6.583/1.111 |
| Deep T512 | 2116ms | 12403ms | 82.9% | 1.003× | 8.675/1.287 |

![Device weight amax](../assets/fp8-device-weight-amax.svg)

Qwen168个Tensor扫描1.431GB，Deep197个扫描6.174GB；全部在device，host扫描0，scale summary
明确标为不可在host获得。resident/peak和scalar unsupported-shape fallback与Exp127相同。

完整logits不bit-exact：RMS大多略降，部分max error变化，四个top保持，但四个FP8门仍失败。
因此保留device weight amax作为FP8实验的优选准备方式，不把它扩写成“模型精度通过”。host模式
继续作为容易读取scale范围的reference。

下一性能债是single-block device weight amax本身：虽然已经比D2H快5.8×，大权重仍由一个block
扫描。应先做multi-block reduction microbenchmark，再确认端到端准备时间；热路径TPS不是该节点。
