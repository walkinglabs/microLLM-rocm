# 2026-08-25：Split-sequence进入显式模型A/B

## 这次只接哪条路

算子矩阵已经过门，但模型默认仍必须保持不变。本节点只给uniform cached decode增加一个固定S开关：

```cpp
model.set_cached_attention_split_sequence(splits, minimum_sequence);
```

CLI对应：

```bash
--cached-attention-splits 32 \
--cached-attention-minimum-sequence 512
```

默认`splits=0`，完全使用旧fused路径。只有cache当前逻辑prefix达到minimum后才切换；较短prefix继续
走旧路，避免sequence小于S，也避免早期token承担两阶段开销。实际S还会取`min(S, prefix)`。

positions-aware/divergent-row decode没有接入。它每行prefix不同，需要另一套partial映射，不能把
uniform结果直接推广过去。

## 接口状态怎样传播

TransformerModel把设置传给每个Block，再传到Attention。设置不写入checkpoint，因为它是运行时
执行策略，不改变权重或训练状态。`to(device)`不清除策略；CLI会在权重准备后显式设置并把最终S和
minimum写入JSON结果。

## 基础正确性

- 默认getter为S0/minimum512；
- S小于0、大于32或minimum不正会拒绝；
- CPU tiny MHA/GQA从第一个token启用S2，cached logits仍与完整prefix一致；
- MI300X tiny真实预分配cache使用更大capacity stride，S2完整logits对CPU通过；
- B1和B2都覆盖，计时区间仍为0 payload transfer；
- CLI二进制合同锁定两个参数和输出字段。

这仍不是性能结论。下一节点会先跑官方完整cached logits，再跑DeepSeek T2048/B2/BF16、S32、N64
的current/candidate三对fresh process，同时记录token、allocation、peak和KV bytes。
