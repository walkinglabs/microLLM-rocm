# 2026-08-25：保序Score物化进入显式模型门

## 新策略与旧split策略必须分开

Model现在有两个互斥的研究开关：

```text
partial split：每段log-sum-exp，已知模型logits失败
materialized：只并行QK，finalize保留原顺序，算子位级相同
```

同时打开会立即拒绝。默认两个都关闭。materialized接口：

```cpp
model.set_cached_attention_materialized_scores(true, minimum_sequence);
```

CLI：

```bash
--cached-attention-materialized true \
--cached-attention-minimum-sequence 512
```

只有uniform cached prefix达到minimum才使用；早期token和positions-aware/divergent路径不变。JSON输出
实际策略与minimum，避免结果文件分不清两个算法。

## 当前门

- Model API默认关闭，minimum默认512；
- minimum不正拒绝；
- materialized与S>0 split互斥；
- CPU tiny cached logits与完整prefix一致；
- MI300X B1/B2预分配stride cache完整logits对CPU通过；
- CLI二进制合同锁定参数、互斥错误与输出字段；
- 官方A/B runner新增`candidate_policy=materialized`，current明确关闭两个候选。

下一节点沿用T2048/B2/BF16/N64三对协议。要求完整303,872 logits逐项相同、64 token相同、性能
中位至少1.05x、leave-one至少1.01x，并记录score Tensor带来的allocation/peak。
