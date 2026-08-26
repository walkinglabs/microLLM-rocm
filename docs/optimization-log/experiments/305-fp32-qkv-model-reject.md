# Experiment 305：Cache完全修好，完整Logits仍不稳

Status: default rejected; post-cache trace selected

## 两个Scope只命中Q/K/V

候选Q=296100、K/V=292135。registry key加入projection scope，因此同形Attention output不会误命中。
每个精度进程注册2项，Q/K/V共84 hits、2次cache miss和82次cache hit。

![FP32 QKV model gate](../../../benchmarks/results/2026-08-26-fp32-qkv-model-gate/model-gate.svg)

Block 0 BF16 K/V在B1/2/4/8全部位级相同，内部重复行也全部exact。这证明Experiment 304的算子
候选在真实权重/输入上真正修复了目标状态。

但完整logits不稳健：

| 指标 | 默认 | Candidate | 结果 |
|---|---:|---:|---:|
| 全局Max | 0.0013535 | 0.00125325 | 改善7.41% |
| 全局RMS | 0.0002294 | 0.0002908 | 恶化1.2677x |
| B2 Max ratio | — | 0.5519x | 改善 |
| B4 Max ratio | — | 1.0957x | 恶化 |
| B8 Max ratio | — | 1.1443x | 恶化 |

full-prefill速度B1/2/4/8为0.9014x/0.9505x/0.9816x/0.9907x，peak不变。decode速度变化不属于
候选区域，只作为噪声边界。

第一次正式调用在default performance结束后被runner拒绝：warmup 1意味着168次QKV dispatch，不是
84。修复公式并新增负向测试后从clean commit完整重跑，没有拼接旧数据。

## 决定

不设默认，也不扩展Qwen。候选路径保留为研究API，因为它提供了一个“cache exact但logits仍漂移”的
干净实验面。下一步在候选路径打开Block0 cache之后的Attention context/output、残差、FFN和block
output，定位第一处重新出现的差异。

证据：[`FP32 QKV model gate`](../../../benchmarks/results/2026-08-26-fp32-qkv-model-gate/)
