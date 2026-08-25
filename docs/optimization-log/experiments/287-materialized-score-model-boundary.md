# Experiment 287：为什么自动边界是2048，不是512

Status: scoped gfx942/BF16/uniform policy admitted at T>=2048

## 八格官方模型门

Qwen2.5-0.5B与DeepSeek-Distill-1.5B，T512/T2048，B1/B2。每格current/materialized各3个
新进程，N32、warm-up 1、measured 3；每个子目录保留完整logits/token/pair证据。

![Materialized model boundary](../../../benchmarks/results/2026-08-25-materialized-attention-model-matrix/matrix.svg)

| 模型 | T | B | speedup | 完整logits | 性能门 |
|---|---:|---:|---:|---:|---|
| Qwen | 512 | 1 | 1.0479x | 位级相同 | 失败 |
| Qwen | 512 | 2 | 1.0484x | 位级相同 | 失败 |
| Qwen | 2048 | 1 | 1.1840x | 位级相同 | 通过 |
| Qwen | 2048 | 2 | 1.1747x | 位级相同 | 通过 |
| DeepSeek | 512 | 1 | 1.1049x | 位级相同 | 通过 |
| DeepSeek | 512 | 2 | 1.1062x | 位级相同 | 通过 |
| DeepSeek | 2048 | 1 | 1.3688x | 位级相同 | 通过 |
| DeepSeek | 2048 | 2 | 1.3209x | 位级相同 | 通过 |

Qwen T512两格不仅略低于1.05，leave-one也稳定约1.047–1.049，所以不能用四舍五入放行。跨两个
模型和两个batch，T2048最低仍为1.1747x，因此单调minimum为2048。

## 有界自动策略

只准入：gfx942、BF16 KV、uniform cached decode、已测Qwen/DeepSeek head签名、prefix>=2048。
其他GPU、FP32 cache、其他head结构和positions-aware serving保持旧路。CLI必须能显式off/on，自动策略
的最终JSON必须说明为什么命中或绕过。

证据：[`materialized model matrix`](../../../benchmarks/results/2026-08-25-materialized-attention-model-matrix/)
