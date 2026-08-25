# Experiment 239 — Attention Norm直接写QKV Arena

Status: `keep default for BF16 QKV Arena`

## 只切一个边界

正式A/B的两边都保留Experiment 237的FFN Norm默认融合，只切换：

```text
old: attention RMSNorm FP32 → QKV Arena cast
new: attention RMSNorm → workspace.input_bf16 → QKV precast projection
```

`bf16_qkv_projection_precast_out_`完整检查workspace、权重、输出和alias。trace、cached、
training、非BF16、非Arena与minimum-row bypass走旧路。bypass也使用单次cache决策，不重复计数。

## 整模门

B1T1024、same binary、每格3进程、2 warm-up + 5 measured：

| Model | Speedup | Logit Max/RMS | Allocation reduction | Peak reduction |
|---|---:|---:|---:|---:|
| Qwen | 1.01309× | 0 / 0 | 120 | 3,670,016 B |
| DeepSeek | 1.01303× | 0 / 0 | 140 | 6,291,456 B |

![BF16 Attention Norm model gate](../assets/bf16-attention-norm-model.svg)

allocation减少仍精确是`layers × 5 measured`，而且这次FP32 Norm临时Tensor不再参与峰值，
所以峰值也下降。

## 默认决定

- `set_bf16_qkv_arena_enabled(true)`默认开启Attention Norm fusion；
- CLI未显式传值时跟随QKV Arena；
- `--bf16-attention-norm-fusion false`保留为反驳路径；
- 下一步重新profile，不直接假设剩余cast的来源。

发布回归：CPU 345/345、ASan/UBSan 343/343、PyTorch-enabled 319/319、完整CPU/HIP
544/544（3个条件跳过）、HIP标签187/187；107个测试文件全部注册。

证据：[`full-model gate`](../../../benchmarks/results/2026-08-25-bf16-attention-norm-model-gate/)
