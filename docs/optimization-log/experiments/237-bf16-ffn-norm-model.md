# Experiment 237 — RMSNorm直写Arena，这次小算子收益进了整模

Status: `keep default for BF16 FFN Arena`

## 路由边界

候选只改graph-free prefill中已经命中BF16 FFN Arena的路径：

```text
hidden FP32
→ ffn_norm 直接写 workspace.input_bf16
→ bf16_ffn_precast_out_
```

training、Autograd、CPU reference、详细trace、cached decode、Arena bypass和非BF16权重仍走旧路。
`bf16_ffn_precast_out_`验证已填充workspace的dtype/shape/device/alias，不把“调用者说已cast”
当成无条件信任。

Arena因minimum rows拒绝时，第一版fallback又进入普通`forward_tensor`并第二次查Arena，
导致bypass计数加两次。现有模型测试抓到它。修正后fallback临时绕开cache，只记一次决策。

## 正式整模门

同一个二进制显式交替`false/true`，B1T1024、每格3进程、2 warm-up + 5 measured：

| Model | Speedup | Complete-logit Max/RMS | Allocation reduction | Peak |
|---|---:|---:|---:|---|
| Qwen | 1.01223× | 0 / 0 | 120 | unchanged |
| DeepSeek | 1.00921× | 0 / 0 | 140 | unchanged |

120=`24 layers × 5 measured`，140=`28 × 5`，说明每层都真正删掉了一个临时FP32
Norm Tensor，而不是计时噪声。

![BF16 FFN Norm model gate](../assets/bf16-ffn-norm-model.svg)

## 默认决定

- `set_bf16_ffn_arena_enabled(true)`同时启用Norm fusion；
- CLI在用户未显式给值时，跟随BF16 FFN Arena自动启用；
- `--bf16-ffn-norm-fusion false`保留为反驳/回归路径；
- 一次未显式传该flag的Qwen实际运行记录`bf16_ffn_norm_fusion_enabled=true`，证明默认生效。

发布回归：CPU 345/345、ASan/UBSan 343/343、PyTorch-enabled 319/319、完整CPU/HIP
544/544（3个条件跳过）、HIP标签187/187；107个测试文件全部注册。

证据：[`full-model gate`](../../../benchmarks/results/2026-08-25-bf16-ffn-norm-model-gate/)
