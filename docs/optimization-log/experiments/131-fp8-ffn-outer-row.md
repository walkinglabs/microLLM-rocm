# Experiment 131：T512速度恢复十六倍，精度仍未过门

FFN gate/up/down使用row scale；Attention/output固定0.2。当前MI300 runtime拒绝native outer-vector，
所以FFN先device row反量化到BF16再GEMM。

| 模型/T | FP8 TPS | /BF16 | /Exp129 | max/RMS | outer fallback | gate |
|---|---:|---:|---:|---:|---:|---|
| Qwen T8 | 1646 | 0.727× | 1.095× | 1.261/0.189 | 288 | fail |
| Qwen T512 | 68689 | 0.746× | **14.09×** | 1.731/0.396 | 288 | fail |
| Deep T8 | 975 | 0.708× | 1.156× | 1.136/0.217 | 336 | fail |
| Deep T512 | 35391 | 0.715× | **16.22×** | 1.175/0.235 | 336 | fail |

![FFN outer row](../assets/fp8-ffn-outer-row.svg)

DeepSeek RMS相对Exp129改善50.4%/5.5%；Qwen T8略改善，T512从0.293退到0.396。所有top token
保持，但RMS仍为门的3.8–7.9倍。内存优势不变。

结论：保留FFN-only路由和显式计数；拒绝默认FP8与当前软件fallback性能。下一步不能继续假设
outer-vector会在MI300上突然可用。可研究两条反驳路线：离线逐层固定Attention scale以避免
one-block动态扫描，或让FFN row量化直接进入支持vector scale的更新ROCm/架构；任何路线仍需
完整logits门。
