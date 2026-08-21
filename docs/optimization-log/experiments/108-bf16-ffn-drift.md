# Experiment 108 — cast exact，首差在gate BF16 GEMM

第一次正式尝试因BF16 trace values为空而被完整值合同拒绝。修复低精度TraceSession并将capture上限
提高到覆盖B2 `[64,8960]` 后，同一runner三对fresh process通过。

![BF16 FFN drift](../assets/bf16-ffn-drift.svg)

## 结果

48个stage、三次统计完全相同，B2 duplicate全部exact：

| stage | B1/B2 shape | max-abs | relative-L2 |
|---|---|---:|---:|
| input_bf16 | `[32,1536]`/`[64,1536]` | 0 | 0 |
| gate | `[32,8960]`/`[64,8960]` | 0.015625 | 0.00006106 |
| up | `[32,8960]`/`[64,8960]` | 0.001953125 | 0.00001942 |
| activated | `[32,8960]`/`[64,8960]` | 0.0078125 | 0.00011019 |
| down | `[32,1536]`/`[64,1536]` | 0.0013504 | 0.00007269 |

gate是首个非零点；up是相同input上的独立GEMM，也出现较小差异。证据支持M32/M64的hipBLASLt
BF16 algorithm/累加路径差异，而不是cast、SwiGLU起源或row污染。

## 保留与下一步

保留低精度trace修复、`bf16_ffn_diagnostics`和内部stage证据；默认非诊断FFN仍走原函数且不保存
中间Tensor。下一节点记录gate/up plan ID并做same-algorithm反驳，不能直接把FFN退回FP32。

数据见[`108-data`](108-data/)。
