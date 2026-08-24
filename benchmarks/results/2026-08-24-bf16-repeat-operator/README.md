# Fused BF16-to-FP32 GQA repeat matrix

Three fresh processes per case/policy compare device-native `cast + repeat`
against one fused typed Kernel. Every output is exactly equal and timed sections
perform no payload transfer.

| Family | B1/T256 | B1/T512 | B1/T1024 | B2/T512 |
|---|---:|---:|---:|---:|
| Qwen | 1.253x | 1.291x | 0.996x | 1.004x |
| DeepSeek | 1.345x | 1.027x | 1.011x | 0.995x |

Only 3/8 pass the fixed 1.05 operator gate. Both batch-two cases fail, so no
model or CLI policy is implemented. The explicit primitive remains available
for research and small-B1 dispatch experiments.
