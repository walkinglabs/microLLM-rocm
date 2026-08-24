# Attention core Arena model rejection

Experiment 187 converts the measured top allocation source into a caller-owned
workspace. It reuses one expanded K/V slot and keeps scaled Q, probabilities and
output in stable storage. The candidate is selected only at sequence 512.

All 60 complete logits and decode tokens are exact. Eligible performance misses 1.01:

| Model | T512 ratio | Core backing | Allocations baseline→core | Peak increase |
|---|---:|---:|---:|---:|
| Qwen | 1.004× | 20,185,088 | 2895→2295 | 2,752,512 |
| DeepSeek | 1.002× | 22,020,096 | 3375→2675 | 4,718,592 |

Eight short cases have zero entries/capacity/eligible calls. Qwen T512 profiler keeps
5,642 Kernels and direct launch counts while malloc/free falls 1,637/1,327→1,395/1,087.
Kernel duration is 47.67/46.78 ms. The model policy is rejected and default-off.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, four profiler CSVs and
`verification.json`.
