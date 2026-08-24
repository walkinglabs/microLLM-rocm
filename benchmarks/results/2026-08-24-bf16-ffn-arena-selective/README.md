# Selective complete-model BF16 FFN Arena

Experiment 184 repeats the complete Experiment 183 matrix with
`minimum_rows=512`. The model uses Arena only when `batch × sequence >= 512`.

All 60 fresh processes produce bit-exact complete logits and expected decode
tokens. Both eligible long-prefill rows pass 1.01:

| Model | T512 B1 | Arena bytes | Allocations baseline→selective |
|---|---:|---:|---:|
| Qwen | 1.019× | 18,612,224 | 3495→2895 |
| DeepSeek | 1.022× | 33,816,576 | 4075→3375 |

The eight bypass rows report zero entries/capacity/eligible calls, positive bypass
calls, and exactly the same allocation/peak counters as baseline. Their measured
ratios stay within 0.999×–1.005×.

Qwen T512 rocprofv3 keeps 5,642 Kernels and both launch counts. malloc/free falls
from 1,879/1,567 to 1,637/1,327; measured Kernel duration is 51.06/47.98 ms.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, four profiler CSVs,
and `verification.json`.
