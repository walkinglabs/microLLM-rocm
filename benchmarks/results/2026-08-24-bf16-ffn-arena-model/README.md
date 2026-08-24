# Complete-model BF16 FFN Arena evidence

Experiment 183 compares the existing BF16 Attention+FFN path with one opt-in,
single-backing FFN Arena shared by every Transformer block. The matrix uses fixed
Qwen2.5-0.5B and DeepSeek-Distill-Qwen-1.5B revisions.

The 60 fresh processes cover two models, five cases, two policies and three process
runs. Baseline/Arena order alternates. Every complete last-position logit vector is
bit-exact, and all decode rows reproduce the fixed generated tokens.

| Model | T32 B1 | T512 B1 | T32 B4 | decode B1 | decode B4 |
|---|---:|---:|---:|---:|---:|
| Qwen | 1.000× | 1.022× | 1.001× | 1.031× | 0.998× |
| DeepSeek | 1.001× | 1.020× | 1.001× | 0.998× | 1.000× |

Only three of ten rows pass the 1.01 keep gate; none crosses the 0.98 regression
boundary. The universal policy is rejected and remains default-off. A following
experiment may test the evidence-backed `rows>=512` subset.

On Qwen T512, rocprofv3 records the same 5,642 Kernels and direct launches. Arena
reduces whole-process malloc/free from 1,879/1,567 to 1,637/1,327. Kernel duration
changes from 49.07 to 49.44 ms, so the small end-to-end gain is a host allocation
effect, not faster device math.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, four profiler stats CSVs,
and `verification.json`.
