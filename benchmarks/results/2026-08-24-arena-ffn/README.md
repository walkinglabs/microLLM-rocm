# Arena-backed official-shape FFN evidence

Experiment 181 runs the real Qwen2.5-0.5B and DeepSeek-Distill-1.5B dense FFN shapes:
two FP32 projections, SwiGLU and the down projection. Weights are deterministic constants so the
experiment isolates shape/execution behavior rather than checkpoint loading.

The 36-process matrix uses rows 32/512, three fresh processes per policy and twenty timed regions.

| Shape | Arena eager | Arena Graph | Graph setup break-even |
|---|---:|---:|---:|
| Qwen R32, 896→4864→896 | 1.148× | 1.202× | 23 replays |
| Qwen R512 | 3.033× | 2.970× | 1 replay |
| DeepSeek R32, 1536→8960→1536 | 1.042× | 1.005× | 568 replays |
| DeepSeek R512 | 1.667× | 1.679× | 2 replays |

Every complete output is bit-exact and each captured region has four nodes. The policy is shape
selective: DeepSeek R32 fails the 1.05 gate. This FP32 region is not yet the production BF16 FFN
path and is not enabled in `TransformerModel`.

The Qwen R512 profile keeps 101 executed Kernels. Arena reduces whole-process malloc/free from
80/79 to 11/10. Graph additionally reduces direct host Kernel launches from 100 to 12 plus 23
Graph launches.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, profiler CSVs and
`verification.json`.
