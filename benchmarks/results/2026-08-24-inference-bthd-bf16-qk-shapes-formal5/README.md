# Direct BF16 Q/K five-process shape matrix

Five fresh processes per policy/model/case, alternating case and policy order,
produce 60 uninstrumented processes. Each process runs two warm-up and five
measured forwards.

| Model | Case | Speedup | Complete logits | Peak |
|---|---|---:|---:|---:|
| Qwen | B1/T256 | 1.0244x | bit-exact | unchanged |
| Qwen | B1/T1024 | 1.0138x | bit-exact | unchanged |
| Qwen | B2/T512 | 1.0128x | two rows bit-exact | unchanged |
| DeepSeek | B1/T256 | 1.0188x | bit-exact | unchanged |
| DeepSeek | B1/T1024 | 1.0152x | bit-exact | unchanged |
| DeepSeek | B2/T512 | 1.0194x | two rows bit-exact | unchanged |

Every candidate reports exactly `blocks * 7` retained Q/K dispatches and every
control reports zero. This evidence retains the exact policy but does not make it
a portable default.
