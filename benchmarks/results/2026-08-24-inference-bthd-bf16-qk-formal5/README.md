# BTHD BF16 Q/K five-process formal gate

This is the expanded formal rerun after the initial three-process DeepSeek
window missed the performance threshold. Every policy/model pair uses five fresh
processes, alternating process order, two warm-up forwards and five measured
forwards at T512/B1 on the same MI300X.

| Model | FP32 Q/K boundary | Direct BF16 Q/K | Speedup | Complete logits | Peak |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 110,961 tok/s | 113,441 tok/s | 1.0224x | bit-exact | unchanged |
| DeepSeek Distill 1.5B | 56,979 tok/s | 58,333 tok/s | 1.0238x | bit-exact | unchanged |

All candidate records report exactly `blocks * 7` retained Q/K dispatches; every
control record reports zero. The policy is kept explicit and default-off until a
broader sequence, batch, GPU, and backend-version matrix passes.
