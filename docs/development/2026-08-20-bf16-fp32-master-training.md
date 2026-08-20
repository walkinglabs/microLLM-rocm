# 2026-08-20 — BF16 Linear training with FP32 masters

`LinearPrecision::BFloat16` runs rounded Linear forwards through the retained BF16
hipBLASLt path while autograd gradients, parameters and AdamW state remain FP32. The policy
is visible in model summaries and `microllm_hf_train_step --linear-precision bf16`.

Tiny full-model Python STE oracles cover logits/loss/every gradient; CPU 20-step loss,
HIP zero-transfer graph execution and official Qwen/DeepSeek multi-step updates pass.

Three-process microLLM BF16 throughput is `3.12×/2.58×` PyTorch BF16 autocast, but only
`0.918×/0.906×` the retained microLLM FP32 path. Peak engine memory is unchanged. This is a
correct training foundation and a measured internal performance failure, not a completed
mixed-precision optimizer.
