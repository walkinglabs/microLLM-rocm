# MI300X microLLM versus Python/PyTorch official HF models

This directory contains FP32 Qwen2.5-0.5B and
DeepSeek-R1-Distill-Qwen-1.5B multi-step inference and training rows.

Both sides run two warm-up iterations and five measured iterations. Warm-up time is
excluded and peak allocator counters are reset after warm-up. Inference paths generated
the same expected token IDs on every repetition; training loss and final-norm updates
also remain aligned.

```text
PyTorch       2.11.0+rocm7.13.0rc2
Transformers  4.55.4
HIP           7.13.99004
GPU           AMD Instinct MI300X VF, gfx942
measurement   4/4 matched, warm-up=2, measured steps=5
```

See `microLLM.jsonl`, `pytorch.jsonl`, and `comparison.jsonl`.
