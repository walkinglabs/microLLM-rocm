# MI300X microLLM versus Python/PyTorch official HF models

This directory contains FP32 Qwen2.5-0.5B and
DeepSeek-R1-Distill-Qwen-1.5B inference and one-step training rows.

Both inference paths generated the same expected token IDs. Training loss and the
observed final-norm update also agree closely. The training timing is first-step only
and is not a steady-state claim.

```text
PyTorch       2.11.0+rocm7.13.0rc2
Transformers  4.55.4
HIP           7.13.99004
GPU           AMD Instinct MI300X VF, gfx942
measurement   4/4 matched
```

See `microLLM.jsonl`, `pytorch.jsonl`, and `comparison.jsonl`.
