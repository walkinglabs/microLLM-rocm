# MI300X microLLM versus Python/PyTorch built-in models

This directory contains comparison-grade FP32 rows for tiny, Model-S and Model-M. Each
model/mode ran in a separate process. The automatic comparator rejects mismatched
workload fields.

```text
PyTorch       2.11.0+rocm7.13.0rc2
HIP           7.13.99004
GPU           AMD Instinct MI300X VF, gfx942
measurement   comparison profile, 6/6 matched
```

See `microLLM.jsonl`, `pytorch.jsonl`, and `comparison.jsonl`. Memory ratios compare
engine-owned microLLM peak bytes with PyTorch peak allocated bytes.
