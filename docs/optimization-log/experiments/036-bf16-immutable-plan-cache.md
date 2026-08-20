# Experiment 036 — immutable BF16 hipBLASLt plan cache

Status: `keep` — all four selected PyTorch BF16 performance rows pass

## Change

The BF16 path now caches immutable hipBLASLt descriptions/layouts per thread and exact
`(M,K,N,output dtype)` key. A public diagnostic API reports entries/hits/misses and can
clear the current thread cache. A test proves one miss, then one hit with stable entries.

No algorithm, Kernel, Tensor value, allocation policy, model input or precision boundary
changed.

## Verification

```text
CPU CTest              164/164 pass
ASan/UBSan             162/162 pass
HIP CTest               64/64 pass
Python/PyTorch oracle     4/4 pass
official candidate rows    6/6 exact expected tokens
```

## Three-process result

Baseline is Experiment 034, before the plan cache.

| Model | Decode | Speedup | Prefill | Speedup | microLLM/PyTorch BF16 decode/prefill |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 261.37 tok/s | 2.930× | 517.21 tok/s | 2.738× | 3.555× / 3.558× |
| DeepSeek Distill 1.5B | 76.83 tok/s | 2.549× | 1713.01 tok/s | 2.666× | 1.358× / 2.712× |

![BF16 plan cache result](../assets/bf16-plan-cache.svg)

All generated IDs and full-logit differences are unchanged from Experiment 034. Persistent
weights and measured engine current bytes are also unchanged; this is host setup reuse,
not a hidden memory trade.

## Why this does not revive Experiment 007

The old general FP32 descriptor cache regressed Qwen generation and DeepSeek training and
was removed. Experiment 036 does not restore it. It caches only the new BF16 path, where a
single DeepSeek token calls 196 BF16 Linear GEMMs and descriptor construction dominated
host submission after Kernel time fell.

Different evidence produced a different scoped decision. Broadening this cache to FP32,
algorithms, mutable scale pointers or cross-thread sharing still requires a separate
experiment.

## Decision and remaining boundary

Keep. The user-requested selected Qwen/DeepSeek inference speed target is now achieved for
this pinned short-prompt BF16 matrix. This does not prove long-context, batch>1, training,
Radeon, other ROCm versions or model-quality parity. Next work moves to a new track rather
than continuing tiny decode edits against an already-green matrix.
