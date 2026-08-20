# 2026-08-20 — Experiment 009 fused cached decode Attention

For FP32 cached sequence up to 4096, one block per query head now fuses score, stable
softmax and context. Longer prefixes use the original readable fallback.

```text
CPU debug                    152/152 pass
ASan/UBSan                   150/150 pass
HIP release                   54/54 pass
PyTorch operator parity         4/4 pass
Qwen generation median   134.87 → 142.25 token/s
DeepSeek median           49.05 → 53.04 token/s
robust score             1.695566 → 1.752183
```

Interleaved Qwen context results are -7.8% at one token, then +18.5%/+18.5%/+57.9%
at 32/128/512. The one-token failure and long-sequence fallback are part of the kept
contract.

Decision: `keep` for cached decode only; prefill and backward remain incomplete.
