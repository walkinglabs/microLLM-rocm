# 2026-08-20 — Experiment 012 two-stage argmax

Large FP32 argmax now uses up to 256 partial blocks and one final reduction. Inputs below
32768 keep the original single block. Tie and non-finite behavior are unchanged.

```text
CPU debug                    152/152 pass
ASan/UBSan                   150/150 pass
HIP release                   54/54 pass
PyTorch operator parity         4/4 pass
argmax Kernel time          2.043 → 0.067 ms
Qwen generation median     142.25 → 147.41 token/s
DeepSeek median             53.04 → 53.36 token/s
robust score               1.752183 → 1.770568
```

Instrumented whole decode regressed even though the Kernel improved, so profiler tool
overhead is explicitly retained as contrary evidence.

Decision: `keep`.
