# Experiment 087 data map

- `summary.json`: keep/discard contract and selected medians.
- `t2048-pair-*`: three alternating DeepSeek B1/B8 baseline/candidate process pairs.
- `qwen-t512-b8-pair-*`, `deepseek-t512-b8-pair-*`: targeted rechecks that reject the
  single-process T512 B8 regression hypothesis.
- `qwen-matrix-*`, `deepseek-matrix-*`: candidate/PyTorch T8/T512/T2048 B1/B8 surveys.
- `gates.json`: CPU, HIP, sanitizer and default/non-default Stream safety gates.

The retained source uses immediate exact-size address reuse only while all engine work remains on
the legacy default Stream. Constructing a non-default Stream permanently disables the pool for that
device. Peak active bytes and KV bytes are unchanged; reserved/cached bytes remain separate metrics.
