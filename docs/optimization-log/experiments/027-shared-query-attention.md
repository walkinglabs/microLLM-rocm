# Experiment 027 — shared-memory query in cached Attention

Status: `discard`

The candidate loaded each head's query into shared memory once before scoring all cache
positions. Focused MHA/GQA, context fallback and official token tests passed.

Three-process medians:

```text
Qwen       219.30 → 219.70 token/s  +0.18%
DeepSeek    78.74 →  77.46 token/s  -1.63%
score       2.478439 → 2.469407
```

For these short sequences, hardware caching already makes query reads cheap; the extra
copy and synchronization do not pay. Candidate code is removed. Raw data is in
[027-data](027-data/README.md).
