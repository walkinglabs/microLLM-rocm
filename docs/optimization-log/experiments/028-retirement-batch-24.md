# Experiment 028 — retirement batch 16 → 24

Status: `discard`

The missing midpoint between retained batch 16 and rejected batch 32 was measured. The
first complete matrix returned Qwen train/generate `152.16/218.64` and DeepSeek
`79.78/77.50` token/s: all four are below batch 16. Backend allocations also increased.

The hard gate already failed, so extra processes were not used to search for a favorable
sample. Batch 16 remains the measured local optimum across 8/16/24/32. Raw evidence is
in [028-data](028-data/README.md).
