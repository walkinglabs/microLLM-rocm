# Experiment 013 GroupedGemm availability probe

MI300X/gfx942, FP32 decode:

| M | K | N vector | Grouped launches | Fallbacks |
|---:|---:|---|---:|---:|
| 1 | 128 | 128,64,64 | 0 | 1 |
| 1 | 128 | 128,128,128 | 0 | 1 |

Both fallbacks were numerically correct. No end-to-end measurement was run because no
grouped Kernel was available.
