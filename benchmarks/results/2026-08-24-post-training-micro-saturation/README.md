# Post-training-micro saturation evidence

Experiment 213 profiles the retained default training path, then subtracts a load-plus-one-step
run from a load-plus-three-step run. The difference is exactly two training steps and removes
checkpoint loading, BF16 mirror preparation, library setup and first allocation effects.

## Result

| Model | Two-step Kernel | GEMM | AdamW | Together |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 67.087 ms | 55.87% | 16.85% | 72.71% |
| DeepSeek Distill 1.5B | 154.251 ms | 62.25% | 21.52% | 83.77% |

Every remaining local category has a perfect-removal upper bound below 1.046× on Qwen and
1.024× on DeepSeek. Three immediately preceding candidates already demonstrate that real gains
are smaller or negative. The local launch/cast fusion track is closed; future training work must
change GEMM grouping/algorithms, optimizer bandwidth/layout, or graph-wide liveness/capture.

`summary.json` contains the exact category shares and bounds. The four compact Kernel-stat CSVs
are sufficient to reproduce the subtraction without committing large per-dispatch traces.
