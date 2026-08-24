# FP32 Attention hipBLASLt solution screening

Experiment 188 enumerates and executes exact batched QK/PV layouts for Qwen and
DeepSeek T512. Each candidate passes a complete finite Max/RMS gate before timing.

Three fresh processes per case produce 64 common passing solution indices:

| Model | Op | Default Event | Recommended | Index | Speedup |
|---|---|---:|---:|---:|---:|
| Qwen | QK | 0.021128 ms | 0.015956 ms | 305434 | 1.324× |
| Qwen | PV | 0.017720 ms | 0.014794 ms | 294519 | 1.198× |
| DeepSeek | QK | 0.023253 ms | 0.018562 ms | 305460 | 1.253× |
| DeepSeek | PV | 0.021088 ms | 0.018923 ms | 292941 | 1.114× |

Recommended maximum absolute/RMS error is at most `4.47035e-7` / `6.6408e-8`.
All four use zero workspace. These are operator candidates, not model speed claims;
registration and complete-logit/end-to-end gates belong to the next experiment.

Files: `raw.jsonl`, `summary.json`, and `verification.json`.
