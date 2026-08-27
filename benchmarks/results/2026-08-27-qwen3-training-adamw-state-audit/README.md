# Qwen3 complete AdamW state audit

After one official B1/T32 step, both frameworks export first and second moments for all 310 independent parameters: 620 Tensors and 1,192,099,840 values. Step is separately required to equal one.

| Forward | Moment storage | Max / RMS | Decision |
|---|---|---:|---|
| FP32 | FP32 | `5.743e-5 / 3.552e-8` | pass |
| BF16 | FP32 | `3.641e-2 / 2.879e-5` | reject: Max > `1e-2` |

FP32's worst state is the tied embedding first moment. BF16 propagates the previously measured gradient split into first moments; second-moment family maxima also reach `2.54e-3` for embedding and `2.73e-3` for FFN down.

`*-raw.jsonl` keeps 620 per-Tensor records, `*-summary.json` keeps 18 family×moment groups, and `*-workers.json` keeps export metadata. Two temporary payloads total 9,536,957,448 bytes per precision and were deleted. Serialization is diagnostic-only.
