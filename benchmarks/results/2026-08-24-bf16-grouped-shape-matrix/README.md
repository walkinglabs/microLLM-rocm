# BF16 grouped rows 256/1024 capability matrix

Experiment 198 extends grouped QKV and gate/up beyond flattened rows 512.
Rows 256 represents shorter prefill; rows 1024 can represent B1/T1024 or
B2/T512 projection GEMMs.

Eight exact model/rows/projection cases run in three fresh processes. Each
process screens the first 64 of 10,227 algorithms with complete output before
timing.

| Rows | Model | Projection | Device-arguments Event | Reinitialize | Winner set |
|---:|---|---|---:|---:|---|
| 256 | Qwen | QKV | 1.695× | 0.930× | 64713, 64752 |
| 256 | Qwen | gate/up | 1.339× | 0.835× | 65197 |
| 256 | DeepSeek | QKV | 1.604× | 0.964× | 64699, 64713 |
| 256 | DeepSeek | gate/up | 1.236× | 0.924× | 65168 |
| 1024 | Qwen | QKV | 1.389× | 0.783× | 64713, 64754, 64755 |
| 1024 | Qwen | gate/up | 1.124× | 0.847× | 65168, 65200 |
| 1024 | DeepSeek | QKV | 1.397× | 0.921× | 64754, 64755 |
| 1024 | DeepSeek | gate/up | 1.225× | 0.916× | 65183, 65212 |

All 24 processes pass. QKV Max/RMS stays below 0.000244/0.000109;
gate/up is at most 0.00000763/0.000000416.

A one-process pilot suggested DeepSeek rows256 QKV reinitialization might be
faster. The three-process median is 0.964×, disproving that pilot. Stable
device arguments remain the only production candidate.

Decision: capability passes; continue to B1/T256, B1/T1024 and B2/T512
complete-model gates. Winner sets are version-local evidence, not defaults.

Environment: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, HIP
runtime/driver 71399004, hipBLASLt 1.3.0. Files: raw.jsonl, summary.json and
verification.json.
