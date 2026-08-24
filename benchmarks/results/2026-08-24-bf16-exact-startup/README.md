# Exact BF16 gate/up solution cold and steady gate

Experiment 194 tests a narrower alternative after all-kernel preload failed:
select only the BF16-output gate/up GEMM used by each official T512 model.

For each model, three fresh tuner processes screen 64 candidates with complete
output before timing. The fastest median candidate common to all three runs is
then compared with the default in 24 fresh model processes: cold/steady,
default/exact, three runs per model.

| Model | Selected index | Operator | Cold forward | Process wall | Steady |
|---|---:|---:|---:|---:|---:|
| Qwen | 76074 | 1.059× | 0.990× | 0.978× | 0.973× |
| DeepSeek | 76091 | 1.032× | 0.996× | 0.981× | 1.007× |

Cold and process ratios are default time divided by exact time. Steady is exact
throughput divided by default throughput, so values above one are better in
every column.

All 64 candidates are common passing choices for both shapes. Complete model
logits are bit-exact in all 24 model processes, and engine peak ratios are
exactly 1.0. The local Event gains do not reduce first-use latency, Qwen steady
regresses 2.70%, and DeepSeek steady remains below the joint 1.01 keep gate.

Decision: reject exact gate/up registration and leave the default unchanged.
Solution indices are version-local evidence, not portable configuration.

Environment: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, HIP
runtime/driver 71399004. Files: tuning-raw.jsonl, model-raw.jsonl,
summary.json, and verification.json.
