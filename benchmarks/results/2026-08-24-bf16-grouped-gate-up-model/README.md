# Pointer-stable BF16 grouped gate/up model gate

Experiment 196 integrates the two-operation capability from Experiment 195
only into the existing caller-owned BF16 FFN Arena path.

One initialized GroupedGemm kernel is shared per exact shape/device/Stream.
Each Transformer block owns only a device user-argument record binding its
persistent gate/up weights and the shared Arena input/output addresses.
Unregistered shapes and non-Arena BF16 FFN keep the old two-GEMM path.

## Complete model

Three fresh baseline and three fresh grouped processes run two warm-ups plus
five measured T512 prefills per model.

| Model | Baseline | Grouped | Speedup | Max/RMS | Peak ratio | Kernel setup |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 93471 tok/s | 95118 tok/s | 1.0176× | 0.07028/0.01538 | 1.000008 | 57.0 ms |
| DeepSeek | 50157 tok/s | 50746 tok/s | 1.0117× | 0.06139/0.01029 | 1.000003 | 56.8 ms |

All 12 complete-logit outputs are finite, remain inside the established BF16
0.25/0.05 envelope, and preserve top-1 tokens. Per-block argument setup totals
0.58/0.70 ms. The final records contain 24/28 plan entries, 144/168 hits and
168/196 grouped dispatches exactly.

## Profile

The committed phase delta subtracts load+one from load+six and divides by five.
The legacy file name three-step-kernel-stats.csv contains the six-step table.

| Model | GEMM calls | GEMM time | Total Kernel |
|---|---:|---:|---:|
| Qwen | 217→193 | 3.139→3.034 ms, 1.035× | 5.733→5.630 ms, 1.018× |
| DeepSeek | 253→225 | 6.603→6.474 ms, 1.020× | 10.504→10.526 ms, 0.998× |

Exactly one GEMM submission per block disappears. The instrumented DeepSeek
total-Kernel counterexample is retained even though the uninstrumented
three-process throughput gate passes.

Decision: keep the explicit exact policy and infrastructure. No version-local
index is installed by default.

Environment: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, HIP
runtime/driver 71399004, hipBLASLt 1.3.0. Files include raw.jsonl,
summary.json, four profile directories, and verification.json.
