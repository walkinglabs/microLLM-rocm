# Expanded BF16 grouped-QKV search

Experiment 191 expands Experiment 190 from 16 to 64 supported algorithms per official
T512 shape. The environment remains one gfx942 MI300X virtual function, HIP runtime/driver
`71399004`, and hipBLASLt `1.3.0`.

The first attempt accidentally launched the operator and model runners concurrently on
GPU0. Both processes were interrupted and no timing from those directories was retained.
All committed measurements below ran serially with exclusive access.

## Operator result

| Model | Exact index | Event speedup | Wall speedup | Reinitialize Event |
|---|---:|---:|---:|---:|
| Qwen | 64713 | 2.010× | 1.569× | 0.954× |
| DeepSeek | 64755 | 1.692× | 1.493× | 0.958× |

All 64 screened candidates per model pass complete projection output. Direct grouped
FP32 output remains unsupported, so candidate timing includes BF16 output plus three casts.

## Complete-model result

| Model | Steady speedup | Max/RMS logits | Peak ratio | Kernel setup |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1.0458× | 0.10881 / 0.02360 | 1.0034 | 207.9 ms |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.0295× | 0.07200 / 0.01255 | 1.0017 | 203.7 ms |

Both models pass steady throughput, BF16 precision, top-token and peak-memory gates.
However, the first exact grouped kernel initialization exceeds the 100 ms default/TTFT
gate. Device user arguments reduce 24/28 full initializations to one kernel initialization
plus 0.46/0.67 ms total argument setup, but they cannot remove the remaining library setup.

The decision is therefore: keep the exact explicit warmed policy; leave default inference
unchanged. A serving process may pay setup before admitting requests, but a one-shot CLI
must not hide it in warm-up.

## Device evidence

Phase delta subtracts `load+one` from `load+six` and divides by five. Relative to the
baseline records in `../2026-08-24-bf16-grouped-qkv/`:

- Qwen total Kernel time falls 5.960→5.848 ms; GEMM calls 217→169 and GEMM time
  3.194→2.807 ms, while added output casts raise cast time 0.314→0.600 ms.
- DeepSeek total Kernel time falls 10.473→10.256 ms; GEMM calls 253→197 and GEMM time
  6.479→6.140 ms, while cast time rises 0.406→0.723 ms.

Files include 12-process operator/model raw data and summary, final candidate one/six-step
Kernel CSVs, derived candidate profile deltas, and regression verification.
