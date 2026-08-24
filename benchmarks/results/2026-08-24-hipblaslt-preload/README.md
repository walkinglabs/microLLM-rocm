# hipBLASLt all-kernel preload cold-start gate

Experiment 193 asks whether HIPBLASLT_PRELOAD_KERNELS=1 can remove the
first-use delay observed after Experiment 192. The answer on this measured
MI300X environment is no.

Every row is a fresh process, has zero warm-up, and executes one T512 prefill.
The three policies run in alternating order for three processes per model:

1. FP32 with preload disabled;
2. BF16 with the ordinary lazy library path;
3. BF16 with every hipBLASLt kernel requested before use.

| Model | FP32 first forward | BF16 lazy first | BF16 preload-all first | Lazy process wall | Preload process wall |
|---|---:|---:|---:|---:|---:|
| Qwen | 3582.3 ms | 5030.0 ms | 17189.7 ms | 6175.9 ms | 19394.6 ms |
| DeepSeek | 3563.8 ms | 4967.9 ms | 17123.1 ms | 6694.0 ms | 19668.5 ms |

Preloading all kernels makes BF16 first-forward latency 3.417×/3.447× slower
and complete process wall time 3.140×/2.938× slower. Engine peak memory is
unchanged at 1,309,500,928/4,561,625,088 bytes. FP32 also pays about 3.6 seconds
on its first forward, showing that a large part of this boundary is common
first-use ROCm/library work rather than a BF16 graph bug.

All 18 complete-logit outputs are finite. The maximum FP32/BF16 Max/RMS
differences are 0.10527/0.01596 for Qwen and 0.04505/0.00875 for DeepSeek,
inside the declared BF16 envelope.

Decision: reject all-kernel preload. Keep lazy loading for ordinary execution
and use only measured, targeted prewarm such as grouped QKV when a serving
lifecycle owns admission.

Environment: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, HIP
runtime/driver 71399004. Files: raw.jsonl, summary.json, and verification.json.
