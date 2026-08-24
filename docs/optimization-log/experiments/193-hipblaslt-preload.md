# Experiment 193 — all-kernel preload is not targeted warmup

Status: discard

## Question

Experiment 192 found roughly five seconds of ordinary first-use work before grouped QKV is even
considered. hipBLASLt exposes HIPBLASLT_PRELOAD_KERNELS; can asking the library to load every
kernel make the first request finish sooner?

The falsifiable rule was written first:

- run each policy in a fresh process with no warm-up;
- include process wall time, not only the timed forward;
- require complete-logit correctness;
- retain preload only if both Qwen and DeepSeek improve;
- treat a ≥1.25× slowdown on both models as a stable counterexample.

## Why this is different from ordinary warm-up

Normal warm-up executes the exact shapes that a service expects. All-kernel preload instead asks
the library to prepare a much wider inventory. The [official hipBLASLt environment-variable
reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.1/reference/env-variables.html)
documents the library controls; the installed 1.3.0 library also exposes the preload switch.
The experiment sets it explicitly to 0 or 1 so the parent shell cannot change a row.

## Formal fresh-process matrix

All rows use T512, batch 1, three fresh processes and alternating policy order.

| Model | FP32 first | BF16 lazy first | BF16 preload first | Forward slowdown | Process slowdown |
|---|---:|---:|---:|---:|---:|
| Qwen | 3582.3 ms | 5030.0 ms | 17189.7 ms | 3.417× | 3.140× |
| DeepSeek | 3563.8 ms | 4967.9 ms | 17123.1 ms | 3.447× | 2.938× |

![hipBLASLt preload failure](../assets/hipblaslt-preload.svg)

FP32 itself needs about 3.6 seconds on first use. BF16 lazy adds about 1.4 seconds. Preloading the
whole inventory raises first forward to about 17.2 seconds and does not reduce engine peak memory.
That is the opposite of the hypothesis.

## Correctness and decision

All 18 outputs are finite. Qwen Max/RMS is 0.10527/0.01596; DeepSeek is
0.04505/0.00875, both inside the retained BF16 boundary.

Reject preload-all and make no runtime default change. Do not add a second model-level API that
only wraps the existing full-forward warm-up. The useful lifecycle remains targeted:
prewarm exact shapes or pointer plans before admission, and always report the cost separately.

Raw evidence:
[benchmarks/results/2026-08-24-hipblaslt-preload/](../../../benchmarks/results/2026-08-24-hipblaslt-preload/).
