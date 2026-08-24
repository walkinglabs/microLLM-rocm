# Experiment 192 — move grouped setup before admission

Status: `keep` explicit prewarm API; default unchanged

## Question

Experiment 191 keeps an exact warmed grouped-QKV policy but rejects the default because first
kernel setup is about 204–208 ms. Can serving prepare every pointer plan before accepting a request,
and can we prove that setup moved instead of disappearing into warm-up?

## API contract

`TransformerModel::prewarm_bf16_grouped_qkv(rows)` requires:

- HIP model with BF16 Attention weights already prepared;
- QKV Arena enabled for the requested flattened row count;
- one exact grouped algorithm registered for that shape/environment.

It allocates one dummy FP32 activation, acquires the same Arena entry used by real inference, and
asks every block to execute QKV once. This creates one shared grouped kernel, one device argument
record per block and all exact pointer plans. The report separates total, kernel and argument setup.
Calling the same row again returns `already_warm=true` without executing work.

## Formal first-request matrix

Each policy runs in a fresh process, zero warm-up, one T512 prefill. Three process orders are
alternated for each model.

| Model | Baseline first | Lazy first | Prewarm | Prewarmed first | Combined |
|---|---:|---:|---:|---:|---:|
| Qwen | 4972.7 ms | 5744.1 ms | 915.3 ms | 4851.9 ms | 5767.3 ms |
| DeepSeek | 4992.9 ms | 5741.4 ms | 886.5 ms | 4794.7 ms | 5681.1 ms |

The baseline itself contains about five seconds of ordinary vendor-plan/code first use. Prewarm
makes the admitted request 892/947 ms faster than lazy grouped, while combined time stays within
roughly ±60 ms of lazy total. Kernel setup is 208.2/201.4 ms; argument setup is 0.64/1.15 ms.

![Grouped QKV prewarm](../assets/bf16-grouped-qkv-prewarm.svg)

## Correctness and decision

All 18 complete-logit outputs are finite and stay inside the retained BF16 Max/RMS boundary.
Prewarm produces exactly one plan hit per block on the first admitted request; lazy policy produces
zero plan hits and builds on-request.

Keep the explicit model/CLI prewarm API and expose all timing fields. Do not call this a startup
speedup: total work is not removed. Default CLI and scheduler behavior remains unchanged until
pre-admission lifecycle ownership is integrated into the serving layer.

Raw evidence:
[`benchmarks/results/2026-08-24-bf16-grouped-qkv-prewarm/`](../../../benchmarks/results/2026-08-24-bf16-grouped-qkv-prewarm/).
