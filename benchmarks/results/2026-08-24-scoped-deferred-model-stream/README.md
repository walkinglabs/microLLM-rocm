# Scoped deferred model Stream evidence

This directory records Experiment 177 on one gfx942 MI300X. The candidate combines otherwise-
default operator/strided-copy Stream routing with deferred temporary allocation release. It is
compared with the retained legacy default-Stream plus exact-size-cache path.

The formal matrix uses BF16 Linear compute, batch one, contexts 32/512, one warm-up, two measured
forwards or complete training steps, three fresh processes per policy and alternating pair order.
It contains 48 processes and 24 paired correctness checks.

| Model | Mode | T32 candidate/control | T512 candidate/control | Largest deferred bytes |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | inference | 0.800× | 0.125× | 1,485,307,904 |
| Qwen2.5-0.5B | training | 0.562× | 0.235× | 7,110,592,008 |
| DeepSeek Distill 1.5B | inference | 0.862× | 0.147× | 2,685,403,136 |
| DeepSeek Distill 1.5B | training | 0.575× | 0.406× | 15,591,456,776 |

All inference complete-logit Max/RMS differences are zero. All paired training losses and the
observed post-update parameter are also exactly equal. No candidate row overflows its 8,192-block
table. Correctness is restored, but every performance row fails.

The Qwen T512 profile keeps 2,751 Kernels and the two launch families unchanged. Candidate
`hipMalloc`/`hipFree` calls rise from 1,180/867 to 2,559/2,557 for the whole profiled process;
their combined API duration rises from 39.6 ms to 183.0 ms. The candidate therefore stays
explicit and default-off. A future retry needs a same-Stream ordered allocator or activation
arena; another lexical routing wrapper is not a new hypothesis.

Files:

- `raw.jsonl`: 48 fresh-process measurements;
- `pairs.jsonl`: complete-logit/loss/parameter pair gates;
- `summary.json`: medians, speed ratios, memory and decision;
- `profile-summary.json`: parsed profiler attribution;
- `verification.json`: same-revision CPU/HIP/PyTorch gates and the pre-existing RCCL A/B;
- `*-hip-api-stats.csv` and `*-kernel-stats.csv`: raw profiler tables.
