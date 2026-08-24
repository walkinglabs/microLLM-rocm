# BF16 QKV Arena model rejection

Experiment 185 measures caller-owned shared-cast Q/K/V on top of the retained FFN
`rows>=512` baseline. Candidate QKV Arena also uses a 512-row threshold.

All 60 complete-logit files are bit-exact and decode tokens match. Both eligible
long-prefill rows miss 1.01:

| Model | T512 incremental ratio | QKV bytes | Allocations baseline→QKV |
|---|---:|---:|---:|
| Qwen | 1.004× | 4,456,448 | 2895→2415 |
| DeepSeek | 1.005× | 7,864,320 | 3375→2815 |

Eight short cases report zero QKV entries/capacity and positive bypass counts. One
Qwen decode B1 process aggregate is 0.976× despite identical allocation/peak counters,
so the matrix also records one below-0.98 row.

Qwen T512 profiler keeps 5,642 Kernels and direct launch counts. malloc/free changes
1,637/1,327→1,446/1,135; Kernel duration is 49.63/49.27 ms. Allocation removal is
real, but it does not yield the required end-to-end gain.

Files: `raw.jsonl`, `summary.json`, `profile-summary.json`, four profiler CSVs and
`verification.json`.
