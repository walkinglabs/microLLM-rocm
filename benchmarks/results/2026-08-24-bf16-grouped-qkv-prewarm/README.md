# BF16 grouped-QKV explicit prewarm

Experiment 192 measures first-request behavior for the final exact grouped indices
`64713/64755` on gfx942 MI300X, HIP runtime/driver `71399004`, hipBLASLt `1.3.0`.

Each model/policy runs in three fresh processes with zero warm-up and exactly one T512
prefill. Policies are baseline, lazy grouped setup on the request, and explicit grouped
prewarm before the request.

| Model | Baseline first | Lazy grouped first | Prewarm | Prewarmed first | Prewarm + first |
|---|---:|---:|---:|---:|---:|
| Qwen | 4972.7 ms | 5744.1 ms | 915.3 ms | 4851.9 ms | 5767.3 ms |
| DeepSeek | 4992.9 ms | 5741.4 ms | 886.5 ms | 4794.7 ms | 5681.1 ms |

The ordinary BF16 model already pays roughly five seconds of first-use vendor-plan/code
setup. Lazy grouped inference adds about 0.75–0.77 seconds to that request. Explicit prewarm
moves 0.89–0.92 seconds before admission and makes the first admitted request 0.89–0.95
seconds faster than lazy grouped. Total prewarm-plus-request time remains close to lazy total;
the work is moved, not erased.

Within prewarm, the grouped kernel reports 208.2/201.4 ms setup and all per-block device
arguments report only 0.64/1.15 ms. The rest of prewarm is the deliberate dummy QKV compute,
casts, Arena setup and synchronization.

All 18 processes remain finite and within the same BF16 complete-logit envelope. The explicit
prewarm API is retained for serving admission. Default one-shot inference remains unchanged.

Files: `raw.jsonl`, `summary.json`, and `verification.json`.
