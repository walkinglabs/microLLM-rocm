# 2026-08-25 — rank-local synchronous gradient buckets

`all_reduce_rank_gradients` groups one rank's contiguous FP32 parameter gradients under the same
byte-limit rule as the single-process reducer. For each range it allocates one local bucket,
enqueues pack copies, rank-local RCCL average and unpack copies on one communication Stream, waits,
then assigns checked parameter gradients.

The worker and launcher expose an explicit `per-parameter|bucket` control and report collective,
bucket, pack and unpack counts. Tiny uses one 4 KiB bucket per step: three steps reduce collectives
36→3, perform 36 pack/unpack copies, keep 728 values rank-exact and remain within 1.19e-7 of CPU.

world-one API validation, real two-process bucket smoke, ordinary per-parameter smoke and peer
failure all pass. The route is intentionally allocating/synchronous; formal two-policy repeated
measurement precedes persistent Storage or gradient-ready overlap migration.

Infrastructure gates pass RCCL-labelled `43/43`; coverage remains 42 graph API entries and 123
registered native/Python test sources.
