# 2026-08-19 — M6 gradient buckets and four-rank environment failure

## Bucket implementation

Rank-local gradients are packed into contiguous HIP buckets with D2D async copies on
each communicator Stream, reduced once per bucket, averaged, unpacked with D2D copies,
and restored through checked `Value::set_grad`. A large bucket reduces all tiny-model
parameters in one collective and leaves every rank gradient identical.

## Two-rank 1MB payload matrix

| bucket elements | collectives | step ms | algorithmic GB/s |
|---:|---:|---:|---:|
| 4,096 | 64 | 6.6761 | 0.157 |
| 65,536 | 4 | 0.4083 | 2.568 |
| 262,144 | 1 | 0.22454 | 4.670 |

All values have zero numerical error. Reducing collective count from 64 to one makes
this fixed payload about 29.7× faster. This result measures synchronized communication
and averaging, not overlap with backward.

## Four-rank failure

All three four-rank initialization attempts failed with `unhandled system error`.
`NCCL_DEBUG=INFO` identifies the concrete cause: RCCL tries to extend multiple shared-
memory segments to 21,823,872 bytes each, while the container `/dev/shm` is only 64MB
with about 63MB available. The later socket failures follow the shared-memory ENOSPC.

Setting `NCCL_SHM_DISABLE=1` did not avoid that allocation path in the installed RCCL
2.28.3 build. Thus four GPUs are visible and XGMI-connected, but four-rank RCCL is not
validated in this container. This is an environment/resource failure, not evidence of
four-rank framework correctness or incorrectness.

Raw two-rank JSONL and a structured four-rank failure record are committed under
`benchmarks/results`.
