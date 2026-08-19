# N7 — 先证明两卡等价，再研究通信重叠

## 运行前预测

Two ranks see different local batches. Predict their parameter difference after one
step if gradients are not reduced. Then write the equivalent single-rank global batch.

## 构建

```bash
./scripts/configure.sh \
  -DCMAKE_BUILD_TYPE=Release \
  -DMICROLLM_ENABLE_HIP=ON \
  -DMICROLLM_ENABLE_RCCL=ON
./scripts/build.sh
ctest --test-dir build -L rccl --output-on-failure
```

Tests skip when fewer than two GPUs are visible. A skipped design is not reported as
a multi-GPU measurement.

## 双卡等价

The baseline uses one Stream/communicator per rank and averages every parameter
gradient. Observed after AdamW step one:

```text
rank_parameter_max_difference=0
single_vs_two_rank_max_difference=1.49012e-08
```

The reference is one CPU rank with the equivalent B2 global batch; GPU ranks each use
B1 and different examples.

## Bucket

The engine packs gradients with D2D async copies, all-reduces one contiguous bucket,
and unpacks with D2D copies. For a fixed 1MB two-rank payload:

| bucket count | step ms | algorithmic GB/s |
|---:|---:|---:|
| 64 | 6.6761 | 0.157 |
| 4 | 0.4083 | 2.568 |
| 1 | 0.22454 | 4.670 |

Fewer collectives are 29.7× faster here. That does not imply one giant bucket always
wins during backward; it may delay readiness and reduce overlap.

## Asynchronous overlap

`enqueue_all_reduce_sum` and `synchronize` are separate. With independent compute on
another Stream, three 50-repetition runs show 30.14–33.46% lower synthetic step time.

This proves hardware/runtime overlap capability, not backward-ready overlap. The eager
graph currently exposes gradients after backward completes; per-node readiness hooks
remain a next design problem.

## 四卡稳定失败

Four gfx942 GPUs are visible and every pair reports one-hop XGMI. Nevertheless,
four-rank communicator initialization fails. RCCL debug shows multiple 21,823,872-byte
shared-memory segments cannot fit in the container's 64MB `/dev/shm`.

`NCCL_SHM_DISABLE=1` did not avoid that path in RCCL 2.28.3. Therefore:

```text
two-card correctness: measured
two-card bucket/overlap: measured
four GPUs visible: measured
four-rank RCCL: failed, not supported by current evidence
```

## 下一步

Retry in an environment with adequate shared memory. If initialization succeeds,
rerun correctness before scaling. Do not carry two-card correctness forward as proof
of four-card behavior.
