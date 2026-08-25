# Data-parallel training with RCCL

## Current execution model

The current `DataParallelTrainer` is a correctness-first, single-process controller for
multiple AMD GPUs. It creates one model and AdamW instance per device, shards the input
batch across ranks, averages gradients with RCCL, and applies identical optimizer steps.

Vendor-library state is rank-local too. hipBLASLt handles are stored per host thread and device
index; alternating GPU 0/1 GEMMs have a dedicated FP32/BF16 test. Reusing one process-wide handle
across ranks previously produced an invalid-device launch even though RCCL collectives themselves
were correct.

```text
rank-local input
→ rank-local forward/loss/backward
→ wait for each device backward
→ pack gradients into buckets
→ RCCL all-reduce sum
→ divide by world size
→ unpack gradients
→ rank-local AdamW step
→ verify parameters remain identical
```

## C++ API

```cpp
microllm::multi_gpu::DataParallelTrainer trainer(
    config,
    seed,
    {
        .device_indices = {0, 1},
        .maximum_bucket_bytes = 4 * 1024 * 1024,
        .optimizer = adamw_config,
    });

auto metrics = trainer.step({rank0_batch, rank1_batch}, step);
```

The step returns rank losses, mean loss, bucket/parameter/element counts, maximum
cross-rank parameter difference, and forward/backward, communication, optimizer, and
total wall times.

Current Experiment 251 evidence also shows that the post-optimizer full-parameter host audit
is not timed separately; on the tiny 20-step baseline the residual is 0.305 ms or 13.32% of
steady total. The next compatibility-preserving change adds a verification metric and interval.

All ranks must currently contribute the same number of targets. Otherwise averaging
rank-local mean gradients would weight small and large local batches equally, so the API
rejects the step.

## CLI

Build the RCCL preset, then:

```bash
./build/rccl-release/apps/microllm_distributed_train \
  --steps 3 \
  --bucket-bytes 4194304 \
  --trace /tmp/microllm-ddp-trace.jsonl
```

The CLI prints one JSON record per step and writes stage/layer/model timing records using
the same trace schema as the alignment infrastructure.

For a complete manifest, environment, raw metrics, trace, stderr, and summary package:

```bash
python3 tools/distributed/run.py \
  --binary build/rccl-release/apps/microllm_distributed_train \
  --output /tmp/microllm-distributed \
  --steps 20 --bucket-bytes 4194304
```

## Correctness evidence

The dedicated test runs three two-rank updates and compares with one CPU model trained
on the equivalent global batch. It requires:

- same initial seed and parameters;
- identical optimizer configuration and step count;
- both rank parameter sets identical after every step;
- single-global-batch and two-rank parameters within the declared tolerance;
- non-empty gradient buckets;
- finite loss and stage timing fields.

## Synchronization boundary

The baseline explicitly synchronizes each GPU after backward before communication
streams read gradients. This is correct but does not overlap backward and all-reduce.

A production one-process-per-GPU DDP path still needs:

- rank initialization from `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and a distributed unique
  communicator ID;
- autograd hooks that mark each parameter gradient ready;
- buckets ordered by observed backward readiness;
- device Events from compute streams to communication streams;
- asynchronous all-reduce work handles;
- gradient-as-bucket views to avoid pack/unpack copies;
- unused-parameter and uneven-input handling;
- timeout, abort, and cross-process error propagation;
- checkpoint ownership and rank-local weight placement.

RCCL provides the collective primitives; the reducer and readiness state machine remain
framework responsibilities.

Primary references:

- [AMD RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/develop/)
- [PyTorch DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch reducer implementation](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.cpp)
