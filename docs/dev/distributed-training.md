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

`DataParallelConfig.parameter_check_interval` now defaults to `1`, preserving that every-step
audit. `0` explicitly disables it and `N` checks steps divisible by `N`. Metrics and traces expose
whether the check ran and its separate wall time. A skipped check is performance evidence only;
it is never reported as a successful rank-consistency check.

Experiment 253 shows the tiny CLI model has one natural bucket at both 4 KiB and 4 MiB.
Artificially forcing 12 buckets raises communication from 0.34–0.39 ms to 1.18–1.26 ms.
It is therefore not an overlap workload; Model-S support is the next prerequisite.

All ranks must currently contribute the same number of targets. Otherwise averaging
rank-local mean gradients would weight small and large local batches equally, so the API
rejects the step.

## CLI

Build the RCCL preset, then:

```bash
./build/rccl-release/apps/microllm_distributed_train \
  --model tiny \
  --steps 3 \
  --bucket-bytes 4194304 \
  --parameter-check-interval 1 \
  --trace /tmp/microllm-ddp-trace.jsonl
```

`--model model-s --context 32 --batch 1` selects the 15,586,176-parameter teaching model.
The current 4 MiB run naturally creates 12 buckets and records bucket parameter/elements plus
maximum process-wide engine peak/current bytes. The current allocation ledger covers all HIP
devices in the process, so it must not be summed or described as a per-rank measurement. This
workload is the prerequisite for real overlap work.

Experiment 254 selects 25 MiB/3 buckets as the current reducer baseline: 19.76 ms total,
6.825 ms communication and 603,383,808 peak engine bytes per rank. It is not yet overlapped.

Bucket metrics now expose bucket/average/unpacked Tensor counts, pack/unpack copies, temporary
bytes and communication-stage allocation deltas. The 3-bucket Model-S identity is 126 backend
allocations and 374,068,224 temporary bytes per step; RCCL non-default streams prevent pool reuse.

Bucket averaging now defaults to an address-stable in-place scale after all-reduce sum. The
explicit CLI control exists only for same-binary A/B. This removes the average Tensor family and
is a prerequisite for persistent bucket addresses; it does not remove pack/unpack copies.

Experiment 256 retains that default: Model-S communication/total improve 1.269x/1.107x with
unchanged peak, exact losses/parameters and RCCL 22/22. Persistent storage targets the remaining
6 bucket plus 114 unpacked backend allocations.

`DataParallelConfig.persistent_gradient_buckets` and the matching CLI flag are default-off while
the Model-S gate runs. The move-only plan is built after the first real backward, binds to the
exact communicator/parameter order/shapes/bucket limit, and keeps its Storage until the trainer
is destroyed. Step one therefore allocates normally; later communication stages must report zero
allocation/backend/cache calls. Persistent capacity, whether a plan was reused, and process-wide
current/peak bytes are emitted separately. Persistent mode requires in-place averaging.

Experiment 257 keeps this path explicit: later backend allocations fall 120→0,
communication/total improve 1.681x/1.285x, and all losses/parameters match. It is not a default
because live/peak bytes rise by 124,689,408/157,958,408. The next reducer step replaces the 114
unpacked gradient Storage objects and copies with parameter-shaped views into reduced buckets.

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
