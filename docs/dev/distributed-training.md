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

The separate default-off `DataParallelConfig.gradient_bucket_views` flag implements that next
step without changing the persistent-copy control. Each parameter keeps its original shape and
contiguous strides while sharing the reduced bucket Storage at an explicit element offset. It
requires persistent buckets and in-place averaging. The first Model-S smoke reports 114 views,
zero unpacked Storage/copies, six first-step bucket allocations, zero later allocations, and a
124,689,408-byte plan; formal three-policy measurement decides whether the combination advances.

Experiment 258 keeps views explicit. They improve communication/total 1.131x/1.067x versus
persistent-copy and 1.937x/1.367x versus transient; 45 losses and nine final parameter audits are
exact. Live bytes equal transient, but peak remains 33,269,000 bytes higher because backward still
creates ordinary gradients before the 114 pack copies. Direct Autograd accumulation into the
views is the next gate before defaults or readiness overlap.

The Experiment 259-only `direct_bucket_gradients` policy started after step one built a valid view
plan. Later steps zeroed each bucket, installed disjoint leaf-gradient targets, and skipped all 114
pack copies. The route is no longer part of `DataParallelConfig` or the CLI; these details remain
here so the recorded failure can be understood. Ordinary Autograd keeps only the independently
tested leaf accumulation-target primitive for future producer out-kernels.

Experiment 259 rejects the model route. Direct accumulation removes both copy families and makes
communication 2.173x faster than views, but producer gradients are still materialized before the
leaf add. Forward/backward falls to 0.830x and total to 0.991x; exact losses and parameters prove
this is a performance failure. The C++/CLI route is removed after recording the experiment. The
leaf-only Autograd target remains independently tested for a future producer out-kernel gate.

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

`DataParallelConfig.record_gradient_ready_order` is a default-off diagnostic. It installs one
hook per parameter and records an index only after the leaf's final graph contribution has been
accumulated. The CLI can emit parameter names, element counts and rank-local orders. This records
host enqueue order only; it does not claim device completion or change the synchronization path.
The audit runner maps that order to the exact natural bucket ranges before any Event/collective
prototype is admitted.

The default-off `overlap_gradient_communication` prototype requires persistent bucket views.
After step-one plan warm-up, each rank records a default-Stream Event when a bucket's final leaf is
ready. Communication Streams wait on both Events, pack, enqueue RCCL sum and in-place average, and
the trainer waits for all buckets before optimizer. The synchronous path remains the control.
Because this controller executes rank0 then rank1 backward in one process, it can overlap only
with the second rank's remaining work; it is not a one-process-per-GPU DDP claim. Candidate
forward/backward timing includes the final overlap wait, so `total_ms` is the primary gate.

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

The bootstrap now provides a rank-local `RankCommunicator`, opaque ID byte exchange, one-model /
one-optimizer / one-device worker, and a launcher with a group deadline. Rank 0 atomically
publishes the ID file; peers wait for the exact byte count. The tiny worker averages every
parameter gradient, then the launcher compares all 728 values across ranks and against a CPU
global-batch reference. A bad-rank injection terminates a peer blocked in communicator init.
This is local single-node bootstrap code, not yet the ready-bucket reducer or a performance claim.

The rank worker now has an explicit `per-parameter|bucket` reducer control. The synchronous bucket
path packs local contiguous FP32 gradients on the rank communication Stream, enqueues one averaged
collective per byte-limited range, unpacks after it, and restores checked parameter gradients.
Tiny's 12 parameters fit one 4 KiB bucket, reducing three-step collectives from 36 to 3 while
preserving rank/CPU parameters. The implementation is still allocating and synchronous; it is the
correctness baseline for later persistent ranked buckets and ready overlap.

The ranked path has since added persistent copy buckets, gradient-as-bucket views and fixed-order
ready Event overlap. Model-S T32 remains synchronous; the measured T128/two-MI300X/25 MiB track
retains explicit overlap. This is context-selective evidence, not a universal default.

Ranked checkpoint ownership is now explicit. All ranks complete optimizer and a barrier; rank0
alone atomically writes the complete model/AdamW/ExperimentState checkpoint and then publishes a
step-specific ready marker. Peers never write the shared path. A separate launcher checks an
interrupted 2+3-step trajectory against uninterrupted five-step checkpoint bytes and injects a
rank0 write failure. Model-S checkpoint size/runtime evidence remains the next gate.

The ranked worker and launcher now accept a general world size and build the CPU global batch from
every rank-local batch. World sizes one and two pass. Four visible MI300X VFs currently fail inside
`ncclCommInitRank` with a 64 MiB `/dev/shm`; the bounded group-init probe records all rank errors and
must not be reported as four-GPU training support.

`run_ranked.py --rccl-debug` follows AMD's per-process logging guidance. Its preflight reports
visible GPUs and `/dev/shm` total/free without inventing a required threshold. On the current
four-rank failure, all debug logs identify a 21,823,872-byte shared-memory segment that cannot be
created because the 64 MiB mount has insufficient free space; extracted diagnostics are retained,
while verbose raw logs are deleted.

RCCL provides the collective primitives; the reducer and readiness state machine remain
framework responsibilities.

Primary references:

- [AMD RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/develop/)
- [PyTorch DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch reducer implementation](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.cpp)
