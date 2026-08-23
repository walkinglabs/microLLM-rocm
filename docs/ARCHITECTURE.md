# Architecture

The beginner course is maintained on the separate
[`tutorial/beginner-course`](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course)
branch. Exact framework shape, error, tolerance, and PyTorch gates live in
[OPERATOR_CONTRACTS.zh-CN.md](OPERATOR_CONTRACTS.zh-CN.md) on `main`.
Model weight ownership, external naming, and safetensors boundaries are specified in
[WEIGHTS.md](WEIGHTS.md).

## Dependency direction

```text
base
├── runtime-api
├── core (Storage, Tensor, TensorView)
│   └── ops-api
│       ├── ops-reference
│       └── ops-hip
│           └── ops-tuned/vendor
├── autograd
├── nn/model
├── io (tokenizer, data, checkpoint)
├── training / inference
└── distributed

C ABI ← C++ engine
Python / PyTorch bindings ← C ABI or stable operator ABI
course / apps / benchmarks ← public engine APIs
```

## Tensor ownership boundary

Serving requests are owned by `ReferenceScheduler`, not by the model. Each request has an
independent B=1 `KVCache`, RNG and lifecycle state. The scheduler releases Cache Storage on
completion and exposes snapshots/metrics. It is intentionally serial; a future slot-batched
scheduler must preserve this state machine. `forward_cached_rows()` is now the divergent-position
model oracle: it serializes B1 views over shared batch Storage and does not claim parallel speed.
`forward_prefill_cached_row()` is the matching admission oracle: it computes a new prompt in a
temporary B1 Cache, copies the active prefix into one empty shared row on the same device, and
leaves every other row untouched. Neither oracle is the final parallel serving path.
`ContinuousBatchScheduler` owns the request-to-row map above these model primitives. It admits only
at scheduler-step boundaries, resets a row on length/stop/cancel, and reuses the lowest free slot.
Its shared Cache allocation persists while active-prefix bytes fall to zero. Divergent decode still
uses the serial B1 oracle; the scheduler API does not imply a parallel Kernel.
`forward_cached_active_rows()` is the compact execution seam: it receives only survivor tokens and
their fixed row IDs, advances those shared-Storage views, and leaves inactive full-capacity rows
untouched. Full, uniform slots still use the original parallel batch path.
The active path is positions-aware rather than row-serial: device `positions[A]` and
`cache_rows[A]` tensors parameterize decode-only RoPE, K/V store and cached Attention while QKV,
FFN and output projection remain ordinary active-batch operations.
For CPU-origin decode tokens, token/position/row metadata shares one `[3,A]` Storage transfer and
is split into zero-copy device views; already-device token callers keep their explicit fallback.
Equal-length pending prompts use `forward_prefill_cached_rows()`: one temporary `[A,T]` cache runs
the model batch, then each prefix is mapped into its target empty shared-cache row. Different prompt
lengths remain separate stable groups.
`ContinuousBatchConfig::max_sequence_length` bounds the persistent shared Cache for a known
workload. Zero keeps the model maximum; a positive value must fit the model and every submitted
request. Official serving runners choose the largest prompt-plus-output request, then verify the
exact layer/head/element-size allocation formula instead of treating estimated memory as evidence.
When every row returns to logical position zero, backing Storage remains available for reuse.
Full-row admission may use the first-allocation fast path only if every layer's K/V Storage is
undefined; otherwise it overwrites the reusable row prefixes through the existing-storage path.
Selection diagnostics are opt-in and intentionally host-synchronizing. They expose request, slot,
position, producer path/batch, device argmax and top-2 margin for numerical investigations. The
default serving path performs none of these logit copies. Equal-length prefill batching also has an
experimental off switch for controlled attribution; production/default behavior remains batched.
Official diagnostic inputs may provide explicit prompt-seed offsets. This is a benchmark input
contract used to hold token sequences constant while swapping or duplicating local batch rows; it
does not alter scheduler admission semantics.
Continuous request snapshots expose submission-to-first-token and submission-to-terminal wall
latency. Negative values mean the lifecycle event has not occurred. Official reports preserve raw
request arrays and derive P50/P95 with linear interpolation.
`LengthBucketedBatchScheduler` composes multiple fixed-capacity continuous schedulers. Every child
references the same `TransformerModel` but owns a separate KV cache. The smallest compatible bucket
is deterministic by default. An opt-in admission rule may place a request in the first larger
compatible bucket with immediate capacity; it never migrates submitted requests or places a long
request into an undersized bucket.
Official continuous workloads may delay individual submissions by a logical arrival step. Request
wall latency starts at actual submission; the logical clock is a state-machine axis, not a fixed-QPS
wall-time load generator.
Graph-free full/last-logit inference participates in the same opt-in TraceSession layer/model
contract as autograd forward. An inactive session performs no Tensor value copies. Full-value
official diagnostics require a single, zero-warm-up prefill step and are explicitly excluded from
timing claims.
When a layer trace is active, inference additionally records block-zero substage values. The scope
is deliberately one block: it locates numerical drift without multiplying diagnostic storage by
the model depth. The ordinary path and every block's computation graph remain unchanged.
The fused BF16 FFN has a diagnostic-return variant exposing device-resident intermediates. The
ordinary API passes a null diagnostic sink and retains no extra Tensor handles. Trace value capture
supports all floating dtypes; unsupported captured integer formats fail instead of appearing empty.
See [serving scheduler](dev/serving-scheduler.zh-CN.md).


`Storage` owns an allocation through shared lifetime state. `Tensor` owns metadata
and shares Storage. `TensorView` is non-owning and is the low-level operator seam.

An operator implementation must not infer ownership from a data pointer. Output,
workspace, device, and execution stream are explicit. This allows the same HIP
kernel to consume engine-owned or PyTorch-owned allocations.

## Tensor N0 invariants

- scalar shape `{}` contains one element;
- zero-sized dimensions are valid;
- negative dimensions and negative strides are rejected;
- storage offset and strides are expressed in elements;
- a view must remain entirely inside Storage;
- reshape is zero-copy and requires contiguous input;
- transpose and positive-step slice are zero-copy;
- contiguous materializes logical order;
- CPU float32 is the first implemented data path; unsupported operations fail
  explicitly.

## Operator implementation levels

```text
reference      readable CPU truth source
hip_readable   direct HIP implementation used for teaching and diagnosis
hip_tuned      fused, tiled, or architecture-specialized implementation
vendor         ROCm library implementation such as hipBLASLt
```

Dispatch may select among validated implementations. It may never bypass the
correctness gate merely because a candidate benchmarks faster.

The AdamW operator is a concrete example: `Scalar` and `Vectorized` are selectable, but
`Auto` stays on Scalar because exact-shape float4 wins did not survive the official-model
gate. Selection policy is evidence, not an alias for the newest Kernel.

hipBLASLt GEMM also supports contiguous strided batches: leading Tensor dimensions become
the batch count and last-two dimensions remain the matrix contract. Explicit batched
selection is tested independently; Auto is not changed by operator-only timing.

## Attention layout boundary, explained simply

Q/K projection produces a contiguous four-dimensional table in this order:

```text
[batch, token, head, value-inside-one-head]     = BTHD
```

Attention wants to visit every head before visiting its tokens:

```text
[batch, head, token, value-inside-one-head]     = BHTD
```

A `transpose` view only changes the address formula. It is like saying “read the same
spreadsheet by columns”; it does not move numbers. A Kernel that accepts only contiguous
rows forces `contiguous()` to copy the whole spreadsheet into the new order.

For attention-bias models with split-half RoPE, the graph uses one layout-aware boundary:

```text
projection BTHD
    │  read old B,T,H,D address
    ▼
bias + split-half RoPE Kernel
    │  write new B,H,T,D address
    ▼
Attention BHTD
```

Backward performs the inverse rotation and inverse address mapping in one Kernel. Its BTHD
output is already contiguous for the projection gradient. The same Tensor can be reshaped
to `[B*T,H*D]` for bias reduction without copying. This is a layout optimization, not a new
RoPE formula.

The public contracts are:

- `rope_split_half_bias_bthd([B,T,H,D], [H*D]) -> [B,H,T,D]`;
- `rope_split_half_bias_bthd_backward([B,H,T,D]) -> [B,T,H,D]`.

Both require FP32, contiguous tensors, an even `D`, matching devices, nonnegative position
offset and positive base. CPU reference, HIP, the eager graph and independent PyTorch
autograd all test the same boundary. `--attention-rope-layout-fusion false` keeps the older
materialized graph available for same-binary diagnosis.

The next building block keeps a whole `P×V` result in BTHD. For one fixed head, the value
matrix is still an ordinary `T×D` matrix. Heads are merely interleaved in memory:

```text
address(batch, token, head, column)
  = batch_base + token * (H*D) + head * D + column
```

hipBLASLt expresses this without a copy by setting the matrix leading dimension to `H*D`
and the strided-batch offset to `D`. Each head touches disjoint columns of the same token
rows. `attention_probability_value_bthd(P[B,H,T,T], V[B,T,H,D])` therefore writes
`[B,T,H,D]` directly. For `B>1`, the engine submits one H-head batched GEMM per outer batch,
because the jump from the last head of one batch to the first head of the next is not a
constant `D` stride.

Training uses the same description in both reverse products:

```text
dP [B,H,T,T] = dO[B,T,H,D] × transpose(V[B,T,H,D])
dV [B,T,H,D] = transpose(P[B,H,T,T]) × dO[B,T,H,D]
```

The complete causal-GQA BTHD graph keeps Q/K in head-major order because QK/softmax work
naturally there, while Value/context stay token-major because their neighboring projection
Linears work naturally there. GQA repeats/reduces Q/K heads on dimension 1 and Value heads
on dimension 2. No component pretends the two layouts are identical; each public shape
contract names the order.

The three interleaved hipBLASLt calls expose an optional immutable plan cache. Its exact key
is `{P×V|dP|dV, H, T, D, device}`. Cached objects contain only descriptions and matrix
layouts; caller pointers, Stream, workspace and algorithms are still supplied per call.
The cache is thread-local, observable and clearable. It is disabled by default because its
operator speedup failed the official-model throughput gate. An available mechanism is not
the same as an enabled optimization policy.

## Stable integration boundary

The long-term integration seam is a C-compatible descriptor plus explicit stream
and workspace. C++ convenience APIs may evolve before 1.0; the C ABI is versioned
once bindings are introduced.
