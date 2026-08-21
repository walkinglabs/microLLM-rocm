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

## Stable integration boundary

The long-term integration seam is a C-compatible descriptor plus explicit stream
and workspace. C++ convenience APIs may evolve before 1.0; the C ABI is versioned
once bindings are introduced.
