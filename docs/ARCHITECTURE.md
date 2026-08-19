# Architecture

For a detailed explanation that starts from arrays and avoids assuming framework
knowledge, read [DESIGN_FOR_BEGINNERS.zh-CN.md](DESIGN_FOR_BEGINNERS.zh-CN.md). Exact
shape, error, tolerance, and PyTorch gates live in
[OPERATOR_CONTRACTS.zh-CN.md](OPERATOR_CONTRACTS.zh-CN.md).

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

## Stable integration boundary

The long-term integration seam is a C-compatible descriptor plus explicit stream
and workspace. C++ convenience APIs may evolve before 1.0; the C ABI is versioned
once bindings are introduced.
