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

## Autograd accumulation ownership

A gradient is not automatically safe to modify just because its `Tensor` object is local.
Several graph nodes or views may still share the same `Storage`. The ordinary rule is therefore:

```text
first contribution  -> assign its Tensor
later contribution  -> allocate add result
```

The experimental `add_in_place_` route is narrower. It requires FP32, equal shape/device,
contiguous inputs, and exactly one persistent destination owner. The temporary owner used to
inspect `Storage::use_count()` makes the accepted count two. A source alias or any saved graph
view raises the count and restores allocating add. Partially overlapping public views are
rejected explicitly; exact self-add is safe.

Candidate and executed calls/elements are distinct diagnostics. Experiment 172 proves that
Qwen/DeepSeek really have 72/84 eligible T512 destinations, but also proves that reusing them
does not remove add Kernel or backend-allocation work after the exact-size cache. The production
default is therefore false. This seam is evidence infrastructure for a future graph-wide
liveness planner, not permission to mutate an arbitrary Tensor.

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

hipBLASLt handles are thread-local and keyed by HIP device index. A handle created while GPU 0
is current is never reused for GPU 1. BF16 and Attention plan caches already include the device in
their keys and now receive the matching handle. This ownership rule is required by the in-process
two-rank reference; algorithms and public Tensor contracts remain unchanged.

The AdamW operator is a concrete example: `Scalar` and `Vectorized` are selectable, but
`Auto` stays on Scalar because exact-shape float4 wins did not survive the official-model
gate. Selection policy is evidence, not an alias for the newest Kernel.

## HIP Graph ownership boundary

`runtime::HipGraphExecutable` captures work submitted to one explicit HIP `Stream`,
instantiates it once and replays it on a Stream from the same device. It is move-only because
one executable owns one backend graph handle.

Graph capture stores raw device addresses inside its nodes. Therefore every Tensor referenced by
the callback must be caller-owned and remain alive until the final replay has completed. The
runtime does not retain arbitrary callback locals and does not turn a temporary Tensor into safe
Storage by accident.

```text
caller allocates stable input/output Storage
        ↓
capture explicit-Stream operators
        ↓
instantiate and destroy template graph
        ↓
replay while every address is still alive
```

Synchronous allocation inside capture is an expected error on the tested HIP runtime. The
exception path ends abandoned capture state and clears the sticky backend error so an eager
fallback or later valid capture can use the same Stream. CPU capture, empty callbacks/graphs,
undefined launch and cross-device replay fail explicitly.

This is not yet a model execution mode. Eager model/autograd APIs create temporary Tensor Storage
and do not propagate one `OpContext::stream` through the whole graph. Until a liveness planner or
caller-owned activation region makes those addresses stable, wrapping `model.loss()` in capture
would violate this ownership contract.

`matmul_out_` is the first vendor-library boundary under this rule. It writes a validated
beta-zero matmul into caller Storage, rejects input aliases, and lets a warmed hipBLASLt call be
captured on the explicit Stream. Experiment 174 proves capture compatibility and exact replay,
but rejects repeated-GEMM Graph routing as a speed policy. A useful model region must combine
small Kernels and GEMMs under one lifetime plan; repeating an independent GEMM is not that region.

Experiment 175 proves that Stream propagation cannot precede lifetime propagation. A lexical,
thread-local Stream override correctly routed caller-owned operators but corrupted a tiny model's
complete logits because temporary Tensor owners disappeared while non-default-Stream consumers
were still queued. The ambient API was removed. Synchronizing every destructor is not an accepted
fix; future model execution must defer releases or allocate intermediates from a planned arena.

`runtime::DeferredHipDeallocationScope` is the first explicit lifetime primitive. It intercepts
same-thread, same-device deallocation records without allocating on the destructor path, waits for
one bound Stream at `finish()`, then frees every raw block. It does not choose an operator Stream.
Scopes cannot nest; fixed-capacity overflow synchronizes and flushes safely. Pending raw bytes are
reported separately because logical Tensor peak no longer describes physical residency.

`runtime::ScopedDeferredHipStream` binds that lifetime primitive to otherwise-default operator
and runtime layout-copy routing. An explicit different Stream/device, nesting and CPU construction
fail; the state is thread-local. This is the first model-wide non-default-Stream path that passes
complete inference logits and every training gradient. It remains opt-in infrastructure, not a
model policy: Experiment 177 shows that non-default Stream use disables the current default-
Stream-only exact-size pool, replacing tens of cache misses with thousands of synchronous backend
allocations and retaining up to 15.6 GB of raw temporaries. Graph capture still needs stable,
planned addresses or a same-Stream ordered allocator.

`runtime::StreamOrderedHipBuffer` exposes HIP's Linux Beta Stream Ordered Memory Allocator without
changing `Storage`. Allocation and release are submitted to one explicit Stream; capability and
default-pool used/reserved current/high values are queryable. Allocation/free can be captured as
Graph nodes, but Experiment 179 rejects both eager model-style use and allocation-node replay.
Same-Stream eager chains reuse two addresses yet are slower; captured chains own one address per
allocation node. A future activation arena must allocate stable backing Storage outside replay.

`runtime::HipActivationArena` is that outside-replay backing primitive. It owns one fixed HIP
allocation, returns aligned non-owning slices from a caller-designed liveness plan, and keeps the
base address fixed by forbidding move/copy. The bound Stream completes before destruction. The
runtime validates alignment/capacity; it does not infer Tensor lifetimes. Experiment 180 proves a
two-slot plan can capture `N+1` compute-only nodes and replay faster, while leaving real model
shape/liveness planning as the next boundary.

Arena slices can enter existing Tensor-shaped operator contracts through
`Storage::from_external`. That Storage is explicitly non-owning: its caller must keep the pointer
and queued work alive. `matmul_out_` plus `swiglu_out_` form the first real four-node FFN region.
Experiment 181 proves official FP32 shapes and stable replay, but a DeepSeek short-row counterexample
and the production BF16 path prevent a global model switch.

The BF16 region uses `Bf16FfnWorkspace`: input cast, gate, up, activated and output fallback are
all caller-owned. Exact shapes that support BF16×BF16→FP32 capture five nodes. Runtime-rejected
direct-output shapes capture a sixth BF16→FP32 cast without allocating. Experiment 182 keeps this
contract but rejects universal Graph routing; complete-model eager Arena evidence is still required.

`TransformerModel` can opt into one `Bf16FfnArenaCache` shared by all blocks. Each exact row count
gets one owned backing Storage containing every non-owning workspace slice and FP32 output. Default
Stream order preserves liveness between the residual consumer and the next block overwrite. The
cache is default-off, cleared on device moves, and externally synchronized for concurrent calls.
Experiment 183 rejects universal routing; the statistics/API remain the gate for shape selection.
The same cache accepts a positive `minimum_rows`. Calls below it increment a bypass counter and
execute the original allocation-returning FFN without creating an entry. Experiment 184 retains
512 as an explicit two-model long-prefill policy; it is not an unconditional hardware default.

Shared-cast BF16 Q/K/V has the same optional cache shape: one input, three fallbacks and three FP32
outputs in one backing allocation, shared across blocks and selected by rows. Experiment 185 keeps
the caller-owned operator seam but rejects model routing at 1.004×/1.005×; it remains default-off.

Allocation attribution is a separate opt-in runtime layer. Fixed enum tags avoid diagnostic string
allocation, nested RAII scopes restore the previous tag, and disabled scopes are no-ops after one
thread-local branch. Records aggregate logical requests by source/device/exact bytes; they are not
backend allocator traces. Experiment 186 selects Attention core as the next liveness boundary.

`CausalGqaAttentionWorkspace` makes scaled Q, probabilities and output caller-owned and reuses one
expanded K/V slot after the QK submission. A model cache can share that exact backing across blocks
for selected sequence lengths. Experiment 187 rejects model routing despite exact results and fewer
allocations; the primitive remains, while persistent-Storage optimization is considered saturated.

FP32 Attention solution screening is deliberately outside default dispatch. It recreates exact
row-major batched QK/PV descriptors, intersects passing solution indices across fresh processes and
times only complete-output-correct candidates. Version-local indices require an exact registry and
full-model gate before use.

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

hipBLASLt alpha is also exposed through `matmul_scaled_with_implementation`. The readable
contract is `scale(matmul(...), factor)` and rejects nonfinite factors. Attention may use
the primitive experimentally to fuse `1/sqrt(D)` into QK/dQ/dK, but that policy is disabled
by default: moving scale from each operand to the post-accumulation alpha changes FP32
rounding order and failed the joint official-model gate.

GQA K/V head expansion also has explicit paired forward/backward operators. They compute
one logical `(batch,head-or-kv,token,column)` coordinate and write the K BHTD and V BTHD
layouts together. This preserves outputs but does not merge Storage or remove bytes. The
policy is default-off because halving repeat launches improved profiler totals yet regressed
official Qwen training.

For GQA only, `attention_probability_value_gqa_bthd` can avoid expanded Value Storage by
using a zero matrix-batch stride inside each KV group. The implementation submits one
batched GEMM per `(outer batch, KV head)`. This is intentionally not a default: extra GEMM
submissions lose on width-64 Qwen and MHA, while width-128 DeepSeek wins strongly. A later
graph policy must include matching backward layouts and an explicit width gate.

The matching zero-stride dP primitive is available, but the combined width-128 P×V+dP
policy is disabled after end-to-end rejection. It removes Value expansion in both phases yet
replaces each removed copy with another KV-group GEMM. Keeping the primitives separate lets
the final forward-only policy reuse proven P×V while backward stays on one H-batched GEMM.

That forward-only policy is also disabled after the final model/profile gate. Zero-stride
P×V and dP remain public capability primitives, not dispatch defaults. This distinction
prevents an isolated shape win from silently changing the Transformer graph.

### Exact FP32 vendor-solution boundary

hipBLASLt solution indices are not portable algorithm names. `Fp32MatmulSolutionKey` records the
flattened batch descriptor, all physical input/output dimensions and batch strides, transpose
flags, exact alpha bits, mode, workspace budget, architecture, HIP runtime/driver and hipBLASLt
version. Registration is explicit and thread-local. On the first exact match the engine asks the
installed library to resolve the index and validate descriptor/workspace support; subsequent calls
reuse only that validated algorithm object from a cache additionally separated by device index.
A miss keeps the ordinary default solution.

The registry is infrastructure, not a dispatch policy. Experiment 189 showed both required gates:
the fastest approximate QK candidates accumulated visible complete-model error, while replacement
bit-exact candidates preserved every logit but failed the joint 1.01 end-to-end threshold. The CLI
therefore exposes explicit QK/PV indices for controlled experiments and installs none by default.

### Pointer-stable BF16 grouped QKV

The BF16 QKV Arena exposes one shared input cast buffer, three BF16 fallback buffers and three
FP32 outputs with stable addresses. An optional GroupedGemm plan binds those buffers plus one
block's three persistent weight pointers. Its cache key includes exact dimensions, backend
environment, device, Stream and every bound pointer; different Transformer blocks deliberately
own different plans even though they share Arena output addresses.

The backend does not support direct grouped FP32 output for the official shapes. The grouped plan
writes BF16 and then runs three explicit casts into the existing FP32 boundary. Descriptor setup is
more expensive than three ordinary submissions, so a plan may be used only after its pointer set is
stable. Experiment 190 retains this as an explicit exact-shape capability but rejects the default
because only Qwen, not DeepSeek, clears the end-to-end 1.01 gate.

Experiment 191 expands the exact search and changes the implementation boundary. One initialized
GroupedGemm kernel is shared across blocks; small device user-argument records carry each block's
weight pointers. Both models then pass steady throughput, but first kernel initialization remains
about 204–208 ms. The explicit warmed policy is retained, while one-shot/default inference remains
off until serving can prewarm before request admission.

`TransformerModel::prewarm_bf16_grouped_qkv(rows)` is that explicit lifecycle seam. It uses a
dummy activation to build the same Arena/pointer plans as a real request, reports total/kernel/
argument setup, and remembers completed row counts. Repeating the same row is a no-op; moving the
model or reconfiguring QKV Arena invalidates the model-side prewarm state. This moves work before
admission but does not claim to remove startup work.

Experiment 193 rejects broad library preload as a substitute for this exact lifecycle. Asking
hipBLASLt to preload every kernel slows official first forward by about 3.4× and process wall by
about 3.0× with unchanged engine peak. Architecture therefore keeps preload ownership at the
model/shape boundary; the runtime never changes the process-wide preload environment implicitly.

## Stable integration boundary

The long-term integration seam is a C-compatible descriptor plus explicit stream
and workspace. C++ convenience APIs may evolve before 1.0; the C ABI is versioned
once bindings are introduced.
