# microLLM-rocm

[![CPU evidence](https://github.com/walkinglabs/microLLM-rocm/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/walkinglabs/microLLM-rocm/actions/workflows/cpu.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://isocpp.org/)
[![ROCm](https://img.shields.io/badge/backend-ROCm%20%2F%20HIP-ED1C24.svg)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](docs/development/STATUS.md)

An independently usable C++20/HIP runtime for studying, training, profiling, and
extending small decoder-only language models on AMD GPUs.

[Documentation](docs/index.md) · [Build](docs/dev/build.md) ·
[CMake package](#use-microllm-from-another-cmake-project) ·
[Architecture](docs/ARCHITECTURE.md) · [Tests](docs/dev/testing.md) ·
[Benchmarks](benchmarks/README.md) · [Roadmap](docs/development/NEXT_STEPS.md) ·
[Optimization log](docs/optimization-log/README.md) ·
[Beginner course](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course)

> **Project maturity:** pre-alpha. The repository has measured CPU, MI300X, PyTorch
> CPU-oracle, and two-rank RCCL evidence. It does not yet claim production readiness,
> all-workload PyTorch ROCm parity, Radeon validation, or reference-length training.

## Status at a glance

| Area | What works now | Honest boundary |
|---|---|---|
| C++ SDK | Installable Config package, `microLLM::microLLM`, narrow component targets, C API | API and ABI may still change before 1.0 |
| Correctness | CPU references, HIP comparisons, PyTorch oracles, graph-gradient and checkpoint gates | Not every dtype, shape, GPU, or ROCm release is validated |
| Inference | Qwen2.5 and DeepSeek-Distill weight loading, prefill, KV-cache decode and generation | This is not yet a general Hugging Face architecture runtime |
| Training | Tiny/Model-S loops, Autograd, AdamW, checkpoint/resume, BF16 experiments | No reference-length Model-S training claim yet |
| AMD GPU | Measured MI300X/gfx942 HIP, hipBLASLt, FP8 experiments and two-rank RCCL | Radeon and four-rank results remain open evidence gaps |
| Optimization | Reproducible operator/model measurements with accepted and rejected experiments | A local kernel win is never reported as an end-to-end win |

Start with [Quick start](#quick-start), consume the installed library through the
[CMake package](#use-microllm-from-another-cmake-project), or read the compact
[evidence status](docs/development/STATUS.md). The long chronology belongs in the
[optimization log](docs/optimization-log/README.md), not in the getting-started path.

<details>
<summary>Latest optimization checkpoints</summary>

> **Current optimization checkpoint:** Experiment 243 closes only the present
> inference local-policy search. The two remaining casts occupy 2.694%/1.841% of
> measured Qwen/DeepSeek Kernel time, so the next inference milestone must use a new
> custom-kernel or graph-wide architecture. See the [generated saturation map](docs/optimization-log/assets/inference-local-saturation.svg).

> **Current training checkpoint:** the current B1T512 BF16 profile measures
> 31.327/71.873 ms of Kernel time for Qwen/DeepSeek; GEMM remains 58.56%/63.43%.
> See the [generated training map](docs/optimization-log/assets/current-training-profile.svg).

> The next measured training candidate is intentionally shape-selective: BF16 gate/up
> weight gradients reach 1.459×/1.890× at operator level, while four query/KV shapes
> regress. The Autograd route is explicit and default-off pending model validation.

> The later 20-step gate rejects that short-run candidate: Qwen falls to 1.0006×
> and both complete parameter-Max gates fail. The Autograd/CLI route and candidate
> runners are removed; the aligned standalone operator and evidence tools remain.

> Allocation attribution is now exact: the rejected route's extra calls are two cached
> cast buffers per weight gradient, with zero backend-allocation or peak-memory delta.
> A workspace API will be considered only after a direct wall/Event cost gate.

> That gate now rejects the workspace: preallocated Qwen/DeepSeek wall ratios are
> 0.986×/0.889×. No workspace type or model route is added.

> The current training local-policy search is now closed after six adjacent measured
> rejections. This does not finish training work; it moves the next milestone to a new
> kernel/graph scale or the production data-parallel reducer.

> The refreshed two-GPU baseline passes RCCL 14/14 and keeps rank parameters identical
> for 20 steps. Its first production fix is to separate the 13.32% host parameter-audit
> residual from training time before implementing real gradient-ready overlap.

> Parameter verification is now separately timed and configurable: the default remains
> every step, while explicit final-step-only measurement is 1.244× faster with all 180
> loss values unchanged. Real overlap still requires a multi-bucket model workload.

> The tiny bucket matrix confirms that prerequisite: its natural path has one bucket,
> while forcing 12 buckets makes communication 3–4× larger. Model-S is the next
> distributed workload before any readiness-overlap claim.

> Model-S now supplies that workload: 25 MiB produces 3 natural buckets and the
> 19.76 ms best step, with exact rank parameters and a documented 54.3 MB peak-memory
> tradeoff. Pack/unpack attribution comes before readiness overlap.

> Attribution now closes exactly: the 3-bucket reducer performs 126 backend
> allocations and 228 D2D copies over 374,068,224 temporary bytes per step.
> In-place averaging is the first persistent-reducer prerequisite.

> In-place bucket averaging now passes the Model-S gate: communication 1.269×,
> total 1.107×, unchanged peak and RCCL 22/22.

> Persistent bucket/unpacked storage then removes all 120 later-step backend
> allocations and improves communication/total by 1.681×/1.285×. It remains
> explicit because live/peak memory rises 124.7/158.0 MB; gradient-as-bucket
> views are the next measured reducer step.

> Gradient-as-bucket views remove all 114 unpack Storage/copies, improve total
> another 1.067× versus persistent-copy (1.367× versus transient), and return
> live bytes to baseline. Peak remains 33.3 MB above transient, so the route is
> still explicit; direct Autograd accumulation is the next falsifiable step.

> Direct Autograd accumulation then removes the remaining 114 pack copies and
> makes communication 2.173× faster than views, but added leaf reductions make
> forward/backward 0.830× and total 0.991×. The model route is rejected; only a
> producer out-kernel may reopen this direction. The failed C++/CLI route is removed.

> The first producer out-kernel now passes independently: four Model-S shapes
> plus tiny are bit-exact, logical allocation falls 1→0, Event improves
> 1.178×–1.873× and wall 1.101×–1.612×. It is admitted only to a scoped
> Autograd first/sole right-leaf gate; no model or DDP route exists yet.

> The scoped Autograd gate then rejects that promotion: all gradients and
> addresses pass and one allocation disappears, but 0/5 shapes clear 1.05
> (Event 0.976×–1.035×, wall 0.991×–1.018×). The Autograd route is removed;
> the independently faster caller-owned operator remains.

> Model-S gradient-ready order now supplies the next architecture-scale evidence:
> 3 processes × 3 steps × 2 ranks all match exact reverse parameter order.
> Natural 25 MiB buckets complete at 57/57, 35/57 and 1/57, so two have a
> structural overlap window. No overlap speedup is claimed before the Event gate.

> Event-driven bucket overlap now passes its scoped gate: total improves 1.0159×
> versus synchronous views and finish wait 2.297× with unchanged view peak and
> exact losses/parameters. It remains explicit because peak is still 33.3 MB over
> transient and single-process sequential-rank backward limits the result.

> One-process-per-GPU bootstrap now passes: three fresh launches produce six
> independent ranks and 18 rank-steps; all 728 parameter values match across
> ranks and stay within 1.19e-7 of CPU global batch. Bad-rank injection terminates
> its RCCL-waiting peer. Bucketization and overlap migration remain next.

> Rank-local synchronous buckets reduce tiny three-step collectives from 36 to
> 3 while preserving every rank/CPU/failure gate. Wall improves only 1.0037×
> because startup dominates, so this is a correctness baseline; ranked Model-S
> natural buckets are required before performance or overlap claims.

> Ranked Model-S then reduces collectives from 57 to 3 with all 15,586,176
> parameter values checked. Reducer median is 1.678× faster, but bucket samples
> span 19.55–158.52 ms (89.3% CV), while training/wall improve only
> 1.0016×/1.0023×. It is a correctness baseline; multi-step cold/steady timing
> must precede persistence or overlap.

> Multi-step timing overturns that cold result: transient buckets are only
> 0.6747× as fast in the steady reducer and 0.8527× over a complete steady step.
> Each step still makes 60 backend allocations over 124.7 MB and 57+57
> pack/unpack copies. The transient performance route is rejected; persistent
> rank-local storage is the next isolated counterfactual.

> Persistent rank buckets remove all 60 later-step backend allocations and
> improve reducer/step by 1.539×/1.250× versus transient. Versus per-parameter,
> the complete step is 1.056× but reducer is 0.933×; current/peak rise by
> 62.34/124.69 MB. The copy route remains explicit while bucket-gradient views
> target the remaining 57 unpack copies and duplicate storage.

> Ranked bucket-gradient views remove all 57 unpack copies, halve plan capacity
> to 62.34 MB, and return current memory to the per-parameter baseline. They
> improve reducer/step 1.120×/1.006× versus persistent copies and yield a
> 1.055× complete step versus per-parameter, while peak remains +62.34 MB.
> Views stay explicit and become the prerequisite for ranked ready overlap.

> Ranked ready overlap cuts optimizer-side finish wait 2.180× without adding
> memory, but hook/Event/enqueue overhead leaves the complete T32 step only
> 1.0052× faster, below the 1.01 gate. The explicit route remains for teaching
> and scale experiments; Model-S T32 reducer-local optimization is closed.

> The separate scale track finds the boundary: T32 remains neutral at 0.9995×,
> while T128 overlap reaches 1.0923× (1.069× even after removing the slowest
> run) with no memory delta. Overlap is retained as an explicit context-selective
> policy for the measured two-MI300X Model-S track, not as a general default.

> Rank0-only checkpoint publication now passes a full tiny interruption test:
> 2+3 resumed and uninterrupted 5-step checkpoints are byte-identical, rank1
> performs zero writes, and injected rank0 failure terminates its waiting peer.
> Model-S checkpoint size and restore cost remain the next reliability gate.

</details>

## Why this project exists

Large frameworks make model development productive, but they hide the ownership,
layout, execution, graph, and synchronization decisions that matter when a result is
wrong or slow. microLLM-rocm keeps those decisions visible while preserving the pieces
needed to run a real training and generation loop:

- explicit Storage/Tensor ownership, shape, stride, dtype, offset, and device;
- readable CPU references and repository-owned HIP kernels;
- an eager reverse-mode graph engine with device-native Transformer backward;
- Decoder-only MHA/GQA, RoPE, RMSNorm, SwiGLU, causal attention, loss, and optimizers;
- named model state and F32/BF16/F16 safetensors loading;
- C, Python ctypes, and optional PyTorch dispatcher adapters;
- installable CMake targets for independent C and C++ applications;
- reproducible CPU, PyTorch, MI300X, profiling, and RCCL evidence.

The framework keeps readable reference code, measured optimized paths, and rejected
experiments separate. The concise status is in [Current status](docs/development/STATUS.md);
the chronological details are in the [optimization log](docs/optimization-log/README.md).

<details>
<summary>Detailed implementation and experiment ledger</summary>

- MI300X FNUZ FP8 quantize/dequantize, scaled hipBLASLt GEMM, FP32-master Transformer
  training policy, and KV-cache decode;
- opt-in FP8 scalar, device Tensor-amax and FFN-only outer-row activation policies with
  explicit native/fallback counters; none is a default precision claim;
- explicit clipped dynamic FP8 Tensor quantization with finite saturation, a compatibility-preserving
  fraction of 1.0, and separate clipped-call counters; model clipping is not enabled by default;
- executed native MI300 evidence for mixed E5M2-FNUZ activations and E4M3-FNUZ weights at the
  operator layer;
- same-revision official-model evidence rejects E5 activation because all eight complete-logit
  Max/RMS metrics worsen by 1.51×–3.43×; the model/CLI policy was removed while the primitive remains;
- host and device-only FP8 weight-amax preparation policies with separate scan/transfer
  evidence; device mode does not copy weight payloads to CPU;
- opt-in device per-output-column FP8 weight preparation with native scalar GEMM plus an
  algebraically equivalent device post-scale; official-model policy remains experimental;
- an O-projection-only counterfactual that leaves Q/K/V scalar to isolate long-context Attention
  effects; it has independent CPU/HIP routing gates and is not a default;
- explicit per-block FP32 counterfactuals inside an FP8 model for precision attribution;
  selected blocks remain single-representation FP32 and are never silently quantized;
- exhaustive one-block leave-one-out finds no safe DeepSeek layer; Qwen layer 9 improves T8
  Max/RMS by 28.7%/33.4% but formal T512 Max/RMS regress 5.3%/36.4%, closing the one-block policy;
- explicit FP8 weight-only, activation-only and both-roundtrip error-attribution modes; all use
  FP32 GEMM, are inference-only diagnostics, and cannot be reported as FP8 speed paths;
- a direct native-FP8/both-roundtrip/FP32 complete-logit runner with rotated process order;
- an external per-weight scalar/output-channel reconstruction audit grouped by Attention, FFN,
  and output head; it selects experiments but never replaces native complete-logit gates;
- single-representation BF16 FFN/Attention projection inference for pinned Qwen/DeepSeek,
  with shared QKV cast, exact-token, memory, throughput and PyTorch BF16 evidence;
- C, Python ctypes, and optional PyTorch dispatcher adapters;
- reproducible benchmarks, rocprofv3 workflows, hipBLASLt, and RCCL experiments.
- an exact matmul tuning key covering dtype, layout/strides, GPU architecture, HIP/driver/
  hipBLASLt versions, inference/training mode and workspace budget, plus transactional persistent
  JSONL save/load with stale-environment filtering;
- thread-local hipBLASLt handles are owned per device index; alternating GPU0/GPU1 FP32/BF16
  GEMMs pass, the RCCL model suite is restored to 11/11, and four T512 single-GPU guards remain
  `0.998×–1.023×` of the previous revision;
- correctness-before-timing matmul autotuning: complete finite Max/RMS gates precede default-Stream
  HIP Event and wall P50/P95; screening never registers a winner without explicit acceptance;
- deterministic block reductions with a post-read barrier; the fix turns repeated fused Attention
  from 20/20 differing outputs to bit-exact while keeping measured T128/B8 training neutral;
- 2D cooperative bias-gradient reduction preserving contiguous column reads; complete-output
  MI300 gates and same-revision T512 training improve Qwen/DeepSeek by 1.222×/1.111× with
  unchanged peak, while rows below the measured 32-row crossover keep Scalar;
- phase-differential training profiling subtracts load+one-step from load+three-step traces;
  it rejects load-only cast-transpose as a false training hotspot and attributes 53.47% of
  current Qwen T512 Kernel time to exact-shape hipBLASLt GEMMs;
- BF16 training solution-index screening over eight exact shapes and 1,536 complete-output
  candidates; isolated medians improve up to 1.189×, but all-shape/selective model policies
  reach only 0.995×–1.020× on Qwen and 1.005×–1.007× on DeepSeek, so no default is retained;
- source-aware Autograd and strided-layout diagnostics identify one Qwen tied embedding/head
  accumulation as 71.2% of added gradient elements; sparse token-row accumulation removes a
  mostly-zero 544 MB Tensor, cuts Qwen peak 8.11%, and keeps throughput neutral-positive;
- exclusive-owner dense-gradient diagnostics find 72/84 real Qwen/DeepSeek in-place candidates;
  the tested primitive removes 144/168 engine allocations over two T512 steps, but leaves every
  add Kernel, backend allocation and peak unchanged, so `1.0042×/0.9952×` keeps the model policy
  default-off and hands future work to graph-wide liveness planning;
- layout-aware Q/K bias + split-half RoPE reads projection `[B,T,H,D]` and writes Attention
  `[B,H,T,D]` directly in forward and reverses the mapping in backward; independent PyTorch
  gradients pass, diagnosed strided-copy bytes fall 60%, and official T512 peaks fall on both
  Qwen and DeepSeek without a throughput regression;
- hipBLASLt interleaved-head `P×V` consumes probabilities `[B,H,T,T]` and value
  `[B,T,H,D]`, then writes context `[B,T,H,D]` directly; the complete-output MI300 matrix
  is bit-exact and improves Qwen/DeepSeek T512 operator Event time by 1.415×/2.200× versus
  two explicit layout materializations;
- complete BTHD causal-GQA Autograd adds matching interleaved `dO×Vᵀ` and `Pᵀ×dO`, keeps
  Value/context token-major through projection forward/backward, and removes the diagnosed
  strided-copy set entirely; same-binary T512 improves Qwen/DeepSeek by 1.0336×/1.0256×
  while saving another 100.4/205.5 MB peak;
- exact interleaved Attention plan-cache statistics and explicit control are available for
  diagnosis, but the default is off: operator wall time improves 1.067×/1.069× on official
  shapes while full Qwen/DeepSeek training reaches only 0.990×/1.001× and fails the 1.01 gate;
- scaled hipBLASLt matmul exposes finite alpha with CPU/PyTorch/HIP parity, while Attention
  alpha fusion remains default-off: it deletes every target scale Kernel but yields mixed
  Qwen/DeepSeek `0.987×/1.011×` throughput and changes the DeepSeek parameter guard;
- paired GQA K/V repeat/reduction primitives preserve BHTD/BTHD contracts and halve the
  repeat-family launches, but remain default-off: profile Kernel time improves while official
  Qwen/DeepSeek T512 reaches only 0.976×/1.008×;
- zero-batch-stride GQA P×V can broadcast V[B,T,KV,D] without an expanded Tensor; complete
  MI300 outputs pass, but the operator is shape-selective (Qwen T512 0.937×, DeepSeek T512
  1.603×), so it remains an explicit primitive pending a width-128 full-backward gate;
- the width-128 full P×V+dP route remains default-off: it removes 112 DeepSeek allocations
  but reaches only 0.997× end to end because removed Value-repeat launches are replaced by
  extra KV-group GEMMs; forward-only broadcast remains the final scoped variant;
- forward-only width-128 broadcast is also default-off (`1.001×` DeepSeek, changed parameter
  guard); universal, full selective and forward-only zero-stride model routes are now closed,
  while their independently tested backend primitives remain available;
- a move-only caller-owned HIP Graph runtime captures, instantiates and replays explicit-Stream
  work with sticky-error recovery; MI300X crosses from slower at 1/8 nodes to
  `1.21×–1.91×` at 32–512 nodes, while dynamic model Storage and implicit Streams explicitly
  block any current Qwen/DeepSeek Graph speed claim;
- caller-owned `matmul_out_` proves current hipBLASLt GEMMs can be captured bit-exact with stable
  addresses, but repeated vendor-only replay is rejected: Qwen reaches at most `1.022×` and
  DeepSeek remains `0.990×` at 32 calls, so model Graph work must capture heterogeneous regions;
- a scoped model-Stream prototype is fully removed after three complete-logit failures
  (worst Max/RMS `3.846/0.931`): routing asynchronous Kernels without extending temporary
  Storage lifetime is unsafe, so deferred release or an activation arena is now prerequisite;
- an explicit fixed-capacity deferred HIP release scope retains destroyed temporary allocations
  until one Stream completion; exact chains improve `2.28×–2.74×` versus per-temporary safe
  synchronization while separately reporting up to 127 blocks / 2,080,768 pending bytes;
- lifetime-safe model Stream routing restores bit-exact Qwen/DeepSeek inference and training,
  but stays explicit and default-off: the 48-process T32/T512 matrix reaches only
  `0.125×–0.862×` inference and `0.235×–0.575×` training while retaining up to 14.5 GiB;
- explicit HIP stream-ordered buffers and memory-pool diagnostics support allocation/free Graph
  nodes, but remain off the Tensor path: eager async reaches only `0.619×–0.709×` and captured
  allocation nodes `0.036×–0.048×` of the deferred control despite exact results;
- a caller-owned HIP activation arena allocates stable backing outside replay and follows a
  two-slot liveness plan; eager chains improve `1.071×–1.768×`, allocation-free Graph replay
  `1.314×–3.066×`, with explicit 9–1,280 replay setup break-even;
- staged full-training capture now rejects dynamic Tensor Storage before it can invalidate the
  HIP Stream; 24/24 FP32/BF16 processes recover cleanly, while captured AdamW replay exposes an
  unchanged host step, so no complete-training Graph performance claim is made;
- an explicit device-owned AdamW step fixes that semantic blocker: FP32/BF16 moments align through
  three Graph replays and an eager step-four transition; 60-process timing keeps only the FP32
  many-small-Tensor candidate (`1.427×/1.436×`) and rejects BF16/universal routing;
- immutable AdamW pointer descriptors then reduce `advance + N updates` to two Graph nodes;
  90-process state gates rescue BF16 64/256-small-Tensor cases to `10.813×/36.929×`, while
  FP32 16×256K remains `0.908×` and real-training gradient addresses remain unproven;
- an 18-process real-backward identity audit resolves that blocker per workload: Qwen BF16
  T8/T512 and DeepSeek T8 retain every gradient address, while DeepSeek T512 replaces 198
  gradients covering 7.108 GB, so immutable Graph reuse is explicitly rejected there;
- graph-ready preflight then rejects all four Qwen/DeepSeek T8/T512 cases before launch: creating
  the required non-default Stream safely disables the default-Stream pool and invalidates every
  prepared gradient snapshot; optimizer-only model Graph claims remain closed;
- an explicit device-wide quiescent handoff starts a new default-Stream allocator phase, while
  every later non-default submission disables reuse again; 24-process preflight rescues Qwen
  T8/T512 and DeepSeek T8 but correctly retains the DeepSeek T512 rejection;
- the final 21-process model gate rejects optimizer-only Graph routing: loss/parameters are exact
  and metadata H2D is removed, yet Qwen/DeepSeek optimizer time is only `0.798×–0.656×` eager and
  two of three complete steps regress; the research primitives remain explicit;
- the first arena-backed heterogeneous FFN region uses official Qwen/DeepSeek FP32 shapes and
  four stable GEMM/SwiGLU nodes; three of four Graph rows improve `1.202×–2.970×`, while
  DeepSeek R32 at `1.005×` keeps routing shape-selective and outside the BF16 model default;
- caller-owned BF16 FFN workspaces preserve direct-output and fallback shapes without hidden
  allocations; 54/54 official-shape processes are bit-exact, eager Arena/Graph each pass five of
  six rows, and DeepSeek R32 Graph at `0.970×` blocks a universal policy pending full-model gates;
- a default-off model BF16 FFN Arena shares one backing across every block and exposes exact cache
  statistics; 60/60 official processes are bit-exact and reduce allocations, but only three of ten
  rows exceed 1.01, rejecting universal routing while leaving a `rows>=512` hypothesis;
- the model-independent `minimum_rows=512` policy passes that follow-up: Qwen/DeepSeek T512 improve
  `1.019×/1.022×`, while eight short cases create no Arena, retain exact baseline allocation/peak
  counters and remain `0.999×–1.005×`; the policy stays explicit pending broader hardware evidence;
- caller-owned shared-cast BF16 Q/K/V is exact and removes another 480/560 measured allocations,
  but complete Qwen/DeepSeek T512 improves only `1.004×/1.005×` while retaining 4.46/7.86 MB;
  model routing is rejected and default-off, and future liveness work requires size/source attribution;
- opt-in allocation source×exact-size diagnostics provide that attribution with disabled-path no-op
  tags; three fresh processes per model identify `attention.core` as 572.5/792.7 MB and
  53.0%/43.6% of Qwen/DeepSeek T512 logical allocation bytes;
- exact Attention core liveness then removes 600/700 model allocations with caller-owned scaled Q,
  reused K/V, probabilities and output, but T512 reaches only `1.004×/1.002×` and peak rises;
  model routing is rejected and the persistent-Storage track is closed in favor of device math;
- exact FP32 Attention solution screening finds 64 common passing indices per QK/PV shape;
  recommended Qwen/DeepSeek T512 operator Event speedups are `1.324×/1.198×` and
  `1.253×/1.114×`, with complete-output Max/RMS ≤`4.47e-7/6.64e-8` before model registration;
- a thread-local exact FP32 solution registry now includes descriptor strides, alpha, workspace,
  architecture and backend versions, then verifies support on first dispatch; 24-process official
  T512 gating is bit-exact with unchanged peak, but QK/PV/both reach at most `1.009×` Qwen and
  `1.004×` DeepSeek, so every solution policy remains explicit and default-off;
- pointer-stable BF16 GroupedGemm submits Q/K/V together through one initialized kernel plus
  per-block device arguments; a 64-candidate search reaches `2.010×/1.692×` operator Event and
  `1.046×/1.030×` steady official T512, but 204–208 ms first setup keeps it explicit/default-off;
- `prewarm_bf16_grouped_qkv(rows)` moves grouped plan construction before serving admission;
  Qwen/DeepSeek prewarm costs 915/886 ms and shortens the first admitted T512 request by
  892/947 ms versus lazy grouped setup, while total startup remains explicitly unchanged;
- a fresh-process cold-start gate rejects hipBLASLt all-kernel preload: Qwen/DeepSeek BF16
  first forward slows 3.417×/3.447× and process wall slows 3.140×/2.938× with unchanged
  engine peak, so only measured exact-shape prewarm remains supported;
- selecting only the exact BF16 gate/up solution improves isolated Event time
  1.059×/1.032×, but cold ratios are 0.990×/0.996× and steady official-model ratios are
  0.973×/1.007×; bit-exact outputs and unchanged peak make this a performance rejection;
- two-operation BF16 GroupedGemm capability passes official T512 gate/up shapes: device-user-
  arguments reach 1.188×/1.155× Event, while per-call reinitialization falls to
  0.823×/0.940×; model routing awaits a pointer-stable FFN Arena gate;
- pointer-stable grouped gate/up is integrated behind an exact explicit registry: Qwen/DeepSeek
  T512 improve 1.0176×/1.0117×, preserve top-1 within the BF16 envelope, add about 10 KB
  peak and remove exactly 24/28 GEMM submissions per forward;
- composing exact grouped QKV and gate/up passes 24-process interaction gates: both versus
  baseline reaches 1.0655×/1.0474× and both versus QKV-only adds 1.0199×/1.0172×,
  with top-1, BF16 precision and peak-memory gates intact;
- cross-shape grouped screening covers rows 256/1024 for both models and both projection
  families: 24 processes pass 64/64 candidates and device-arguments Event ratios span
  1.124×–1.695×; complete B1/B2 model gating remains separate;
- six complete sequence/batch cases then pass: Qwen B1/T256, B1/T1024 and B2/T512 gain
  1.1075×/1.0280×/1.0311× and DeepSeek gains 1.0755×/1.0212×/1.0223×;
  the same node fixes and tests CLI export of every batch logit row;
- post-composition rocprof phase delta confirms GEMM submissions fall 217→145 and
  253→169, GEMM time improves 1.182×/1.099× and total Kernel improves
  1.009×/1.034×; the next candidate must cross a larger Attention/cast/layout boundary;
- source-aware strided diagnostics attribute all remaining 96/112 T512 copies and
  100.7/205.5 MB to Attention: Q/K/V BTHD→BHTD plus context BHTD→BTHD once per
  block, selecting an inference BTHD Attention island instead of a faster copy Kernel;
- the explicit inference BTHD island reuses existing layout-aware RoPE/GQA primitives,
  eliminates those copies completely, remains bit-exact, saves 4/7 MiB peak and improves
  composed Qwen/DeepSeek T512 throughput 1.1146×/1.0936×;
- BTHD then passes B1/T256, B1/T1024 and B2/T512 on both models with bit-exact
  per-row outputs, zero Attention copies, 2–14 MiB lower peak and speedups spanning
  1.0852×–1.1421×; B2 last-row residual copies are reported separately;
- post-BTHD phase profiling confirms strided time is zero and total Kernel improves
  1.169×/1.118×; cast is now 0.519/0.757 ms and selects a BF16-input fused Q/K
  bias+RoPE candidate while V stays unchanged;
- direct grouped BF16 Q/K consumption removes exactly 48/56 conversion launches per
  T512 forward; five-process Qwen/DeepSeek medians improve 1.0224×/1.0238× with
  bit-exact complete logits and unchanged peak, while an earlier three-process DeepSeek
  window at 1.0068× is retained as the reason the policy stays explicit/default-off;
- the direct BF16 Q/K path also passes B1/T256, B1/T1024 and B2/T512 for both
  official models: five-process speedups span 1.0128×–1.0244× with bit-exact
  per-row logits and unchanged peak; a three-process Qwen B2 result at 1.0091×
  remains the published small-signal counterexample;
- an explicit 128-thread causal-softmax primitive passes complete T256/512/1024
  outputs, but only four of six three-process operator rows clear 1.01×; DeepSeek
  T512 reaches 1.0071×, so no model/CLI policy is retained and Auto stays at 256 threads;
- a fused BF16-to-FP32 GQA repeat primitive is exact and speeds small B1 operator
  shapes up to 1.345×, but B2/T512 reaches only 1.004×/0.995× for Qwen/DeepSeek;
  the model route is rejected while the explicit research primitive remains;
- the post-BF16-Q/K saturation audit closes blind inference micro-fusion scans:
  GEMM is 57%/67% of current T512 Kernel time, two consecutive scoped candidates
  fail cross-model/shape gates, and the next Attention step must be MFMA-tiled online math;
- a phase-independent exact-size HIP pool with immediate legacy-default-Stream reuse and strict
  permanent disablement for non-default Streams;
- a cross-framework trace runner for operator/layer values and latency comparisons.
- correctness-before-timing AdamW tuning with exact element/mirror/alignment/environment keys,
  transactional cache and complete parameter/moment/mirror gates; 15 fresh MI300 processes find
  no aligned case above the 1.05 keep gate, so Auto remains on the model-validated Scalar policy;
- a tested descriptor-driven multi-tensor AdamW research primitive reduces Qwen/DeepSeek AdamW
  launches from 870/1,017 to 3 in a three-step profile; Qwen reaches 1.0573× but DeepSeek only
  1.0094× end-to-end, so the ordinary optimizer and CLI do not route through it;
- BF16 QKV and gate/up multi-output Autograd primitives share one activation cast while preserving
  independent FP32-master gradients; three official model policies fail at least one `1.01×` gate,
  so the primitives remain available without changing Transformer or CLI routing;
- a load-subtracted training saturation audit shows GEMM plus AdamW now account for
  72.71%/83.77% of Qwen/DeepSeek Kernel time; local launch/cast fusion is closed and the next
  training milestone must change GEMM, optimizer traffic, or graph-wide lifetime;
- an explicit BF16 AdamW-moment policy halves both optimizer state tensors while keeping FP32
  master weights and gradients; five-process Qwen/DeepSeek T512 training improves
  `1.0226×/1.0356×` and peak memory falls to `0.8329×/0.8084×`; Qwen optimizer reaches only
  `1.0687×`, so this is a documented partial keep and FP32 remains the default;
- a hybrid BF16 AdamW dispatcher merges only tensors up to 1,048,576 elements and leaves larger
  matrices on the independent vector path; five-process Qwen/DeepSeek optimizer reaches
  `1.2404×/1.2631×` and end-to-end training reaches `1.0490×/1.0528×`; a 16M threshold makes
  DeepSeek slower, so the selected boundary is explicit and MI300X-specific;
- a fresh load-subtracted post-hybrid profile closes the AdamW-threshold track: AdamW Kernel time
  improves `1.372×/1.293×`, while GEMM now owns `59.33%/63.81%` of Qwen/DeepSeek training
  Kernel time; the next accepted training change must alter GEMM architecture rather than another
  local optimizer or cast launch;
- an FP32 grouped weight-gradient capability matrix checks QKV and gate/up for both official
  models with direct `N,T` and one-shared-transpose `N,N`; all eight cases expose thousands of
  inventory algorithms but zero supported candidates, so no fragile multi-output Autograd route
  is introduced;
- a packed-gradient counterfactual includes every D2D pack before one large ordinary GEMM and
  exposes the packed result as potential shared Storage; all four official cases are numerically
  aligned but run at `0.835×–0.979×`, closing both grouped and packed weight-gradient composition;
- an explicit rank-2 FP32 solution registry and complete-output gate/up tuner find stable
  `1.077×/1.133×` operator winners and prove 144/168 exact model dispatches, yet official
  end-to-end training is `0.993×/0.996×`; indices remain diagnostic and no default is installed;
- rank-N strided-batched hipBLASLt with last-two-dimension transpose contracts for Attention.
- T≥256 causal GQA backward using batched GEMM for K/V gradients, with short-sequence fallback.
- optional autograd probability saving for T≥256, reported as a long-sequence speed/memory trade-off.
- T≥256 saved Attention forward using batched hipBLASLt for QK/PV; Qwen/DeepSeek context-512
  training improves another 1.091×/1.165× with unchanged measured peak.
- T≥256 saved Attention backward using batched hipBLASLt for dP/dQ/dK/dV; the same
  context-512 matrix improves another 1.201×/1.309× with unchanged measured peak.
- T≥256 causal-softmax forward/backward uses one cooperative block per row; Qwen/DeepSeek
  context-512 training improves another 1.302×/1.196× with unchanged measured peak.
- rows≥256 RMSNorm weight gradients use one cooperative block per hidden column; the same
  training matrix improves another 1.220×/1.125× with unchanged measured peak.
- paired Qwen/DeepSeek inference matrices across context, batch and cache modes, including
  N1/8/32/64 output lengths, KV allocated/active/waste efficiency and explicit unsupported/OOM
  rows; the T2048/B2/N64 gate records Qwen at 1.250× and DeepSeek at 0.868× PyTorch.
- graph-free long prefill reuses public causal GQA and batched hipBLASLt; Qwen/DeepSeek
  T512/T1024 gain 6.7×–16.7× with explicit T1024 memory cost.
- B1 full-sequence prefill populates capacity-strided KV Storage directly; profiled Qwen
  T512 cache preparation improves 275× over explicit token replay.
- last-dimension row-wise GPU argmax keeps batched logits on device; Qwen/DeepSeek B8
  uncached reference decode gains 2.15×/1.68× with unchanged peak and tokens.
- greedy generation without stop tokens writes argmax results into a device history and performs
  one final D2H; N8×3 measured calls fall 24→3 with unchanged bytes and tokens.
- batch-aware full prefill, KV Storage, step store and cached GQA support B1/2/4/8;
  the corrected steady-decode matrix records one real forward per measured token and exposes
  long-context throughput as the current primary inference gap.
- opt-in BF16 KV Storage halves cache bytes with FP32 Attention accumulation; Qwen's
  repeat-prompt 32–2048 gate passes, while retained multi-prompt failures keep FP32 default.
- explicit per-layer FP32/BF16 Cache policies can restore a strict complete-logit gate without
  hiding their extra Cache and long-batch prefill cost.
- a correctness-first multi-request scheduler supports delayed arrival, independent Cache/RNG,
  completion cleanup and CPU/HIP equivalence as the oracle for future slot batching.
- static `generate_batch()` performs real cross-request `[B,T]`/`[B,1]` inference for compatible
  requests, reaching 7.31× serial throughput at HIP B8 with exact row outputs.
- admission bucketing groups pending compatible requests with stable singleton fallback and
  cross-drain arrivals; HIP plateaus near 1,260 token/s when queues split into B4 groups.
- `forward_cached_rows()` consumes unequal per-row positions through shared-Storage B1 views;
  it is a CPU/HIP correctness oracle, while uniform rows keep the original parallel fast path.
- `forward_prefill_cached_row()` admits a new prompt into one empty shared-cache row without
  changing other rows, completing the model-level oracle needed by a future slot scheduler.
- `ContinuousBatchScheduler` now owns fixed shared KV rows, refills completed/cancelled slots,
  preserves per-request RNG/stop state and reports slot/KV/dummy-row efficiency; divergent
  positions remain a measured performance gap rather than a claimed speedup.
- active-row compaction removes inactive dummy model work while preserving fixed slot Storage;
  five divergent Release shapes improve 1.134×–1.348× and reach 0.935×–0.985× serial reference.
- positions-aware RoPE, mapped KV store and per-row-prefix cached Attention batch real divergent
  rows; alternating Release medians improve another 1.295×–1.670× with exact request outputs.
- `--continuous-only true` isolates scheduler profiling with exact transfer/allocation counters;
  its first trace rejected a logits-scatter candidate at 0.993×/0.973× baseline.
- packed `[3,A]` token/position/cache-row metadata halves tiny H2D calls without changing bytes;
  alternating Release throughput improves 1.033×/1.065×.
- stable equal-length admission groups batch prompt prefill into arbitrary shared-cache rows;
  uniform R8/S8 improves 2.931× and reaches 87.4% of static batch throughput.
- the official continuous-serving runner covers Qwen/DeepSeek short and 2048-token contexts,
  2/4 slots, refill, request-bounded BF16 KV bytes and engine peak memory; Qwen is exact in 4/4
  PyTorch cases while DeepSeek has three recorded token mismatches.
- a fixed eight-request 1/2/4/8-slot sweep reports S1-relative efficiency and exact KV/peak bytes;
  it also turned an 18-process full-row recycle failure into 48/48 passing executions while
  preserving a DeepSeek short cross-slot token mismatch.
- opt-in continuous diagnostics report producer path/batch and top-2 margin without changing the
  default timed path; a prefill-only counterfactual isolates one DeepSeek low-margin divergence
  while PyTorch evidence rejects serial prefill as the production default.
- explicit prompt offsets support official B2 row/order/duplicate audits; 12/12 DeepSeek processes
  show identical B2 logits and tokens across row zero/one, refuting a stride or KV-copy defect.
- graph-free inference now supports opt-in layer traces; complete P5 snapshots locate the first
  B1/B2 difference at block 0 and quantify final 151k-logit max-abs/relative-L2 as 0.1530/1.3777%.
- block-zero detail proves Attention norm, Q/K/V, RoPE, context/output, residual and FFN norm are
  exact; the first nonzero value is the fused BF16 FFN output.
- BF16 FFN detail shows cast is exact and gate/up GEMMs independently differ at M32/M64; low-precision
  TraceSession capture now records real values and honest truncation.
- a standalone hipBLASLt inventory finds 53 common solution indices across the M32/M64 DeepSeek
  gate shape without changing default dispatch.
- optional solution 75892 restores all B1/B2 values exactly with 1.3%–3.8% prefill cost; no
  version-local index is hard-coded as default.
- continuous serving reports raw per-request TTFT/completion and P50/P95; long-context S4 minimizes
  median TTFT while S8 maximizes throughput at lower KV utilization.
- explicit length buckets share one model while splitting KV capacity; the first policy keeps total
  slots fixed and exposes routing/memory/latency evidence without claiming unmeasured speedup.
- a benchmark-only rocWMMA BF16 QK tile covers T16–2048 and D64/128 with 48 complete-output
  process gates; it beats the matched library baseline at T512 but falls to 0.688× at T2048 D128,
  admitting an online-Attention prototype while keeping every model route disabled.
- the follow-up online causal-GQA prototype fuses rocWMMA QK/PV with online max/sum and writes
  no global score tensor; 42 fresh processes cover real Qwen/DeepSeek head grids at T32–2048,
  improving the current operator by 1.260×–4.041× while retaining short scalar counterexamples
  and keeping public/model dispatch unchanged pending fallback and full-logit gates.
- `online_causal_gqa_attention_bthd` is now a public BF16-input/FP32-output operator with
  batch-native gfx942 dispatch, explicit tail/width/architecture fallback and auditable counters;
  42 fresh public-API processes pass CPU/PyTorch/HIP outputs, with native cases 1.534×–2.456×
  current while exact fallbacks retain 0.607×–0.696× counterexamples; model use remains disabled.
- the explicit full-model gate correctly hits 168/196 native calls per Qwen/DeepSeek process and
  saves 3.5–57 MiB peak, but every prefill regresses to 0.761×–0.884× and Qwen complete-logit
  Max/RMS reaches 0.511/0.112; the model route is rejected while the public operator remains.
- a rebuttal removes all three per-layer casts by retaining BF16 V and writing BF16 bias/RoPE
  outputs directly; every case improves slightly but remains only 0.777×–0.906× current and Qwen
  Max/RMS remains 0.485/0.110, closing the online-Attention model track without removing its APIs.
- a fresh load-subtracted B1T1024 profile puts hipBLASLt GEMM at 59.7%/66.8% of Qwen/DeepSeek
  Kernel time and causal softmax at 14.8%/9.2%; prior thread tuning has under 0.3% whole-step
  upside, so the next bounded track is exact T1024 QK/PV solution screening.
- T1024 screening finds four local 1.060×–1.538× QK/PV winners, but interleaved BTHD PV has a
  different descriptor (175 misses, zero dispatch); Qwen QK reaches 1.051× with Max/RMS logits
  0.0733/0.0157 while DeepSeek stays exact at only 1.002×, so no index becomes default.
- a tail-safe BF16 SwiGLU vector candidate is bit-identical and 1.249×/1.190× at the exact
  Qwen/DeepSeek operator shapes, but full models improve only 1.007×/1.001×; the explicit API
  stays available while Auto remains scalar.
- hipBLASLt gate-Swish epilogues have 64/64 correct candidates and pointer-stable local speedups
  of 1.097×/1.069×, yet the same-binary model gate is 1.000×/0.991× with changed logits;
  the explicit research switch stays default-off and the local FFN activation track is closed.
- direct BF16 RMSNorm output removes a full FP32 write plus cast while preserving every GPU BF16
  value; exact B1T1024 operator Event speedups are 1.866×/2.070×, with model routing deferred
  to a separate full-logit gate.
- the FFN Arena model route passes separately at 1.0122×/1.0092× with bit-identical complete
  logits, unchanged peak and 120/140 fewer measured allocations; BF16 FFN Arena now enables it by
  default while an explicit false path remains.
- a fresh load-subtracted profile of that default cuts cast calls 96→72 and 112→84 and Kernel
  time to 8.208/14.659 ms; GEMM now occupies 60.9%/68.2%, and the next isolated boundary is
  Attention Norm directly feeding the existing BF16 QKV Arena.
- that Attention route passes at 1.01309×/1.01303× with bit-identical complete logits,
  120/140 fewer measured allocations and 3.67/6.29 MB lower peak; BF16 QKV Arena enables it by
  default while explicit false remains available.
- reprofiled with both defaults, Kernel time is 8.069/14.489 ms and casts fall to 48/56; each
  layer now has exactly one FP32→BF16 and one BF16→FP32 conversion, which must be attributed
  before another route is proposed.
- direct BF16 P×V output is rejected before timing: both interleaved BTHD and zero-stride GQA
  descriptors return backend status 6, so candidate APIs were removed and no model claim exists.
- retaining BF16 V with FP32 probabilities/context is rejected by the same two descriptors;
  both directions of the remaining vendor mixed-dtype cast shortcut are therefore closed.

</details>

The design keeps three implementations where they provide engineering value:

```text
readable CPU reference → readable HIP kernel → measured optimized candidate
```

An optimized candidate must pass the same numerical and shape/error contracts as the
reference. A faster kernel is not accepted as a correctness argument.

## Architecture

```text
Applications / Examples / Benchmarks
                 │
       C++ API / C ABI / Python adapters
                 │
 Tensor ── Operators ── Autograd ── Transformer
   │           │             │          │
Storage     OpContext      Backward   Train / Generate
   │        Stream/Event                   │
   └──────── CPU reference / HIP runtime ──┘
                    │
             hipBLASLt / RCCL
```

Public interfaces live under `include/microllm`; implementation details stay under
`src`. Optional bindings depend on the engine, never the reverse. See the
[repository layout](docs/dev/repository-layout.md) for component ownership and
dependency invariants.

## Quick start

The commands below always follow the same order:

```text
configure -> build -> test
```

`ctest` only runs binaries that already exist. It does not rebuild a library after source
files change, so run the matching `cmake --build --preset ...` command before every test run.

### CPU

Requirements: Linux, CMake 3.25+, a C++20 compiler, and Python 3.9+ for optional tests.
The current evidence was produced with CMake 3.31.10 and GCC/G++ 13.3.0.

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug
```

Run the sanitizer configuration:

```bash
cmake --preset cpu-sanitize
cmake --build --preset cpu-sanitize --parallel
ctest --preset cpu-sanitize
```

### AMD GPU

Install a ROCm release supported by the target GPU, then:

```bash
cmake --preset hip-release
cmake --build --preset hip-release --parallel
ctest --preset hip-release
```

Use an explicit architecture when auto-detection is not appropriate:

```bash
cmake --preset hip-release -DMICROLLM_HIP_ARCHITECTURES=gfx942
```

For RCCL:

```bash
cmake --preset rccl-release
cmake --build --preset rccl-release --parallel
ctest --preset rccl-release
```

### Use microLLM from another CMake project

microLLM installs a relocatable CMake Config package. In plain language, the Config file
is the SDK's address card: another project asks for microLLM, and CMake supplies the
headers, libraries, C++20 requirement, and enabled backend dependencies. Do not copy
source files or hand-write `-I`, `-L`, and `-l` flags.

#### 1. Install microLLM

The small SDK preset omits tests, command-line applications, examples, benchmarks,
Python tests, and PyTorch adapters while retaining the C++ libraries and stable C ABI:

```bash
cmake --preset sdk-cpu
cmake --build --preset sdk-cpu --parallel
cmake --install build/sdk-cpu --prefix "$PWD/install/microllm"
```

For HIP or RCCL, install the already verified `hip-release` or `rccl-release` build
directory instead. Installation never changes which backends were compiled into it.

#### 2. Create a separate C++ consumer

Put these two files in a new directory. `microLLM::microLLM` is the simplest supported
target: it contains the complete single-device C++ training and inference SDK, and its
lower-level dependencies are carried automatically:

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_microLLM_app LANGUAGES CXX)

find_package(microLLM 0.1 CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE microLLM::microLLM)
```

```cpp
#include <iostream>
#include <microllm/base/device.h>
#include <microllm/model/config.h>

int main() {
    const auto device = microllm::Device::cpu();
    const auto config = microllm::model::ModelConfig::model_s();
    std::cout << (device.is_cpu() ? "cpu" : "hip")
              << " parameters=" << config.parameter_count() << '\n';
}
```

#### 3. Configure, build, and run

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/install/microllm
cmake --build build
./build/my_app
```

For the stable plain-C ABI, request `capi` and link the shared-library target:

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_microLLM_c_app LANGUAGES C)

find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS capi)
add_executable(my_c_app main.c)
target_link_libraries(my_c_app PRIVATE microLLM::capi)
```

The C header is `<microllm/capi/microllm.h>`. The C++ component libraries are static;
the C ABI is installed as a versioned shared library. CMake supplies its include path
and runtime link information through the imported target. This path works in a genuinely
C-only CMake project: the consumer does not need to enable C++ or know which C++ libraries
implement the shared ABI.

`CMAKE_PREFIX_PATH` points at the installation root. If a larger environment contains
many packages, `-DmicroLLM_DIR=/prefix/lib/cmake/microLLM` can point directly at the
installed Config directory. For local development, `microLLM_DIR` may instead point at
the configured microLLM build directory. Do not point either variable at the unbuilt
source directory.

For active development, installation is optional. Point `microLLM_DIR` at an already
configured and built microLLM directory:

```bash
cmake -S . -B build \
  -DmicroLLM_DIR=/absolute/path/to/microLLM-rocm/build/cpu-debug
```

Use the build-tree form only while developing both projects together. Use the installed
prefix for deployment, CI artifacts, and sharing an SDK.

Installed targets are:

| Target | Purpose |
|---|---|
| `microLLM::microLLM` | Recommended complete single-device C++ SDK |
| `microLLM::runtime` | Device, Stream, Event and memory runtime |
| `microLLM::core` | Storage, Tensor, dtype and view primitives |
| `microLLM::profiling` | In-process trace API |
| `microLLM::ops` | CPU/HIP operators and optimized dispatch |
| `microLLM::autograd` | Eager reverse-mode graph |
| `microLLM::io` | Tokenizers, datasets and safetensors |
| `microLLM::model` | Decoder-only Transformer |
| `microLLM::training` | Optimizers, checkpoints and Trainer |
| `microLLM::inference` | Generation, KV cache and schedulers |
| `microLLM::capi` | Stable plain-C ABI when built with `MICROLLM_BUILD_CAPI=ON` |
| `microLLM::multi_gpu` | RCCL data-parallel components when built with RCCL |

`microLLMConfig.cmake` exposes `microLLM_VERSION` and its `MAJOR`, `MINOR`, and
`PATCH` fields, plus `microLLM_CXX_STANDARD`, `microLLM_HIP_ARCHITECTURES`,
`microLLM_WITH_HIP`,
`microLLM_WITH_HIPBLASLT`, `microLLM_WITH_ROCWMMA`, `microLLM_WITH_RCCL`, `microLLM_WITH_CAPI`,
`microLLM_WITH_SANITIZERS`, `microLLM_WITH_COVERAGE`, and
`microLLM_AVAILABLE_COMPONENTS` and `microLLM_DEFAULT_TARGET`. It resolves the backend
dependencies recorded by the installed build; a CPU installation does not require ROCm. Mixing libraries from one
build with a config file from another is unsupported, so install the complete prefix
atomically. Before the project reaches 1.0, version compatibility is limited to the
installed `0.1.x` minor line.

CTest has three repository-external consumer gates. `PackageConfig.BuildTreeConsumer`
uses `microLLM_DIR` directly, while `PackageConfig.InstalledConsumer` installs into a
temporary prefix, moves that prefix, and consumes the moved SDK. Both configure,
compile, link, and run a C++ project, a mixed C/C++ project, and a separate project whose
only enabled language is C. They check that internal
compile flags do not leak and that ordinary builds add no link flags; an instrumented
build may carry only its required runtime link option. Both gates also prove that a
missing component and an incompatible pre-1.0 minor version are rejected.
`PackageConfig.PublicExample` separately installs the SDK and builds the short public
example shown above, keeping the beginner path under continuous test.

The copy-paste-ready independent project in
[`examples/package-consumer`](examples/package-consumer) is the smallest supported
consumer. The three package tests execute that public path rather than merely checking
that Config files were copied.

The complete compiler, CMake, ROCm, library, Python, and troubleshooting matrix is in
[Build from source](docs/dev/build.md).

## Measured evidence

Current `main` gates:

| Gate | Result | Scope |
|---|---:|---|
| Full CPU/HIP configuration | 544/544 | ordinary CPU suite plus HIP-labelled conformance; 3 intentional environment-dependent skips |
| CPU Debug | 367/367 | host code, CLI, model/graph, benchmark, all three package paths and evidence schemas |
| ASan/UBSan CPU | 365/365 | host lifetime, external Storage and instrumented-package linking |
| MI300X/gfx942 HIP label | 187/187 | allocator/arena/Stream/Graph, public rocWMMA online Attention, BF16 RMSNorm/SwiGLU, grouped/exact vendor solutions, FP8 and model paths |
| PyTorch-enabled CPU build | 319/319 | dispatcher parity, 32-step BF16 optimizer state, full graph/model oracle and all package paths |
| Multi-GPU/RCCL | 49/49 | ranked overlap/checkpoint ownership/equivalence/failure, package and evidence gates |
| Registered test files | 125 | machine-audited native/Python test sources; package consumers run inside the integration gate |
| CMake Config package | CPU + HIP + RCCL pass | build tree, relocated install tree and public example; external `find_package`, components, compile, link and run |
| CPU source coverage | 78.4% lines / 86.6% functions / 59.1% branches | 8,878/11,329 lines; quiescent handoff and other HIP-only branches remain visible; GCC 13.3 + gcovr 8.3 |

Latest PyTorch-reference maximum absolute differences:

| Domain | Maximum absolute difference |
|---|---:|
| Forward operators | `1.90734863e-06` |
| Autograd graphs | `9.53674316e-07` |
| Tiny Transformer | `1.43051147e-06` |
| SGD/AdamW | `3.72529030e-08` |

These results cover the declared FP32 domain and representative shapes, not every
dtype, model size, context length, or GPU. Detailed gates are maintained in
[Testing and evidence](docs/dev/testing.md) and
[current status](docs/development/STATUS.md).

Latest single-MI300X FP32 model matrix. Built-in rows are CI smoke measurements;
official rows exclude two warm-up iterations and measure five iterations:

| Model | Mode | Measured throughput | Peak engine memory |
|---|---|---:|---:|
| Model-S, 15.6M | train / generate | 1.111 / 1.217 token/s | 238.687 / 59.608 MiB |
| Model-M, 31.3M | train / generate | 0.528 / 1.226 token/s | 478.765 / 119.754 MiB |
| Qwen2.5-0.5B official | train / generate | 24.027 / 18.847 token/s | 8.901 / 2.349 GiB |
| DeepSeek Distill Qwen 1.5B official | train / generate | 13.295 / 10.053 token/s | 26.514 / 6.622 GiB |

These are short functional measurements with random built-in models or fixed official
prompts, not long-context or stable serving claims. “Peak engine memory” excludes
driver/vendor-private allocations. Commands and raw JSONL are documented in
[single-GPU benchmarking](docs/dev/single-gpu-benchmark.md).

Matched Python/PyTorch ROCm comparison on the same MI300X:

| Model | Mode | microLLM | PyTorch | microLLM/PyTorch |
|---|---|---:|---:|---:|
| Model-S | train / generate | 13.57 / 139.22 token/s | 177.57 / 293.55 token/s | 0.076× / 0.474× |
| Model-M | train / generate | 3.51 / 90.57 token/s | 59.94 / 237.60 token/s | 0.059× / 0.381× |
| Qwen2.5-0.5B | train / generate | 24.03 / 18.85 token/s | 51.32 / 70.18 token/s | 0.468× / 0.269× |
| DeepSeek Distill Qwen 1.5B | train / generate | 13.30 / 10.05 token/s | 26.23 / 62.40 token/s | 0.507× / 0.161× |

All comparison rows use matched warm-up/repetition settings and exclude warm-up from
reported throughput. See [Python/PyTorch comparison](docs/dev/pytorch-benchmark.md) for
raw data, memory ratios, implementation differences and limitations.

Current Release steady-decode matrix (microLLM mixed BF16-weight/FP32 paths versus full-model
BF16 PyTorch; warm-up excluded; every measured token executes one post-prefill forward):

| Model | T8 B1 / B8 | T512 B1 / B8 | T2048 B1 / B8 |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 3.029× / 3.366× | 2.598× / 2.511× | 1.499× / 1.012× |
| DeepSeek Distill 1.5B | 2.372× / 2.142× | 1.674× / 1.450× | 0.866× / 0.671× |

Qwen token pairs match all six shapes. DeepSeek matches T8/T512 and retains a T2048
cross-framework divergence, so the long-context rows are performance evidence with an explicit
correctness limit, not a parity claim. At T2048 B8, microLLM/PyTorch peak is 3.58/10.68 GiB for
Qwen and 6.93/13.59 GiB for DeepSeek. Output lengths 1/8/32, KV allocated/active bytes, boundary
contexts and invalid free-first-token evidence are reported in
[Experiment 085](docs/optimization-log/experiments/085-inference-shape-memory-matrix.md). The older
[Experiment 036](docs/optimization-log/experiments/036-bf16-immutable-plan-cache.md)
remains historical short-shape evidence; its 4/4 performance result is superseded by the
corrected steady-decode matrix.

Experiment 087 removes the exact-size allocator's 16-block retirement phase under its strict
legacy-default-Stream-only contract. DeepSeek T2048 B1/B8 alternating medians improve
`1.010×/1.033×`; backend allocations fall to 94 with unchanged peak, KV and tokens. Qwen/DeepSeek
T512 B8 targeted rechecks improve `1.014×/1.099×`. See
[allocator evidence](docs/optimization-log/experiments/087-immediate-default-stream-pool.md).

After Experiment 061 routes graph-free long prefill through batched hipBLASLt, the retained
T512 prefill ratios become `0.308×` (Qwen) and `0.229×` (DeepSeek); T1024 reaches
`0.152×/0.156×`. This is 6.72×–16.73× faster than Experiment 060, while T1024 adds
12%–33% microLLM peak. See
[Experiment 061](docs/optimization-log/experiments/061-batched-long-prefill-inference.md).

Experiment 062 removes prompt token replay. Qwen/DeepSeek T1024 cache preparation is now
71/109 ms and end-to-end four-token generation is 228/351 ms; all token pairs match.
The explicit Qwen T512 token/full profiler control reduces Kernel calls 155× and Kernel
time 112×. See
[Experiment 062](docs/optimization-log/experiments/062-full-sequence-prefill-to-cache.md).

Experiment 063 reduces each batch row on device. Same-card B1/2/4/8 uncached decode gains
1.13×–2.15×; Qwen B8 D2H falls from 38,895,616 to 256 bytes. Cached batch remains a
separate unsupported capability. See
[Experiment 063](docs/optimization-log/experiments/063-device-rowwise-argmax.md).

Experiment 064 closes cached batch `unsupported`: Qwen B1→B8 scales 91.9→721.1 tok/s,
DeepSeek 62.2→494.6 tok/s, with exact paired tokens and explicit FP32-vs-BF16 KV bytes.
See [Experiment 064](docs/optimization-log/experiments/064-batched-kv-cache.md).
Its historical generated-token accounting includes the first token already produced by prefill;
Experiment 085 supersedes it for steady-decode throughput.

Experiment 065 adds explicit FP32/BF16 KV Storage, FP32 accumulation, complete-logit diagnostics
and B2 T4097 fallback coverage. BF16 halves Cache bytes and improves 11/12 Release shapes, but a
retained DeepSeek T512 B1 RMSE failure keeps it opt-in instead of changing the default. See
[Experiment 065](docs/optimization-log/experiments/065-bf16-kv-cache.md).

Experiment 066 tests a one-Kernel BF16 prefix writer. It removes all measured D2D copies and
improves the local profile, but Qwen T2048 B8 repeated cache preparation/end-to-end regress
30.5%/21.1%; the candidate is removed and the failure remains published. See
[Experiment 066](docs/optimization-log/experiments/066-fused-prefix-pair-discard.md).

Experiment 067 adds explicit per-layer Cache dtypes. The pinned DeepSeek strict policy keeps only
layer 1 FP32 on the original prompt: complete-logit gates improve from 11/12 to 12/12 while Cache
remains 1.931× smaller than FP32. See
[Experiment 067](docs/optimization-log/experiments/067-mixed-layer-kv-policy.md).

Experiment 068 retries prefix fusion only for that one FP32 layer. The same binary removes 160 D2D
calls and 167.8 MB, yet prepare/end-to-end regress 1.53%/0.59%; the route is removed. See
[Experiment 068](docs/optimization-log/experiments/068-targeted-prefix-pair-discard.md).

Experiment 069 pairs uniform and strict policies in alternating fresh processes from one binary.
It invalidates the earlier cross-window 13.4% slowdown claim; DeepSeek T2048 B8 same-window E2E is
1.011×. See [Experiment 069](docs/optimization-log/experiments/069-same-binary-kv-policy.md).

Experiment 070 challenges the one-layer policy with four prompt patterns; it passes only 9/14.
The robust-strict pinned policy uses layers 0–3 FP32, passes 14/14, retains a 1.75× Cache reduction
and stays within about 3% of uniform BF16. See
[Experiment 070](docs/optimization-log/experiments/070-kv-policy-prompt-robustness.md).

Experiment 071 applies the same prompt challenge to Qwen. Constant inputs fail at all tested
contexts; at T2048 only an all-FP32 Cache restores logits and tokens. Uniform BF16 remains explicit,
not universally strict-safe. See
[Experiment 071](docs/optimization-log/experiments/071-qwen-kv-prompt-failure.md).

Experiment 072 establishes delayed multi-request serving semantics. CPU/HIP 1/2/4/8-request outputs
match independent generation; the serial reference deliberately has zero batched-forward calls.
See [Experiment 072](docs/optimization-log/experiments/072-reference-serving-scheduler.md).

Experiment 073 connects compatible requests to one batched KV path. HIP B1→B8 scales
337→2,443 token/s with 90.7% efficiency and exact per-row outputs. It remains static: no delayed
arrival or slot refill. See
[Experiment 073](docs/optimization-log/experiments/073-static-batch-generation.md).

Experiment 074 adds stable admission buckets and singleton fallback. HIP B4 reaches 3.78× serial;
B8/B16 queues split into multiple B4 groups and correctly plateau, exposing the need for token-level
slot refill. See [Experiment 074](docs/optimization-log/experiments/074-admission-batch-scheduler.md).

Experiment 102 runs the real continuous scheduler on pinned Qwen2.5-0.5B and
DeepSeek-R1-Distill-Qwen-1.5B. The 24/24 fresh microLLM processes are deterministic and report
exact KV allocation, active KV, slot use, transfers and peak memory. Qwen matches PyTorch tokens
in 4/4 cases; DeepSeek matches 1/4, so long-context parity remains blocked. See
[Experiment 102](docs/optimization-log/experiments/102-official-continuous-serving.md).

Experiment 103 holds the request set fixed while changing only 1/2/4/8 slots. Its first run found
18 stable full-row refill failures; the lifecycle fix passes the unchanged 48-process matrix.
Short S8 reaches 4.32×/4.69× S1 throughput, while long S8 efficiency falls to about 40% and KV byte
utilization to 46.85%. DeepSeek short still changes one request across slot counts. See
[Experiment 103](docs/optimization-log/experiments/103-fixed-request-slot-sweep.md).

Experiment 104 locates that DeepSeek split at request 5/token 4. S4/S8 swap the same two candidates
at a 0.000669 margin. Serializing only prefill restores S1 logits while keeping B4/B8 decode,
refuting decode batching as the cause; however default B2 matches PyTorch at this request and the
serial control adds an external mismatch, so the optimization remains. See
[Experiment 104](docs/optimization-log/experiments/104-deepseek-prefill-divergence.md).

Experiment 105 places the same DeepSeek P5 prompt in B2 row zero, row one, swapped order and both
duplicate rows. All B2 prefill signatures and complete outputs are identical while B1 remains
different, so the difference does not follow local row, stride or cache-copy order. See
[Experiment 105](docs/optimization-log/experiments/105-b2-prefill-row-audit.md).

Experiment 106 compares every value after embedding, 28 blocks, final norm and the complete output
vocabulary. Embedding and duplicate B2 rows are exact at all stages; drift starts in block 0 and
accumulates through block 27. See
[Experiment 106](docs/optimization-log/experiments/106-prefill-layer-drift.md).

Experiment 107 adds twelve block-zero substage records. Eleven stages through FFN norm are exact;
the fused FFN output is the first difference at max 0.0013504. See
[Experiment 107](docs/optimization-log/experiments/107-block0-drift.md).

Experiment 108 opens the fused FFN. Gate GEMM is the first nonzero stage (max 0.015625), up differs
independently, and SwiGLU/down propagate the drift. See
[Experiment 108](docs/optimization-log/experiments/108-bf16-ffn-drift.md).

Experiment 109 queries 64 M32 and 64 M64 BF16 candidates and finds a 53-index intersection. See
[Experiment 109](docs/optimization-log/experiments/109-bf16-algorithm-inventory.md).

Experiment 110 injects common solution 75892 and eliminates all 48-stage drift. See
[Experiment 110](docs/optimization-log/experiments/110-bf16-same-algorithm.md).

Experiment 113 adds request-level latency across the official S1–S8 matrix. See
[Experiment 113](docs/optimization-log/experiments/113-request-latency.md).

The [length-bucketed KV-cache guide](docs/dev/length-bucketed-kv-cache.zh-CN.md) explains the
memory formula, shared-weight ownership, CLI, tests and current no-work-stealing boundary.
[Experiment 114](docs/optimization-log/experiments/114-length-bucketed-cache.md) records the
official MI300X result: 52.9% less KV backing and lower median TTFT, with a measured 42% throughput
loss and worse completion/tail latency, so the policy remains opt-in.
[Experiment 115](docs/optimization-log/experiments/115-bucket-pareto.md) adds an idle-gated
1/2/4-bucket sweep: two B4 pools form the current balanced point, while one B8 pool remains the
throughput/tail-latency default.
The [continuous arrival guide](docs/dev/continuous-arrivals.zh-CN.md) explains skewed lengths,
logical delayed submission, focus-request P95 and the physical-GPU idle gate in beginner-friendly
terms.
[Experiment 116](docs/optimization-log/experiments/116-traffic-skew.md) proves why this matters:
fixed buckets can improve median TTFT while making queued-request P95 roughly three times worse.
[Experiment 117](docs/optimization-log/experiments/117-compatible-overflow.md) adds an opt-in
compatible overflow rule. It recovers about 13% throughput and 61%–62% TTFT P95 versus fixed
buckets under short-heavy traffic, without claiming uniform-pool parity.
[Experiment 118](docs/optimization-log/experiments/118-slot-ratio-sweep.md) shows that a known
short-heavy workload prefers 6:2 slots while long-heavy prefers 2:6; static auto-selection by
model name is therefore rejected.
[Experiment 119](docs/optimization-log/experiments/119-mi300-precision-roofline.md) replaces
peak-speculation with executed FP32/FP16/BF16/FP8 roofline data: FP8 is slower through 512 and only
1.107x FP32 at 1024, far below MI300X peak utilization.
[Experiment 120](docs/optimization-log/experiments/120-large-precision-roofline.md) extends the
matrix to 2048/4096 with an explicit FP32 GPU-reference boundary; FP8 reaches 477 TFLOPS at 4096,
4.31x FP32 but only 18.25% of its official peak.
[Experiment 121](docs/optimization-log/experiments/121-int8-executed-probe.md) executes raw
hipBLASLt INT8xINT8→INT32 through 4096³ (416 TOPS, exact CPU samples) while explicitly keeping
public Tensor and Transformer INT8 support out of scope.
[Experiment 122](docs/optimization-log/experiments/122-official-fp8-static-scale.md) runs official
Qwen/DeepSeek with single-representation FP8 Linear weights. Residency drops sharply, but every
static-scale precision gate fails, so FP8 remains experimental and opt-in.
[Experiment 140](docs/optimization-log/experiments/140-fp8-selective-block-counterfactual.md)
shows that restoring the highest-cancellation block to FP32 still fails all complete-logit gates.
The retained [error-attribution modes](docs/dev/fp8-error-attribution.zh-CN.md) therefore isolate
weight and activation rounding before another precision policy is proposed.
[Experiment 141](docs/optimization-log/experiments/141-fp8-error-source-isolation.md) finds that
Qwen is weight-error dominated while DeepSeek RMS is activation-error dominated; every isolated
complete-logit gate still fails, so neither one-sided fix is accepted as a cross-model default.
[Experiment 142](docs/optimization-log/experiments/142-fp8-native-vs-roundtrip.md) directly shows
that native FP8 GEMM materially changes the logit vector but does not increase total FP32 RMS by
the fixed 5% gate; replacing it with FP32 GEMM is rejected because both-roundtrip also fails.
[Experiment 143](docs/optimization-log/experiments/143-fp8-output-channel-policy.md) improves
DeepSeek RMS but worsens Qwen and reduces both T512 throughputs by about 13%; the output-channel
operator and opt-in policy stay available, but the cross-model default is rejected.
[Experiment 144](docs/optimization-log/experiments/144-fp8-output-column-native-probe.md) proves
that the installed runtime rejects weight-side outer-vector scaling; the portable probe caches this
result and uses native scalar FP8 GEMM plus a device post-scale without software GEMM fallback.
[Experiment 145](docs/optimization-log/experiments/145-fp8-weight-reconstruction-audit.md) audits
365 official Linear weights and finds less than 1.1% family-level reconstruction improvement; it
selects a DeepSeek output-head-only counterfactual rather than claiming model accuracy.
[Experiment 146](docs/optimization-log/experiments/146-fp8-output-head-only.md) adds a same-revision
device-Tensor control and finds zero Max/RMS change with small overhead; the targeted scope is
rejected, and the initially tempting host-Tensor historical comparison is explicitly invalidated.
[Experiment 147](docs/optimization-log/experiments/147-fp8-attention-only.md) improves seven of
eight Max/RMS metrics and passes both T512 speed gates, but Qwen T512 RMS regresses 8.91%; the scope
remains experimental and is not a cross-model default.
[Experiment 148](docs/optimization-log/experiments/148-fp8-attention-output-only.md) narrows the
scope to O projections: Qwen is unchanged, DeepSeek improves, and both T512 speed gates pass. The
scope is retained as opt-in evidence, while complete FP8 precision remains 0/4.
[Experiment 149](docs/optimization-log/experiments/149-fp8-clipped-pilot-invalid.md) records an
invalid clipped-activation pilot: external GPU contention triggered the strict post/preflight gates,
so zero fraction suites are accepted and the retry must start from scratch.
[Experiment 150](docs/optimization-log/experiments/150-fp8-fraction-pilot-workload-invalid.md)
invalidates a fully executed pilot whose weight minimum did not match the retained O-only policy;
the runner now exposes and tests the 0.005 baseline before a fresh retry.
[Experiment 151](docs/optimization-log/experiments/151-fp8-clipped-coarse-grid.md) validates the
corrected baseline and rejects fractions at or below 0.75; a narrow 0.95/0.9/0.85 refinement remains
before model clipping can be closed.
[Experiment 152](docs/optimization-log/experiments/152-fp8-clipped-fine-grid.md) closes the remaining
0.85–0.95 gap: even a 5% clip more than doubles worst RMS. Model/CLI clipping is removed while the
explicit low-level operator remains available for research.

BF16 Linear training keeps FP32 parameters/gradients/AdamW masters. In the fixed 2-warm-up,
5-step matrix it reaches 138.66 token/s (Qwen) and 74.06 token/s (DeepSeek), or
3.122×/2.583× the matched PyTorch BF16-autocast reference. It is still 8%–9% slower than
microLLM FP32 and has identical peak engine memory, so it is a correctness foundation,
not a completed internal optimization. See [Experiment 037](docs/optimization-log/experiments/037-bf16-fp32-master-training.md).

AdamW can separately store its first and second moments as BF16 with
`--adamw-moment-precision bf16`. This leaves FP32 parameters and gradients unchanged, records the
policy in checkpoint v2, and loads v1 checkpoints as FP32. It is opt-in because Qwen misses the
`1.10×` optimizer stretch target even though both official models pass the end-to-end, memory and
loss gates. See the [beginner design guide](docs/dev/bf16-adamw-moments.zh-CN.md) and
[Experiment 214](docs/optimization-log/experiments/214-bf16-adamw-moments-partial.md).
On HIP, that explicit BF16 policy now uses a measured hybrid Auto dispatcher: tensors up to one
Mi elements share a submission and larger tensors retain the vectorized bandwidth path. Use
`--adamw-bf16-multi-tensor-threshold 0` for the per-tensor counterfactual or a positive value for a
labeled experiment. The retained five-process result and the rejected 16M boundary are in
[Experiment 215](docs/optimization-log/experiments/215-hybrid-bf16-adamw.md).

## External weights

The framework supports independent named state dictionaries, strict/non-strict model
loading, Hugging Face-style name/transpose mapping, and single or sharded safetensors:

For an uninitialized HIP model, the single-file path preflights metadata and streams the
original low-precision payload through bounded staging directly into parameter Storage.
Pinned MI300X measurements are 0.580 s for Qwen2.5-0.5B and a 1.356 s median for DeepSeek
Distill 1.5B; multi-shard/index streaming remains future work.

```cpp
#include <microllm/model/model.h>

microllm::model::TransformerModel model(config);
auto mapping = microllm::model::qwen_style_weight_mapping(config);

microllm::model::LoadWeightsOptions options;
options.strict = true;
options.mapping = std::move(mapping);

model.load_safetensors_index("model.safetensors.index.json", options);
model.to(microllm::Device::hip(0));
```

The mapping API handles names and 2D linear-weight orientation. It does not implement
architecture differences such as QK-Norm, Q/K/V bias, explicit head width, MLA, MoE,
or quantization. See [Weight API](docs/WEIGHTS.md).

## Performance workflow

Run the same model in microLLM and PyTorch, then compare every recorded value, shape,
operator time, layer time, and full forward time:

```bash
python3 tools/alignment/run.py \
  --microllm-binary build/hip-release/apps/microllm_alignment \
  --python /path/to/python-with-pytorch \
  --output /tmp/microllm-alignment \
  --microllm-device hip \
  --pytorch-device cpu \
  --warmup 5 --repetitions 20
```

See [Alignment experiments](docs/dev/alignment.md) for the trace schema, four-pass
measurement design, comparison metrics, artifact manifest, and model-extension process.

Inspect the pinned Qwen2.5-compatible architecture without allocating model weights:

```bash
build/cpu-debug/apps/microllm_hf_inspect \
  --config tests/fixtures/qwen25-0.5b-config.json
```

The official Qwen2.5-0.5B checkpoint now passes complete-logit and greedy KV-cache
comparison on MI300X. See the commands, metrics, remaining chat/BF16 gates, and honest
scope in [the Qwen2.5 development record](docs/development/2026-08-19-qwen25-architecture.md).

```bash
# Repeated Event/wall-clock micro-benchmarks
MICROLLM_BUILD_DIR=build/hip-release \
MICROLLM_BENCH_DEVICE=hip \
./scripts/run_benchmarks.sh

# Explicit optimizer candidate comparison (Auto stays Scalar)
./build/hip-release/benchmarks/microllm_bench_adamw \
  --elements 802816 --mirror true \
  --implementation vectorized --warmup 5 --repetitions 20

# HIP API, kernel, memory, JSON/CSV and Perfetto traces
./scripts/profile_hip.sh /tmp/microllm-trace -- \
  ./build/hip-release/benchmarks/microllm_bench_model \
  --mode train --model tiny --device hip \
  --steps 5 --warmup 1 --batch 1 --context 8 --new-tokens 8
```

The current exact-shape registry covers readable 2D matmul and hipBLASLt. It is not a
general autotuner. The C++ `TraceSession`/`TraceTimer` API is implemented; a Python
`@profile` decorator and asynchronous rocprof range correlation remain future work.
See [Profiling](docs/dev/profiling.md) and
[Operator development](docs/dev/operator-development.md).

## Multi-GPU training

The RCCL build includes a correctness-first `DataParallelTrainer` and CLI:

```bash
./build/rccl-release/apps/microllm_distributed_train \
  --steps 3 --bucket-bytes 4194304 \
  --trace /tmp/microllm-ddp-trace.jsonl
```

It runs rank-local forward/backward, bucketed average all-reduce, identical AdamW
updates, cross-rank parameter checks, and stage-level profiling. The current baseline
synchronizes backward before communication; it does not yet claim gradient-ready
overlap or one-process-per-GPU production semantics. See
[Distributed training](docs/dev/distributed-training.md).

## Repository map

| Path | Responsibility |
|---|---|
| `include/microllm/` | public C++ and C APIs |
| `src/` | runtime, Tensor, operators, autograd, model, IO, train/infer, RCCL |
| `bindings/` | optional C, Python, and PyTorch adapters |
| `apps/` | command-line applications |
| `examples/` | small executable API examples |
| `benchmarks/` | micro/e2e/distributed benchmarks and curated evidence |
| `tests/` | unit, graph, conformance, integration, and coverage gates |
| `docs/` | framework and developer documentation |
| `scripts/` | reproducible build, test, benchmark, and profile workflows |
| `tools/alignment/` | microLLM/PyTorch run orchestration and comparison reports |

## Documentation

- [Documentation index](docs/index.md)
- [Developer guide](docs/dev/index.md)
- [Build system and validated toolchains](docs/dev/build.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Operator contracts](docs/OPERATOR_CONTRACTS.zh-CN.md)
- [Weights and safetensors](docs/WEIGHTS.md)
- [Tensor dtypes and MI300/MI350 precision policy](docs/DTYPES.md)
- [Hugging Face and verified Qwen2.5 workflow](docs/HUGGINGFACE.md)
- [DeepSeek Distill support and flagship boundary](docs/DEEPSEEK.md)
- [Hardware compatibility](docs/COMPATIBILITY.md)
- [Alignment experiments](docs/dev/alignment.md)
- [Distributed training](docs/dev/distributed-training.md)
- [Current evidence status](docs/development/STATUS.md)
- [Roadmap and explicit gaps](docs/development/NEXT_STEPS.md)
- [Chronological development records](docs/development/README.md)
- [Living 0→1 optimization blog and experiment log](docs/optimization-log/README.md)

The course-only N0–N10 curriculum is maintained separately on
[`tutorial/beginner-course`](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course).
That branch contains teaching documents and assignments, not a copy of this engine.

## Contributing

Changes require an explicit contract, a reference, positive and negative tests, and
reproducible evidence. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/TASK_CONTRACT.md](docs/TASK_CONTRACT.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
