# Framework next steps

This file records work that was missing or too implicit in the original six-month
roadmap. Passing a smaller smoke test does not complete a later item.

## P0 — real external weights

- [x] independent named state dict;
- [x] strict/non-strict atomic model load;
- [x] F32/BF16/F16 safetensors and sharded index;
- [x] Qwen-style name and transpose mapping seam;
- [x] single-file low-precision streaming into an uninitialized HIP model;
- [x] multi-shard streaming with whole-set preflight and bounded HIP staging;
- [x] indexed weight-map streaming with declared-shard membership preflight;
- [x] read-only memory-mapped payload visits with portable buffered fallback;
- [x] signed one-byte INT8 Tensor plus scalar F32 weight scale, CPU/HIP
  quantize/dequantize and mixed I8/F32 safetensors interoperability;
- [x] scalar/per-output-column INT8 metadata, device preparation, fused M=1,
  scoped whole-model compute and complete-output official rejection gates;
- [ ] standardized FP8 files and packed INT4 software/storage path;
- [x] add pinned official Qwen/DeepSeek fixture preparation: fixed revisions/licenses,
  complete safetensors header/index validation, exact parameter/Tensor counts,
  config/vocab/merges gates and a shared local manifest; Qwen3 tied payloads carry separate
  stored/runtime counts, and large payloads remain outside Git.

## P1 — dense Qwen correctness target

First target: one pinned small dense checkpoint, not every Qwen release.

- [x] parse pinned Qwen2.5 `config.json` and reject unsupported family/window/MRoPE fields;
- [x] load official Qwen byte-level vocabulary and merges with English/Chinese ID parity;
- [x] load core Qwen special tokens and basic system/user/assistant chat template;
- [x] support tool-call/tool-response branches of the instruction chat template;
- [x] add Q/K/V bias parameters, backward, HIP Kernel, and strict weight mapping;
- [x] add explicit attention head dimension and shared Q/K-Norm with CPU/HIP/PyTorch
  forward, loss, all-gradient, cache and mapping gates; official parser remains separate;
- [x] parse pinned Qwen3-0.6B head_dim/QK-Norm/no-bias config and validate its
  311-Tensor BF16 fixture, separating unique runtime parameters from stored tied payloads;
- [x] verify the byte-identical tied lm_head alias during bounded strict streaming and
  align official Qwen3-0.6B complete logits/greedy tokens with Transformers on MI300X;
- [x] run explicit Qwen3 FFN+Attention BF16 with FP32 QK-Norm; it is closer to the
  shared FP32 oracle than Transformers BF16 and passes matched token/s/residency gates;
- [x] preallocate request-bounded device-native KV cache with stable Storage evidence;
- [x] route graph-free T>=256 prefill Attention through strided-batched hipBLASLt;
- [x] keep inference Attention in BTHD layout and remove all four per-block layout copies;
- [x] let the fused BTHD bias+RoPE Kernel consume exact grouped BF16 Q/K directly,
  deleting 48/56 T512 casts with bit-exact official logits and 1.0224x/1.0238x
  five-process model speedups;
- [x] expand direct BF16 Q/K to B1/T256, B1/T1024 and B2/T512: six
  five-process cases pass at 1.0128x–1.0244x with bit-exact per-row logits;
- [x] test 128-thread causal softmax across T256/512/1024; only 4/6 operator
  rows pass 1.01x and DeepSeek T512 is 1.0071x, so model routing is rejected;
- [ ] revisit softmax only as part of an online/fused Attention design that avoids
  materializing the probability tensor;
- [x] fuse BF16 V conversion with GQA repeat; small B1 improves strongly, but
  only 3/8 operator cases pass 1.05 and both B2/T512 cases fail;
- [ ] remove expanded V only through a broadcast/tiled consumer that passes the
  already-recorded width-128 end-to-end counterexamples;
- [x] close the current inference micro-fusion track after two consecutive
  cross-model/shape rejections and publish perfect-elimination upper bounds;
- [x] prove the gfx942 rocWMMA BF16 QK tile boundary across T16–2048 and D64/128;
  all complete outputs pass, while T2048 D128 preserves a 0.688x library counterexample;
- [x] build a benchmark-only MFMA/rocWMMA online Attention prototype with tiled
  QK/PV, online max/sum, causal masking and real Qwen/DeepSeek GQA grids;
- [x] integrate a public BF16-input/FP32-output operator with batch support,
  explicit tail/architecture fallback, counters, PyTorch and CMake Config gates;
- [x] run explicit Qwen/DeepSeek full-logit, peak-memory and end-to-end model A/B;
  reject the route at 0.761x–0.884x and Qwen Max/RMS drift up to 0.511/0.112;
- [x] make grouped QKV, value bias and fused RoPE produce BF16 directly, eliminate
  all three per-layer casts, and repeat the same model gate;
- [x] reject the direct-BF16 route at 0.777x–0.906x with Qwen Max/RMS up to
  0.485/0.110; close the online-Attention model track rather than tune local knobs;
- [x] reprofile the retained B1T1024 default path with load-subtracted rocprof traces;
  GEMM is 59.7%/66.8%, while old softmax tuning has less than 0.3% whole-step upside;
- [x] screen exact T1024 QK/PV hipBLASLt solutions; all four operator rows win;
- [x] reject the model policy: BTHD PV has a different descriptor, Qwen QK
  reaches 1.051x but fails complete logits, and DeepSeek QK reaches only 1.002x;
- [x] vectorize the next open BF16 SwiGLU kernel and reject Auto after the operator
  reaches 1.249x/1.190x but the two full models reach only 1.007x/1.001x;
- [x] test hipBLASLt gate Swish epilogue with pointer-stable grouped plans; reject
  the model route at 1.000x/0.991x and close the local FFN activation track;
- [x] add bit-identical direct BF16 RMSNorm output; operator Event improves
  1.866x/2.070x and is admitted to a separate FFN model gate;
- [x] connect FFN Norm directly to BF16 Arena; keep as default at 1.0122x/1.0092x,
  exact logits, unchanged peak and 120/140 fewer measured allocations;
- [x] reprofile the new default; casts fall 96→72/112→84 and the next bounded
  target is Attention Norm directly feeding the existing QKV Arena;
- [x] connect Attention Norm directly to QKV Arena; keep by default at
  1.01309x/1.01303x, exact logits, 120/140 fewer allocations and lower peak;
- [x] reprofile both retained Norm fusions; Kernel is 8.069/14.489ms and each
  layer retains exactly one FP32→BF16 plus one BF16→FP32 cast;
- [x] reject direct BF16 P×V output before timing: both interleaved BTHD and
  zero-stride GQA descriptors return backend status 6; remove candidate APIs;
- [x] reject BF16 V with FP32 probabilities/output before timing for the same
  two descriptors; close the current vendor mixed-dtype cast track;
- [x] publish the current local-search saturation audit: remaining cast shares are
  2.694%/1.841%, perfect Kernel-only deletion ceilings are 1.0277x/1.0188x,
  and six adjacent scoped tracks are closed;
- [ ] start the next inference milestone only with a new custom-kernel/graph-wide
  contract or a new backend/hardware matrix; do not reopen local default-policy knobs;
- [x] populate B1 KV cache from one full-sequence prefill instead of token replay;
- [x] support batched cached decode with batch-aware KV Storage;
- [x] add opt-in BF16 KV Storage with FP32 accumulation, complete-logit gates and a retained
  DeepSeek Release RMSE failure; keep FP32 as default;
- [x] add explicit per-layer FP32/BF16 Cache policy and a pinned DeepSeek strict profile that
  restores multi-prompt complete-logit gates without becoming an automatic default;
- [x] retain a Qwen constant-context failure proving that strict callers still need full-FP32
  Cache fallback;
- [x] keep uncached batch greedy row-wise argmax on device;
- [x] compare tokenizer IDs, complete logits, and greedy tokens with PyTorch on fixed prompts;
- [x] compare every per-layer hidden state with PyTorch on real FP32 Qwen/DeepSeek:
  27/31 stages complete, embeddings exact, first nonzero at block 0, final logits
  Max/RMS 8.01e-5/1.01e-5 and 2.48e-5/4.19e-6;
- [x] record engine peak, resident policy, prefill/decode latency, tokens/s and KV utilization;
- [x] require one cached model forward per measured decode token and sweep output lengths
  1/8/32 across short/medium/long contexts and B1/B8;
- [x] add a serving suite with output length 64, B2/B4, explicit KV waste and a paired
  T2048/B2 long-context pilot;
- [x] provide a dedicated 31/32/33, 127/128/129 and 511/512/513 dispatch-boundary
  suite plus odd batch 3;
- [x] remove per-token selected-ID D2H for HIP greedy/no-stop generation after allocator
  stabilization; calls fall 24→3 with neutral alternating performance;
- [x] profile the DeepSeek T2048 Release cached path: cached Attention is about 60% of measured
  decode wall and B8 allocator/cache reuse is the second hotspot;
- [x] remove B8 exact-size allocator phase sensitivity with immediate exact-size reuse under the
  legacy-default-Stream-only contract;
- [ ] add a per-position dot/codegen gate before revisiting BF16 cached-Attention pair loads;
- [x] retry and retain device token-history D2H batching after allocator stabilization;
- [x] test and reject paired BF16 Value-column reads despite bit-exact official logits;
- [ ] design a score-level oracle before a wave/MFMA or online cached-Attention rewrite;
- [x] add a delayed-arrival multi-request reference scheduler with independent Cache/RNG state;
- [x] batch compatible equal-length requests through one public static generation path;
- [x] group pending requests by compatibility with singleton fallback and cross-drain admission;
- [x] execute unequal cache positions through a shared-Storage serial B1 oracle on CPU/HIP;
- [x] prefill one empty shared Cache row from a temporary B1 oracle without changing other rows;
- [x] batch and refill active request slots without changing reference request semantics;
- [x] compact inactive rows instead of advancing and resetting dummy rows;
- [x] batch compatible equal-length row prefills instead of one B1 prefill per admission;
- [ ] evaluate bounded padding or packed prefill for mixed prompt lengths;
- [x] replace active divergent-row serial execution with positions-aware parallel RoPE/store/Attention;
- [ ] keep positions and row mappings device-resident across scheduler steps;
- [x] pack CPU token/position/row metadata into one H2D transfer per active step;
- [x] profile the new positions-aware path before choosing its next Kernel optimization;
- [x] test and reject a batched logits scatter despite correct Kernel routing;
- [x] run official Qwen/DeepSeek continuous serving across short/long context and 2/4 slots,
  with exact request-bounded KV bytes, peak memory and complete generated-token evidence;
- [x] hold one official request set fixed and sweep 1/2/4/8 slots for a fair batch-efficiency curve;
- [x] fix full-row recycled Storage admission after the sweep exposed 18 stable refill failures;
- [x] add a deterministic fixed-capacity length-bucket composition that shares model weights,
  preserves eight total slots and exposes exact routing/cache-byte evidence;
- [x] run the uniform-vs-four-bucket long-S8 matrix on an uncontended MI300X: KV backing -52.91%
  and median TTFT -56% to -57%, but throughput -42% and completion latency +74% to +76%;
- [x] sweep one/two/four length buckets with idle-gated fresh processes; two B4 pools retain about
  86% throughput while reducing KV backing 37.4% and median TTFT about 35%;
- [x] measure skewed lengths and delayed arrivals: fixed B4 buckets improve median TTFT while
  worsening focus P95 about 3x and cutting long-heavy throughput about 43%;
- [x] let short requests overflow into an idle compatible larger bucket: short-heavy throughput
  +13%, TTFT P95 -61% to -62%, completion P95 about -40%, token exact and neutral when unused;
- [x] compare 2:6/4:4/6:2: short-heavy prefers 6:2 with 56% less KV and 85% throughput;
  long-heavy prefers 2:6 with 19% less KV and 87% throughput; all token-exact;
- [ ] prototype a safe dynamic ratio transition only when affected buckets are idle; measure
  reallocation/reserved-memory cost before deciding whether paged Cache is required;
- [x] locate the first DeepSeek token/logit divergence and record source, real batch, top-2 and margin;
- [x] refute decode batching by serializing only prefill while preserving B4/B8 positions-aware decode;
- [x] swap/duplicate B2 prefill local rows and refute row, order, stride and KV-copy defects;
- [x] capture complete B1/B2 logits and all 28 block outputs for fixed P5;
- [x] split block 0 and isolate the first difference to fused BF16 FFN output;
- [x] split fused BF16 FFN and isolate first drift to gate/up BF16 GEMMs;
- [x] inventory M32/M64 hipBLASLt solutions and find 53 common candidates;
- [x] inject common solution 75892: all stages exact at 1.3%–3.8% prefill cost;
- [ ] extend strict/fast solution policy across Qwen/DeepSeek FFN shape families;
- [x] test and reject Qwen solution 75789: neutral speed but nonexact logits;
- [ ] scan remaining Qwen common candidates with a complete-logit gate;
- [x] scan all 56 common Qwen candidates: supported, but none tensor-exact;
- [x] add raw request TTFT/completion plus P50/P95 across short/long S1–S8;

## P2 — operator registry and profiler API

- [x] add a one-command current B1T512 training profiler that pins every retained/rejected
  switch and validates exact optimizer metadata transfers;
- [x] reprofile the current binary: Kernel is 31.327/71.873 ms, GEMM remains
  58.56%/63.43%, and the next training target is still architectural;
- [ ] select a new training GEMM or graph-wide candidate from the current top-k families;
- [x] measure cast-inclusive BF16 weight gradients on six real T512 shapes; admit only
  gate/up at 1.459x/1.890x and preserve four query/KV counterexamples;
- [x] add a CPU/HIP/PyTorch-aligned BF16 weight-gradient API and a default-off,
  gate/up-only Autograd/CLI research switch;
- [x] run the short official-model same-binary gate: 1.0213x/1.0638x, unchanged peak,
  exact 48/56 route counts and passing warm-up/final loss gates;
- [x] run 20-step loss and complete gate/up parameter Max/RMS: Qwen falls to 1.0006x,
  both parameter-Max gates fail and only 1/5 aggregate gates passes;
- [x] remove the rejected Autograd/CLI model route and candidate-only runners while retaining
  the aligned operator, shape matrix, loss export and complete safetensors comparison;
- [x] attribute the rejected candidate's temporary allocations exactly to two BF16 cast buffers
  per route; backend allocation, peak and cached-byte deltas are zero;
- [x] measure allocating versus preallocated wall/Event cost: Qwen/DeepSeek wall is
  0.986x/0.889x, so no public weight-gradient workspace is added;
- [x] publish the current training local-saturation audit: six adjacent tracks closed and
  perfect cast-deletion ceilings are only 1.0332x/1.0277x;
- [ ] move the next experiment to a genuinely new kernel/graph scale or the production
  data-parallel reducer; do not reopen local training policy knobs;

- [x] make the matmul registry key exact over its implicit op identity, GPU architecture,
  dtype, shape, strides/layout, mode, workspace limit, and library/runtime versions;
- [ ] register multiple candidates for every hotspot, not only 2D matmul;
- [x] add exact Scalar/Vectorized AdamW selection over elements, mirror, state alignment,
  mode and HIP environment, with transactional persistence and Scalar fallback;
- [x] gate every matmul candidate with complete finite Max/RMS before timing;
- [x] gate AdamW candidates over complete parameter, first/second moment and BF16 mirror
  state before Event/wall timing; reject unaligned Vectorized input with zero timing;
- [x] attribute gradient-add sources and replace Qwen tied embedding's dense second
  contribution with unique-buffer sparse row accumulation; peak memory falls 8.11%;
- [x] replace serial-row bias gradients with a 32-column/8-row-lane cooperative Kernel,
  keeping Scalar below the measured 32-row crossover and passing two official training gates;
- [x] warm up and repeat HIP Event/wall timing with P50/P95 for passing candidates;
- [ ] automate the model end-to-end regression before accepting a recommendation;
- [x] enumerate 64 hipBLASLt solution indices over eight exact BF16 training shapes with
  complete output before timing; reject all/selective model policies below the 1.05 gate;
- [ ] persist solution indices only after a future policy passes both official models;
- [x] persist exact matmul decisions with schema validation, atomic replacement,
  transactional duplicate/corruption rejection and environment-version invalidation;
- [x] schema-versioned `TraceSession`, scoped activation, and RAII `TraceTimer` C++ API;
- [x] same-weight microLLM/PyTorch tiny-model forward, loss, every named gradient, and
  forward/backward timing runner;
- [x] manifest, raw JSONL, comparison JSON, and Markdown report artifacts;
- [x] optional standard-library Python context manager/decorator with schema JSONL;
- [x] Chrome/Perfetto Trace Event export for Python schema spans;
- [x] default-off ROCTX ranges with rocprof marker trace evidence;
- [x] merge C++ ROCTX, contained HIP launch APIs and exact-ID GPU Kernels into one
  Perfetto timeline; never equate marker and Kernel IDs directly;
- [x] calibrate Python `perf_counter_ns` against rocprof timestamps and merge Python,
  ROCTX and GPU events; three runs keep residual at most 1.340us and correlate 24/24 adds;
- [x] represent default-Stream asynchronous HIP completion with C/Python Events and a
  background observer; three softmax runs are pending at submit with zero device/Stream sync;
- [x] expose explicit C/Python Stream bindings and prove target Event wait leaves an
  independent 64-GEMM Stream pending in 3/3 runs with zero wide synchronization;
- [x] add non-owning native Stream interop and pass 3/3 bidirectional PyTorch ROCm Event
  ordering runs; wrapper never owns the Torch handle and output Max is 2.57e-8;
- [ ] resolve the rocprof/PyTorch duplicate-LLVM-option injection failure before claiming
  a mixed-framework HIP API/Kernel timeline;
- [x] add non-owning C/Python Tensor descriptors and pass 3/3 PyTorch ROCm FP32
  contiguous zero-copy add runs: 144MiB exposed, 0 wrapper bytes copied, output Max 0;
- [x] extend external Tensor dtype to FP16/BF16 and pass 6/6 PyTorch ROCm dtype-run
  multiply/matmul caller-output gates: 180MiB exposed, 0 copy, all Max 0;
- [x] add caller-owned FP32 Softmax/RMSNorm, BF16 RMSNorm output and F32/F16/BF16
  SwiGLU; 63/63 random PyTorch rows pass with explicit Max/RMS and 0 wrapper copy;
- [x] add caller-owned FP32 MHA/GQA Attention output/workspace; 15/15 random contexts,
  105/105 pointers and T256 workspace pass with output Max 8.35e-7 and 0 copy;
- [x] add caller-owned RoPE/Embedding/CrossEntropy plus explicit loss workspace;
  36/36 random outputs and 108/108 pointers pass with Max ≤9.54e-7 and 0 copy;
- [x] add caller-owned backward for Softmax/RMSNorm/SwiGLU/RoPE/CrossEntropy and
  Embedding accumulation; 114/114 PyTorch gradients and 285/285 pointers pass;
- [x] add explicit leaf-only Autograd gradient-buffer binding; CPU and MI300X tests prove
  address stability, repeated accumulation, zeroing and error boundaries;
- [x] bind every Tiny/Model-S leaf to one external pool: all gradients are exact and all
  addresses stable, but Event is only 0.792×–0.871× and Model-S peak grows; keep the API
  explicit for interoperability and reject it as the default engine training policy;
- [x] build the optional dispatcher against PyTorch ROCm and cover FP32/FP16/BF16,
  current HIP Stream, error contracts, Autograd and Meta/fullgraph; 20/20 measured cases
  are exact with equal peaks, but 0.469×–0.973× Event rejects an elementwise speed claim;
- [x] vectorize typed caller-owned add/multiply behind a complete tail/alignment/PyTorch
  gate: broad FP32 route is rejected; FP16/BF16 ≥4M aligned route improves 1.277×–1.411×
  versus scalar with all 20 cases exact and unchanged peaks;
- [x] move the adapter boundary to fused SwiGLU: 16M forward is 1.142×–1.570× native
  Torch and peak halves across three dtypes, while F+B 0.615×–0.761× is retained as
  a training counterexample;
- [x] isolate FP32 fused SwiGLU backward and reject vector4: 0.946×–1.039× scalar,
  while the retained scalar producer is already 2.07×–2.82× the readable native formula;
- [x] remove `sum()` zero-stride output-gradient materialization with an exact FP32
  scalar-seed contract: 64K/1M Event improves 1.164×/1.081× and peak falls
  99.42%–99.96%; mean/weighted/general gradients keep the ordinary path;
- [x] attribute remaining FP32 F+B gap: manual submission of the same fused producers is
  4.855×–5.271× custom Autograd and 3.859×–4.105× native; close mathematical Kernel tuning;
- [x] reject `torch.compile` capture: compiled/eager 0.584×–0.610×,
  compiled/native 0.462×–0.476×, cold 55.8–1160.3ms; gradient gate passes and loss
  reduction-order delta remains explicit;
- [x] replace the Python SwiGLU callback with C++ Autograd: C++/Python improves
  1.286×–1.475×, FP32 reaches 1.136×–1.144× native with 1,536B peak, and low-precision
  peak returns to native;
- [x] implement typed fused FP16/BF16 SwiGLU backward: 1.257×–1.319× C++ ATen,
  1.048×–1.084× native Torch, equal peak, BF16 exact and FP16 Max2.38e-7;
- [ ] start another adapter optimization only from a new graph/model profile; the
  FP32/FP16/BF16 SwiGLU scoped line is closed;
- [ ] define a typed low-precision SwiGLU backward contract before replacing the readable
  PyTorch fallback formulas;
- [x] implement direct caller-owned FP16/BF16 Softmax with FP32 reduction and no
  Tensor-shaped FP32 temporary; 10/10 PyTorch/pointer gates pass and peak extra is zero;
- [x] replace the serial typed row Kernel above width32 with 64/128/256-thread block
  max/sum; width128/1024 reaches 1.213×–1.252×/1.103×–1.114× Torch and the same
  10 precision/pointer/zero-temporary gates pass;
- [x] cache FP32 exponentials in bounded width2048–8192 block-local LDS without a
  Tensor allocation; width4096 Event/wall improves 1.217×–1.244×/1.193×–1.226×,
  with 2047/2048 and 8192/8193 correctness/resource boundaries;
- [x] test broad wave-level reduction and remove it: FP16 width4096 improves
  1.071×/1.070× Event/wall, but BF16 reaches only 1.050×/1.033× and fails the
  two-dtype 1.05 gate;
- [x] admit an explicit FP16-only wave predicate: width4096 Event/wall improves
  1.077×/1.080×, while compile-time BF16 fallback remains 1.002×/1.004× baseline;
- [x] reject FP16 fast-exp despite precision passing: width4096 Event/wall is only
  1.045×/1.034× versus the retained wave path, below the 1.05 gate;
- [x] measure FP16 cached/wave 128/256/512/1024 threads: width4096 Event is
  12.027/7.567/5.472/5.086μs; 1024 beats 512 by 1.076×/1.061× Event/wall and
  reaches 0.880× PyTorch;
- [x] attribute FP16 width4096 submission: PyTorch/raw/C++/Python Event is
  4.530/4.764/4.815/5.086μs; C++/raw is 1.011×, Python/C++ 1.056× and
  raw/PyTorch 1.052×;
- [x] register C++ PyTorch Softmax for CPU/ROCm/Meta/C++ Autograd; an inference gate
  improves FP16 width4096 by 1.158× and width1024 reaches 1.026×/0.993× native for
  FP16/BF16;
- [x] add caller-owned `Tensor(a!)` Softmax out with exact pointer, inference-only
  gradient rejection and zero peak; width1024 FP16/BF16 reaches 1.116×/1.087× native out;
- [x] admit BF16 cached wave1024: core width4096 Event/wall improves
  1.687×/1.578× and Custom out improves about 1.687×; current native-out ratios are
  FP16/BF16 0.821×/0.804×;
- [ ] stop local thread/exp tuning and select the next target from a model or adapter
  profile; 1024 is the legal workgroup ceiling and both wide dtypes remain partial.
- [x] reject `softmax_out` Autograd fallthrough: FP16/BF16 width4096 changes only
  about 1.008×/0.998×, so the explicit inference-only rejection kernel remains;
- [ ] refresh the full adapter/model profile before extending another Custom Op family.

## P2.5 — production data parallel reducer

- [x] synchronous single-process multi-device DataParallelTrainer baseline;
- [x] equal-local-batch validation, bucketed average all-reduce, identical updates;
- [x] forward/backward, communication, optimizer, total, and rank-difference metrics;
- [x] refresh the current two-GPU baseline: RCCL 14/14, 20-step parameters identical,
  communication 15.28% and unseparated host verification 13.32% of steady total;
- [x] add explicit parameter-check interval and verification timing while preserving the
  default every-step behavior; 180 loss values match and final-step audit reaches 1.244x;
- [x] run a final-step-audited tiny bucket-count matrix: one natural bucket is fastest;
  artificial 12-bucket paths increase communication about 3–4x;
- [x] add a Model-S natural multi-bucket workload: 25MiB/3 buckets is the 19.76ms best
  baseline with exact losses/parameters and a +54.3MB peak tradeoff versus 4MiB;
- [x] attribute Model-S bucket copies and temporaries exactly: 126 backend allocations,
  228 D2D copies and 374,068,224 bytes per 3-bucket step;
- [x] make bucket averaging in-place and keep it after Model-S communication/total improve
  1.269x/1.107x, peak is unchanged and RCCL 22/22 passes;
- [x] build persistent bucket plus unpacked-gradient storage: subsequent communication backend
  allocations fall 120→0 and total improves 1.285x, but live/peak add 124.7/158.0MB, so it
  remains explicit;
- [ ] one process per GPU communicator initialization;
- [ ] autograd gradient-ready hooks and bucket rebuild by observed readiness;
- [ ] compute-stream Events to communication streams and asynchronous work handles;
- [x] replace 114 unpacked-gradient Storage/copies with gradient-as-bucket views: total improves
  1.067x versus persistent-copy and 1.367x versus transient, live matches transient, but peak is
  still +33.3MB, so the policy remains explicit;
- [x] pre-seed Autograd gradients with bucket views: pack/unpack copies reach zero and peak falls
  13.2MB versus views, but added leaf accumulation makes forward/backward 0.830x and total 0.991x,
  so the model route is rejected;
- [x] gate caller-owned rank-2 weight-gradient producer on four Model-S shapes plus tiny:
  complete outputs exact, allocation 1→0, Event 1.178x–1.873x and wall 1.101x–1.612x;
- [x] test scoped Autograd right-leaf producer with safe overwrite/fallback state: one logical
  allocation disappears, but 0/5 shapes pass 1.05 (Event 0.976x–1.035x, wall 0.991x–1.018x),
  so the route and target-state API are removed;
- [x] record Model-S ready order across 3 processes × 3 steps × 2 ranks: all 57-parameter orders
  match and reverse parameter order; natural buckets complete at 57/57, 35/57 and 1/57;
- [x] prototype Event plus asynchronous all-reduce: exact losses/parameters, 3 buckets, zero later
  communication allocations, unchanged view peak and 1.0159x total versus synchronous views;
- [x] bootstrap one process per GPU: 3 fresh launches/6 ranks, 728 values rank-exact and CPU max
  diff 1.19e-7; atomic RCCL ID exchange and injected peer termination pass;
- [x] add synchronous rank-local buckets: tiny collectives/rank 36→3 (12x), rank/CPU/failure
  gates pass; wall only 1.0037x because startup dominates, so it is a correctness baseline;
- [ ] run ranked Model-S B1T32 one-step with per-parameter 57 vs natural 25MiB/3-bucket paths,
  full parameter equivalence and process/peak/collective records before persistent migration;
- [ ] unused parameter, uneven input, timeout, and cross-process failure handling.

## P3 — DeepSeek distill target

- [x] run one pinned DeepSeek-R1-Distill-Qwen checkpoint after its underlying Qwen
  architecture passes;
- [x] compare official chat template, reasoning output tokens, logits, and memory;
- [x] publish the name “Distill” explicitly so it is not confused with flagship R1/V3.

## P4 — MoE/MLA flagship systems

- [ ] MLA projections, compressed KV cache, decoupled RoPE, and reference tests;
- [ ] top-k expert router, shared/routed experts, grouped routing, and deterministic
  dispatch/combine;
- [x] execute FP32/FP16/BF16/FP8 E4M3-FNUZ across 128–1024 square GEMMs with
  accuracy, Event P95, achieved TFLOPS and MI300 peak/bandwidth roofline;
- [x] add explicit FP32-GPU-reference 2048/4096 roofline: FP8 reaches 477 TFLOPS,
  4.31x FP32 and 1.42x FP16, but only 18.25% of official FP8 peak;
- [x] execute raw hipBLASLt INT8xINT8→INT32 for 128–4096 with exact CPU samples;
  4096 reaches 416 TOPS and 15.91% official peak;
- [x] implement scalar/per-column weight-only INT8 Tensor/scale, device-only amax,
  fused M=1, transactional model preparation and explicit scopes; fixed official
  Qwen gates reject all/FFN/Attention/QKV/O PTQ routes and close the current line;
- [x] run official Qwen/DeepSeek T8/T512 with single-representation FP8 Linear weights;
  all four static-global-scale precision gates fail despite 35%–46% FP32 residency;
- [x] screen a fixed 4x4 global FP8 scale grid on official T8 complete logits: 34/34
  workers execute, but 0/32 scale pairs pass and Qwen's best activation sits at the upper boundary;
- [x] expand only the activation-scale boundary to 0.1/0.2: both model errors improve,
  0/16 pass, and both best candidates land at the new upper boundary;
- [x] test activation scale 0.4/0.8: DeepSeek's top-equal RMS turns upward, while
  Qwen improves to 0.303 but remains six times over the gate at the upper boundary;
- [x] run Qwen-only 1.6/3.2: 0/8 pass and best RMS 0.217 remains at the boundary;
  stop the cross-model global search without claiming a mathematical refutation;
- [x] implement explicit weight `tensor-amax`, transactional nonfinite rejection, scale-range/
  scan/timing reports and a zero-payload-transfer prepared HIP path;
- [x] rerun 36-worker official T8/T512 matrix: RMS improves 39%–78% from the first
  static point but all four gates still fail; Qwen/Deep preparation costs 2.8/12.2 seconds;
- [x] capture 208 all-layer activation boundaries: fixed scale 0.2 potentially saturates
  16 FFN inputs by as much as 64.2x while ordinary Attention inputs use little of the range;
- [x] implement device-side per-Linear-input Tensor amax/scale with no payload H2D/D2H;
  official RMS improves 63%–81%, but 0/4 gates pass and single-block T512 loses 95% throughput;
- [x] measure 208 T8 Tensor row distributions: FFN median spread is 3.8–4.8x with
  1106x/2076x outliers, while Deep Attention is nearly uniform;
- [ ] define an FFN-only row-scale output-rescaling contract compatible with hipBLASLt;
- [x] confirm the installed hipBLASLt exposes native FP32 outer-vector scale mode and
  derive that user-left row scales map to descriptor B after row-major transpose submission;
- [x] implement/test `Scalar/OuterRow`, row quantize/dequantize and FP32-output GEMM;
  gfx942 returns status 3 for native outer-vector, so cache the capability and use device BF16 fallback;
- [x] connect OuterRow only to FFN gate/up/down, with exact three-Linears-per-layer,
  scale-byte, zero-payload-transfer and explicit fallback-counter gates;
- [x] measure official FFN-only row scale: T512 recovers 14–16x versus full Tensor amax,
  but 0/4 precision gates pass and all 288/336 row calls are BF16 software fallback;
- [x] add device-only weight Tensor amax with zero preparation D2H and explicit scan metadata;
- [x] compare official host/device weight amax: device cuts preparation 82%–83%, removes
  host payload scans and keeps TPS within ±2.3%, but is not bit-exact and FP8 gates still fail;
- [x] replace single-block Tensor amax with at most 1024 partial blocks plus one finalize block;
- [x] remeasure multi-block amax: weight prep gains 24.6x/73.3x and T512 activation
  gains 15.5x/20.6x with unchanged complete-logit errors;
- [x] profile retained T512: dynamic scale/finalize/quantize consumes 2.12/3.11ms,
  40.5%/36.0% of dynamic+GEMM attributable time;
- [x] share one quantized activation across Q/K/V and gate/up; tiny machine gates reduce
  Tensor calls 8→5 and FFN row calls 3→2 with exact outputs;
- [x] verify official T512 calls 168→96 and 197→113; errors are exact and throughput
  improves 12.81%/12.39%, with Deep FP8 reaching 1.028x BF16 but failing precision;
- [x] re-profile shared T512: dynamic time drops 45.6%/43.6% and attributable
  forward time drops 20.5%/17.1%, while GEMM/other calls stay unchanged;
- [x] capture complete FP32/FP8 block outputs: Qwen jumps at block21, Deep hidden
  error grows at block27 and logits amplify it further;
- [x] trace Qwen21/Deep27 details: FFN output is only 1.74%/3.24% relative error,
  but residual addition yields 21.21%/11.50%; gate/up are not the primary explosion;
- [x] prove residual cancellation algebraically: Qwen factor17.02x, Deep4.45x,
  with exact block/error-vector reconstruction;
- [x] implement validated `fp8_fp32_layers` mixed prepared models with zero payload transfer;
- [x] run Qwen21/Deep27 FP32-block counterfactual on complete logits and performance;
- [x] exhaustively screen every single FP32 block at T8: Qwen layer 9 improves both
  metrics, while no DeepSeek layer keeps both Max/RMS non-worse;
- [x] run the selected Qwen layer 9 candidate through repeated T8/T512 gates:
  T8 improves, but T512 Max/RMS regress 5.3%/36.4%, closing the one-block policy;
- [x] test and reject model-level E5 activations: all eight Max/RMS metrics worsen
  1.51x–3.43x while the mixed-format operator primitive remains supported;
- [ ] replace the one-block reduction only if the next accepted numerical policy reuses it;
- [ ] replace global FP8 scales with weight per-tensor and activation per-row/token amax,
  starting from saturation/trace evidence rather than top-token search;
- [ ] expert/tensor/data parallel weight placement and communication;
- [ ] multi-node fault handling and profiler timeline;
- [ ] DeepSeek-V3/R1 checkpoint conversion and official-logit comparison.

## Release gaps outside model architecture

- [ ] complete Model-S train/validation curves and checkpoints;
- [ ] real instruction-corpus SFT report;
- [x] direct Python/PyTorch ROCm performance environment; the retained context-512
  matrix reports native device discovery without the earlier AMDSMI fallback;
- [ ] broaden the direct PyTorch ROCm numerical oracle beyond pinned official-model
  loss/parameter checks to the full per-operator matrix;
- [ ] Radeon run;
- [ ] four-rank RCCL after shared-memory environment repair;
- [ ] backward-ready bucket overlap;
- [x] broader context 8–2048, batch 1–8 and corrected steady-decode PyTorch inference matrix;
- [ ] identical-residency dtype matrix and llama.cpp benchmarks;
- [ ] learner-independent tutorial dry run on `tutorial/beginner-course`.
