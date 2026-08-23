# Framework next steps

This file records work that was missing or too implicit in the original six-month
roadmap. Passing a smaller smoke test does not complete a later item.

## P0 — real external weights

- [x] independent named state dict;
- [x] strict/non-strict atomic model load;
- [x] F32/BF16/F16 safetensors and sharded index;
- [x] Qwen-style name and transpose mapping seam;
- [x] single-file low-precision streaming into an uninitialized HIP model;
- [ ] multi-shard/index streaming and memory-mapped loading;
- [ ] FP8/INT8/INT4 tensors and quantization metadata;
- [ ] official Qwen checkpoint fixture and tokenizer files.

## P1 — dense Qwen correctness target

First target: one pinned small dense checkpoint, not every Qwen release.

- [x] parse pinned Qwen2.5 `config.json` and reject unsupported family/window/MRoPE fields;
- [x] load official Qwen byte-level vocabulary and merges with English/Chinese ID parity;
- [x] load core Qwen special tokens and basic system/user/assistant chat template;
- [ ] support tool-call/tool-response branches of the instruction chat template;
- [x] add Q/K/V bias parameters, backward, HIP Kernel, and strict weight mapping;
- [ ] add explicit attention head dimension and QK-Norm where required;
- [x] preallocate request-bounded device-native KV cache with stable Storage evidence;
- [x] route graph-free T>=256 prefill Attention through strided-batched hipBLASLt;
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
- [ ] compare every per-layer hidden state with PyTorch;
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

- [ ] registry key includes op, GPU architecture, dtype, shape, strides/layout, mode,
  workspace limit, and library/runtime version;
- [ ] register multiple candidates for every hotspot, not only 2D matmul;
- [ ] correctness gate before timing;
- [ ] warm-up, repeated Event timing, median/percentiles, and end-to-end regression;
- [ ] persistent tuning cache with version invalidation;
- [x] schema-versioned `TraceSession`, scoped activation, and RAII `TraceTimer` C++ API;
- [x] same-weight microLLM/PyTorch tiny-model forward, loss, every named gradient, and
  forward/backward timing runner;
- [x] manifest, raw JSONL, comparison JSON, and Markdown report artifacts;
- [ ] optional Python context manager/decorator after the C++ profiler is stable;
- [ ] rocprof marker correlation and Chrome/Perfetto export.

## P2.5 — production data parallel reducer

- [x] synchronous single-process multi-device DataParallelTrainer baseline;
- [x] equal-local-batch validation, bucketed average all-reduce, identical updates;
- [x] forward/backward, communication, optimizer, total, and rank-difference metrics;
- [ ] one process per GPU communicator initialization;
- [ ] autograd gradient-ready hooks and bucket rebuild by observed readiness;
- [ ] compute-stream Events to communication streams and asynchronous work handles;
- [ ] gradient-as-bucket views and zero-copy optimizer integration;
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
- [ ] decide and implement a weight-only INT8 Tensor/scale contract before any model claim;
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
- [ ] connect OuterRow only to FFN activation inputs and measure precision/cost before any default claim;
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
