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
- [ ] replace max-length-per-slot reservation with length-aware Cache blocks and remeasure long S8;
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
- [ ] add request-level TTFT and P50/P95 latency rather than throughput alone;

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
- [ ] FP8 weight/activation scales and accumulation policy;
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
