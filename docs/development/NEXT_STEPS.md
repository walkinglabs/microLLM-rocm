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
- [x] add a delayed-arrival multi-request reference scheduler with independent Cache/RNG state;
- [x] batch compatible equal-length requests through one public static generation path;
- [ ] batch compatible active request slots without changing reference request semantics;

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
- [x] broader context 8–2048 and batch 1–8 PyTorch inference matrix;
- [ ] identical-residency dtype matrix and llama.cpp benchmarks;
- [ ] learner-independent tutorial dry run on `tutorial/beginner-course`.
