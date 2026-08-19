# Framework next steps

This file records work that was missing or too implicit in the original six-month
roadmap. Passing a smaller smoke test does not complete a later item.

## P0 — real external weights

- [x] independent named state dict;
- [x] strict/non-strict atomic model load;
- [x] F32/BF16/F16 safetensors and sharded index;
- [x] Qwen-style name and transpose mapping seam;
- [ ] streaming/memory-mapped loading;
- [ ] FP8/INT8/INT4 tensors and quantization metadata;
- [ ] official Qwen checkpoint fixture and tokenizer files.

## P1 — dense Qwen correctness target

First target: one pinned small dense checkpoint, not every Qwen release.

- [ ] parse `config.json` and reject unsupported fields;
- [ ] load official tokenizer vocabulary, merges, special tokens, and chat template;
- [ ] add bias parameters needed by the selected architecture;
- [ ] add explicit attention head dimension and QK-Norm where required;
- [ ] preallocate device-native KV cache;
- [ ] compare tokenizer IDs, per-layer hidden states, logits, and greedy tokens with
  PyTorch on fixed prompts;
- [ ] record peak host/GPU memory, prefill/decode latency, and tokens/s.

## P2 — operator registry and profiler API

- [ ] registry key includes op, GPU architecture, dtype, shape, strides/layout, mode,
  workspace limit, and library/runtime version;
- [ ] register multiple candidates for every hotspot, not only 2D matmul;
- [ ] correctness gate before timing;
- [ ] warm-up, repeated Event timing, median/percentiles, and end-to-end regression;
- [ ] persistent tuning cache with version invalidation;
- [x] schema-versioned `TraceSession`, scoped activation, and RAII `TraceTimer` C++ API;
- [x] same-weight microLLM/PyTorch tiny-model value and timing runner;
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

- [ ] run one pinned DeepSeek-R1-Distill-Qwen checkpoint after its underlying Qwen
  architecture passes;
- [ ] compare official chat template, reasoning output tokens, logits, and memory;
- [ ] publish the name “Distill” explicitly so it is not confused with flagship R1/V3.

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
- [ ] direct PyTorch ROCm oracle environment;
- [ ] Radeon run;
- [ ] four-rank RCCL after shared-memory environment repair;
- [ ] backward-ready bucket overlap;
- [ ] comprehensive PyTorch/llama.cpp benchmarks;
- [ ] learner-independent tutorial dry run on `tutorial/beginner-course`.
