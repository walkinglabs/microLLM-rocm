# microLLM-rocm

[![CPU evidence](https://github.com/walkinglabs/microLLM-rocm/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/walkinglabs/microLLM-rocm/actions/workflows/cpu.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://isocpp.org/)
[![ROCm](https://img.shields.io/badge/backend-ROCm%20%2F%20HIP-ED1C24.svg)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](docs/development/STATUS.md)

An independently usable C++20/HIP runtime for studying, training, profiling, and
extending small decoder-only language models on AMD GPUs.

[Documentation](docs/index.md) · [Build](docs/dev/build.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Tests](docs/dev/testing.md) ·
[Benchmarks](benchmarks/README.md) · [Roadmap](docs/development/NEXT_STEPS.md) ·
[Optimization log](docs/optimization-log/README.md) ·
[Beginner course](https://github.com/walkinglabs/microLLM-rocm/tree/tutorial/beginner-course)

> **Project maturity:** pre-alpha. The repository has measured CPU, MI300X, PyTorch
> CPU-oracle, and two-rank RCCL evidence. It does not yet claim production readiness,
> all-workload PyTorch ROCm parity, Radeon validation, or reference-length training.

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
- MI300X FNUZ FP8 quantize/dequantize, scaled hipBLASLt GEMM, FP32-master Transformer
  training policy, and KV-cache decode;
- single-representation BF16 FFN/Attention projection inference for pinned Qwen/DeepSeek,
  with shared QKV cast, exact-token, memory, throughput and PyTorch BF16 evidence;
- C, Python ctypes, and optional PyTorch dispatcher adapters;
- reproducible benchmarks, rocprofv3 workflows, hipBLASLt, and RCCL experiments.
- a phase-independent exact-size HIP pool with immediate legacy-default-Stream reuse and strict
  permanent disablement for non-default Streams;
- a cross-framework trace runner for operator/layer values and latency comparisons.
- explicit Scalar/Vectorized AdamW experiments with HIP Event micro-benchmarks; Auto remains
  on the model-validated scalar policy.
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

The complete compiler, CMake, ROCm, library, Python, and troubleshooting matrix is in
[Build from source](docs/dev/build.md).

## Measured evidence

Current `main` gates:

| Gate | Result | Scope |
|---|---:|---|
| Full CPU/HIP configuration | 314/314 | 218 CPU-labelled + 96 HIP-labelled gates; 2 intentional environment skips |
| ASan/UBSan CPU | 211/211 | host code, CLI, model/graph, benchmark and evidence schemas |
| MI300X/gfx942 HIP | 96/96 | allocator/stream, graph, BF16/FP8, batched GEMM and model matrix |
| PyTorch-enabled CPU build | 196/196 | dispatcher parity, full graph/model oracle and ordinary CPU suite |
| Two-rank RCCL | 11/11 | collectives, global-batch equivalence, DDP trainer/CLI |
| Registered test files | 37 | machine-audited CTest registration |
| CPU source coverage | 83.9% lines / 66.6% branches | GCC 13.3 + gcovr 8.3; `src/` and `include/` |

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

BF16 Linear training keeps FP32 parameters/gradients/AdamW masters. In the fixed 2-warm-up,
5-step matrix it reaches 138.66 token/s (Qwen) and 74.06 token/s (DeepSeek), or
3.122×/2.583× the matched PyTorch BF16-autocast reference. It is still 8%–9% slower than
microLLM FP32 and has identical peak engine memory, so it is a correctness foundation,
not a completed internal optimization. See [Experiment 037](docs/optimization-log/experiments/037-bf16-fp32-master-training.md).

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
