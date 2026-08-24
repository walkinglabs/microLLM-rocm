# Evidence status

States: `draft`, `implemented`, `smoke-tested`, `reference-trained`, `released`.

| Component | State | Current evidence | Missing gate |
|---|---|---|---|
| CPU configuration | smoke-tested | CPU 313/313; full CPU/HIP 487/487 with 3 conditional skips; HIP label 164/164; ASan/UBSan 311/311 | broader compiler/OS CI matrix |
| CPU code coverage | smoke-tested | 80.3% lines, 89.7% functions, 61.4% branches over `src/` + `include/`; HIP-only tuner/Kernel paths remain visible as CPU gaps | split CPU/HIP reports and add justified thresholds |
| Device/DType | smoke-tested | real FP16/BF16 two-byte CPU/MI300X storage, native cast, views and transfer | remaining low-precision operator families |
| CPU Storage | smoke-tested | sharing/lifetime/zero-byte tests | sanitizer log in CI |
| Tensor metadata/views | smoke-tested | hand values, randomized shapes, bounds | more dtypes |
| HIP view materialization | smoke-tested | gfx942 transposed logical-order copy | rank>8/more dtypes |
| Tensor PPM sample | smoke-tested | executable output/checksum | documented golden value |
| HIP Storage/runtime | smoke-tested | gfx942 allocation, transfer, Stream/Event tests | CI on more GPUs |
| CPU reference operators | smoke-tested | hand/PyTorch oracles plus deterministic rank/edge/shape properties and randomized finite differences | more dtypes |
| HIP readable operators | smoke-tested | FP32 suite plus native FP16/BF16 basic kernels with zero host transfers | remaining low-precision forward/backward families |
| Parallel HIP CrossEntropy | smoke-tested | rows 1/3/32, classes through 151936, PyTorch oracle; CE share 75.7%→0.62%, official train 3.29×/2.29× | additional dtype track and fusion with output head |
| Transpose-aware GEMM | smoke-tested | CPU/readable HIP/hipBLASLt NN/NT/TN/TT in FP32/FP16/BF16; tied graph/PyTorch gradients; score 0.318328→0.479227 | batched transpose flags and descriptor/algorithm cache |
| Parallel HIP RMSNorm | smoke-tested | rows 1/3/32 × widths 16/384/512/896/1536; forward/backward/PyTorch gates; RMSNorm 75.85ms→1.55ms; score 0.479227→0.885816 | low-precision path and fusion |
| MI300X precision capabilities | smoke-tested | 4096 FP8 477 TFLOPS/18.25% peak; raw INT8xINT8→INT32 416 TOPS/15.91% peak with exact CPU samples | public weight-only INT8 Tensor/scale contract, official-model FP8 policy and packed INT4 software path |
| FP8 training/inference | smoke-tested | native E4 path, dynamic activation amax, O-only column weights and full official logits; Exp153 rejects model E5 while retaining mixed-format primitives | four full precision gates still fail; layer calibration and full training curve |
| Qwen2.5-0.5B | smoke-tested | official weights, full-logit oracle and Release steady decode at 1.01x–3.39x PyTorch over T1–2048/B1–8/N1–64 | repeated-process full matrix, tool chat and multi-step SFT |
| DeepSeek-R1-Distill-Qwen-1.5B | smoke-tested | official 339 tensors plus Release steady decode above PyTorch at T1–512/N1–64 | T2048 is 0.868x at B2/N64 and 0.866x/0.671x at prior B1/B8; longer reasoning/SFT |
| Operator context | smoke-tested | explicit Stream ordering and mismatch tests | low-level C descriptor |
| CPU Transformer Autograd | smoke-tested | dedicated graph construction tests, finite differences, PyTorch full-graph gradients | more dtypes |
| HIP Autograd | smoke-tested | CPU/HIP full Transformer gradient comparison; zero host transfers during graph execution | optimized reductions/more dtypes |
| Tied embedding accumulation | smoke-tested | source order, CPU/HIP duplicate-row gates, Qwen peak -8.11%, throughput 1.018× and untied DeepSeek zero routing | more tied model families and longer training trajectory |
| HIP bias gradient | smoke-tested | 78 complete-output process rows; T512 operator 3.21×–3.27× and Qwen/DeepSeek training 1.222×/1.111× at unchanged peak | low-precision gradients and broader GPUs |
| SGD/AdamW | smoke-tested | PyTorch parity, zero-transfer HIP update, exact persistent Scalar/Vectorized registry and 15-process complete-state-before-timing matrix | mixed precision/scaler and a candidate clearing the 1.05 model gate |
| Checkpoint | smoke-tested | atomic complete-state load, corruption, 3-step resume | mixed precision |
| Model-S/Model-M config | smoke-tested | executable exact parameter/byte tests | model layers/training |
| Model-S CPU forward | smoke-tested | 15,586,176 parameters and 8192 finite logits | training/HIP |
| Model-S CPU training | smoke-tested | 3-step loss trajectory, AdamW state, parameter delta | real corpus/HIP |
| Model-S HIP forward | smoke-tested | MI300X/gfx942 8192-logit CPU comparison | multi-token/preallocated cache |
| Tiny HIP training | smoke-tested | 5-step finite loss/grad trajectory on MI300X | device-native AdamW |
| Model-M HIP train step | smoke-tested | 31.3M params, finite backward/update, 518.8MB engine peak | multi-step/real corpus |
| Decoder Transformer structure | smoke-tested | tiny GQA graph topology, PyTorch logits/loss/all-gradient parity, CPU/HIP parity | overfit/full recipe |
| Byte tokenizer/token dataset | smoke-tested | all-byte round-trip and cursor equivalence | real-corpus run |
| BPE/TinyStories source | smoke-tested | BPE round-trip + immutable licensed range + Model-S smoke | full corpus/reference train |
| Weight/state API | smoke-tested | independent state_dict, atomic strict/non-strict load, Qwen-style transpose mapping, official single-file device streaming | multi-shard streaming/model-specific architecture validation |
| safetensors | smoke-tested | F32/BF16/F16, shards/index/corruption/CPU-HIP, Python two-way interop, low-precision single-file streaming | FP8/quantized formats, memory mapping and multi-shard streaming |
| C++ training CLI | smoke-tested | CPU save/resume fixture and Model-S HIP real-text steps | validation/report CLI |
| Model-S TinyStories smoke | smoke-tested | immutable 1MiB train prefix, 10 HIP steps | full train/validation curve |
| Tiny Transformer training | smoke-tested | 40-step overfit and finite gradients | validation split/Model-S |
| SFT response masking | smoke-tested | CPU/HIP ignored targets and tiny response-loss curve | Model-S instruction corpus/run |
| CPU/HIP KV cache | smoke-tested | per-layer dtype policies, unequal-position decode and single-empty-row prefill oracles on CPU/HIP | parallel positions-aware kernels and broader checkpoint portability |
| Device greedy sampling | smoke-tested | scalar/two-stage plus last-dim batched argmax; Qwen B8 D2H 38.9MB→256B with exact tokens | stochastic device top-k/RNG |
| HIP exact-size allocator | smoke-tested | phase-independent immediate reuse on legacy default Stream; 82–94 backend allocations and strict non-default disable | size classes and explicit multi-Stream ownership |
| Fused cached decode Attention | smoke-tested | FP32 MHA/GQA 1/32/128/512 + fallback; repeated-process score 1.752183; long decode up to +57.9% | one-token regression, prefill/backward/BF16 |
| Fused Q/K bias + split-half RoPE | smoke-tested | CPU/HIP/PyTorch forward+backward; 1,120 fewer profiled launches; paired generation +13.7%/+6.6%; score 1.784147 | interleaved/low-precision variants and remaining launch fusion |
| BTHD BF16 Q/K inference boundary | smoke-tested | grouped hit removes exactly 48/56 T512 casts; six B1/T256–1024 and B2/T512 cases are bit-exact at 1.0128x–1.0244x; peak unchanged | Radeon/other Instinct and backend-version matrix; explicit/default-off |
| Causal-softmax thread tuning | implemented | explicit Rows128 passes T256/512/1024 outputs, but only 4/6 operator medians clear 1.01x; DeepSeek T512 1.0071x | model route rejected; future work requires online/fused Attention |
| Fused cached residual + RMSNorm | smoke-tested | pair-output oracle, 532 fewer launches, 512-thread wide path; DeepSeek +9.6%; score 1.845199 | broader width matrix and training graph fusion |
| BF16 GEMM/autograd/inference | smoke-tested | transactional FFN/QKV/O, shared cast, immutable plans, batched T≥256 QK/PV and full logits/tokens | long prefill remains 0.15×–0.31×, cached batch and Radeon remain |
| BF16 FP32-master training | smoke-tested | full STE graph/PyTorch gradients, CPU 20-step loss, HIP zero-transfer, 18 official rows; 3.12×/2.58× PyTorch AMP | 0.918×/0.906× micro FP32 and no peak-memory reduction; continuous islands |
| Token generation | smoke-tested | deterministic sampling, cache-backed bounds and one-copy HIP greedy token history | stochastic device top-k/RNG and trained text report |
| Serving scheduler | smoke-tested | slot-ratio matrix 48/48 token-exact; matched 6:2 short and 2:6 long retain 85%–87% throughput while reducing KV 56%/19% | safe dynamic ratio transition and allocator cost; uniform remains default, overflow opt-in |
| Stable model failure | smoke-tested | low-loss cycle breaks beyond training context | rebuttal experiments |
| PyTorch Custom Ops | smoke-tested on CPU | Torch 2.13 add/multiply via dispatcher | build/run with PyTorch ROCm |
| PyTorch correctness oracle | smoke-tested | PyTorch-enabled build 287/287; Tensor/graph/model/optimizer parity plus package and schema gates | broader direct PyTorch ROCm operator matrix |
| PyTorch ROCm environment | smoke-tested | Torch 2.10.0+rocm7.13 and Transformers 5.8.1 run official Qwen/DeepSeek BF16 training on MI300X with native device discovery | additional Torch/ROCm versions and Radeon |
| C ABI v1 | smoke-tested | pure C CPU/HIP create/copy/ops/error client | zero-copy external views |
| Python ctypes API | smoke-tested | CPU/HIP Tensor/ops/error unittest | packaging/broader ops |
| External TensorView ops | smoke-tested | caller-owned CPU/HIP buffers and Stream | Torch build validation |
| In-process profiling | smoke-tested | TraceSession/TraceTimer, values/operator/layer passes, CPU/HIP tests | async Event completion/rocprof markers/Python decorator |
| Cross-framework alignment | smoke-tested | CPU and MI300X both pass 58/58 forward/loss/all-parameter-gradient checkpoints, plus op/layer/backward timings | Qwen/DeepSeek runners/direct PyTorch ROCm |
| Profiling/autotune | smoke-tested | rocprofv3, exact registries, complete output/state before timing, training phase delta, Autograd source and strided-layout diagnostics | automated model regression and broader trace correlation |
| Micro-benchmark harness | smoke-tested | CPU/HIP Event+wall JSONL and error gate | PyTorch operator timing/more shapes |
| Engine allocation tracker | smoke-tested | CPU/HIP current/peak/total accounting | external allocator integration |
| End-to-end benchmark | smoke-tested | matched Python/PyTorch ROCm official training plus phase-separated prefill and one-forward-per-token steady-decode JSONL | serving concurrency, board-level memory and llama.cpp |
| Single-GPU model matrix | smoke-tested | MI300X tiny/Model-S/Model-M plus official Qwen0.5B/DeepSeek-Distill1.5B train+generate, parameters, time, tokens/s and engine peak bytes | repeated contexts/dtypes and external board-level memory sampling |
| Python/PyTorch ROCm performance matrix | smoke-tested | official Qwen/DeepSeek train plus inference context 1–2048, batch 1–8, output 1–64, KV allocated/active/waste, continuous slot metrics and token gates | DeepSeek continuous 3-case mismatch, identical residency policy, version matrix and llama.cpp |
| Optimization experiment journal | implemented | experiments through 207; raw evidence, rejection gates and generated SVGs validated in CTest | Radeon and production data-parallel tracks |
| rocprofv3 workflow | smoke-tested | kernel/HIP API/memory/full trace generated | release artifact retention |
| hipBLASLt matmul | smoke-tested | FP32/BF16, rank-N strided batches, four transpose contracts, Model-S and T≥256 Attention forward/backward | workspace-aware candidate enumeration/timing |
| BF16 solution tuning | smoke-tested | eight T512 shapes, 24 processes, 1,536 complete-output candidates and two rejected model policies | stable cross-process winner plus both-model 1.05 gate before persistence |
| Matmul tuning registry | smoke-tested | exact persistent key plus complete-output gate, Event/wall P50/P95 and explicit acceptance | solution-index enumeration and automatic model regression |
| RCCL two-GPU baseline | smoke-tested | XGMI average and global-batch parameter equivalence | buckets/4 GPU/failure timing |
| DataParallelTrainer | smoke-tested | 3-step two-rank/global-batch equivalence, rank diff 0, stage trace | one-process-per-GPU/gradient-ready overlap |
| RCCL gradient buckets | smoke-tested | 1MB payload with 64/4/1 bucket matrix | overlap with backward |
| RCCL four-GPU | draft | 3 stable init failures and debug root cause | environment with >87MB /dev/shm |
| RCCL compute overlap | smoke-tested | 3 runs, 30–33% synthetic overlap gain | bucket readiness during real backward |
