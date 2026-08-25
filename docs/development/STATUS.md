# Evidence status

States: `draft`, `implemented`, `smoke-tested`, `reference-trained`, `released`.

| Component | State | Current evidence | Missing gate |
|---|---|---|---|
| Current validation configurations | smoke-tested | CPU 349/349; HIP label 188/188 with 1 conditional skip; ASan/UBSan 347/347; PyTorch-enabled 323/323 after building the installed comparison CLI target | broader compiler/OS/GPU matrix |
| CPU code coverage | smoke-tested | 78.4% lines, 86.6% functions, 59.1% branches over `src/` + `include/`; quiescent handoff and other HIP-only paths remain visible as CPU gaps | split CPU/HIP reports and add justified thresholds |
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
| SGD/AdamW | smoke-tested | Hybrid BF16 state remains current model path; two-node Graph aligns state and removes metadata H2D but model optimizer gate is only 0.656×–0.807× and is rejected; DeepSeek T512 snapshot remains unsafe | loss scaling, other-GPU tuning and reference-length training; optimizer-only Graph track closed |
| Checkpoint | smoke-tested | atomic complete-state load, corruption, 3-step resume, v2 moment policy and tested v1 compatibility | long-run cross-version fixture corpus |
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
| HIP exact-size allocator | smoke-tested | immediate legacy-default reuse; non-default submissions disable; explicit device-wide quiescent handoff safely restores a new default phase and rescues three model/context snapshots | Event-granular retirement and end-to-end handoff cost |
| Fused cached decode Attention | smoke-tested | FP32 MHA/GQA 1/32/128/512 + fallback; repeated-process score 1.752183; long decode up to +57.9% | one-token regression, prefill/backward/BF16 |
| Fused Q/K bias + split-half RoPE | smoke-tested | CPU/HIP/PyTorch forward+backward; 1,120 fewer profiled launches; paired generation +13.7%/+6.6%; score 1.784147 | interleaved/low-precision variants and remaining launch fusion |
| BTHD BF16 Q/K inference boundary | smoke-tested | grouped hit removes exactly 48/56 T512 casts; six B1/T256–1024 and B2/T512 cases are bit-exact at 1.0128x–1.0244x; peak unchanged | Radeon/other Instinct and backend-version matrix; explicit/default-off |
| Causal-softmax thread tuning | implemented | explicit Rows128 passes T256/512/1024 outputs, but only 4/6 operator medians clear 1.01x; DeepSeek T512 1.0071x | model route rejected; future work requires online/fused Attention |
| BF16 V repeat fusion | implemented | exact typed primitive; B1 operator up to 1.345x, but only 3/8 pass 1.05 and B2 is 1.004x/0.995x | model route rejected; expanded-V elimination requires a different consumer |
| Inference micro-fusion saturation | smoke-tested | GEMM is 57%/67%; two consecutive scoped candidates rejected; perfect repeat deletion only 1.046x/1.035x Kernel upper bound | separate MFMA/rocWMMA online Attention milestone |
| Fused residual + RMSNorm | smoke-tested | cached inference pair-output oracle and 512-thread path retained; training Autograd primitive aligns CPU/HIP/PyTorch, but official B1/T512 model route was rejected at 0.9785×/0.9980× | future training fusion must cover a larger branch |
| Shared BF16 projection activation | smoke-tested | gate/up and QKV multi-output primitives align every CPU/HIP/PyTorch output and FP32-master gradient; 56-process model matrix rejects all eager routes | combine with grouped GEMM or graph-level scheduling |
| BF16 GEMM/autograd/inference | smoke-tested | transactional FFN/QKV/O, shared cast, immutable plans, batched T≥256 QK/PV and full logits/tokens | long prefill remains 0.15×–0.31×, cached batch and Radeon remain |
| BF16 FP32-master training | smoke-tested | full STE graph/PyTorch gradients, CPU 20-step loss, HIP zero-transfer, 18 official rows; 3.12×/2.58× PyTorch AMP | 0.918×/0.906× micro FP32 and no peak-memory reduction; continuous islands |
| Token generation | smoke-tested | deterministic sampling, cache-backed bounds and one-copy HIP greedy token history | stochastic device top-k/RNG and trained text report |
| Serving scheduler | smoke-tested | slot-ratio matrix 48/48 token-exact; matched 6:2 short and 2:6 long retain 85%–87% throughput while reducing KV 56%/19% | safe dynamic ratio transition and allocator cost; uniform remains default, overflow opt-in |
| Stable model failure | smoke-tested | low-loss cycle breaks beyond training context | rebuttal experiments |
| PyTorch Custom Ops | smoke-tested on CPU | Torch 2.13 add/multiply via dispatcher | build/run with PyTorch ROCm |
| PyTorch correctness oracle | smoke-tested | PyTorch-enabled build 323/323; Tensor/graph/model/optimizer parity plus package, trajectory and schema gates | broader direct PyTorch ROCm operator matrix |
| PyTorch ROCm environment | smoke-tested | Torch 2.10.0+rocm7.13 and Transformers 5.8.1 run official Qwen/DeepSeek BF16 training on MI300X with native device discovery | additional Torch/ROCm versions and Radeon |
| C ABI v1 | smoke-tested | pure C CPU/HIP create/copy/ops/error client plus build-tree and relocated-install C-only Config consumers | zero-copy external views |
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
| Optimization experiment journal | implemented | experiments through 247; raw evidence, rejection gates and generated SVGs validated in CTest | Radeon and production data-parallel tracks |
| BF16 weight-gradient operator | operator admitted; model route removed | CPU/HIP/PyTorch BF16 math; 18 processes: gate/up 1.459×/1.890×, query/KV 0.718×–0.976×; 20-step model rebuttal retained | standalone op remains explicit; no Autograd precision policy |
| BF16 gate/up weight-gradient model route | rejected and removed | 20-step Qwen/DeepSeek 1.0006×/1.0528×; only 1/5 gates passes; 979,894,272 parameters compared; route/runners removed | retain standalone op; investigate generic training workspace/liveness |
| Current B1T512 training profile | smoke-tested | four current-binary rocprof processes; Kernel 31.327/71.873ms, GEMM 58.56%/63.43%, AdamW 13.22%/18.16%, zero negative call deltas | new training GEMM or graph-wide architecture plus model A/B |
| Current inference local policy search | measured saturation | remaining casts are 2.694%/1.841% of Kernel time; perfect-deletion ceilings are 1.0277×/1.0188×; six adjacent scoped tracks closed | new custom-kernel/graph architecture or a new backend/hardware matrix |
| BF16 V into FP32 P×V | capability rejected | ordinary BTHD and zero-stride GQA both return status 6 before timing | candidate APIs removed; vendor mixed-dtype cast track closed |
| Direct BF16 P×V output | capability rejected | ordinary BTHD and zero-stride GQA both return hipBLASLt status 6 before timing; retained FP32 paths pass | candidate APIs removed; requires a different kernel/consumer |
| Post-Attention-Norm profile | smoke-tested | four load-subtracted processes; Kernel 8.069/14.489ms, casts 48/56, GEMM 61.5%/68.8%, both Norm defaults true | attribute remaining FP32→BF16 and BF16→FP32 boundaries |
| Attention Norm into BF16 QKV Arena | smoke-tested, default route | full models 1.01309×/1.01303×, exact complete logits, allocations -120/-140 and peak -3,670,016/-6,291,456 bytes | reprofile before selecting another cast boundary |
| Post-FFN-Norm inference profile | smoke-tested | four load-subtracted processes; Kernel 8.208/14.659ms, casts 72/84, GEMM 60.9%/68.2%, default route explicitly true | next bounded target is Attention Norm into QKV Arena |
| Current B1T1024 inference profile | smoke-tested | four rocprof processes, `(6−1)/5`; GEMM 59.7%/66.8%, softmax 14.8%/9.2%, default online calls zero | exact Attention solution track measured and closed; select a new profile-backed target |
| Exact T1024 Attention solutions | implemented, rejected | 12 operator processes find four 1.060×–1.538× local winners; PV has 175 descriptor misses/0 dispatch; 12 model processes reject Qwen for logits and DeepSeek for only 1.002× | a future interleaved-BTHD PV tuner must screen the real descriptor; no default index |
| BF16 vectorized SwiGLU | implemented, explicit only | caller-output API and tail-safe four-value kernel are bit-identical; operator 1.249×/1.190×, but full models only 1.007×/1.001× | Auto remains scalar; future work must cross the grouped-GEMM epilogue boundary |
| Grouped gate Swish epilogue | implemented, rejected | 64/64 candidates pass per T1024 process and pointer-stable operator is 1.097×/1.069×; full models are 1.000×/0.991× with Max logits 0.0973/0.0362 | explicit switch remains default-off; local FFN activation track closed |
| Direct BF16 RMSNorm output | smoke-tested, default FFN Arena route | operator bit-identical at Event 1.866×/2.070×; full models 1.0122×/1.0092×, exact logits, unchanged peak and 120/140 fewer measured allocations | Attention Norm still has a separate boundary |
| rocWMMA QK tile | smoke-tested | gfx942 rocWMMA 2.2.0, 48 complete-output processes over T16–2048/D64–128; T512 is 1.654×–1.784× hipBLASLt while T2048 D128 is 0.688× | standalone tile track handed to the separate online prototype; no direct model route |
| rocWMMA online Attention prototype | smoke-tested | 42 complete-output processes, real Qwen/DeepSeek GQA grids T32–2048, 1.260×–4.041× current operator and zero global score bytes | handed to the public operator; benchmark prototype has no direct model route |
| Public online Attention operator | smoke-tested | BF16→FP32 BTHD API, gfx942 batch-native route, exact counters, PyTorch/CPU/HIP; 10 native cases 1.534×–2.456× and 4 exact fallback counterexamples | other-GPU validation; full-model route was measured and rejected |
| Online Attention model route | implemented, rejected | 36 processes, exact 168/196 native hits, 3.5–57MiB saved; all six prefill ratios 0.761×–0.884× and Qwen Max/RMS up to 0.511/0.112 | superseded by direct-BF16 rebuttal; track closed |
| Direct-BF16 online model rebuttal | implemented, rejected | grouped QKV retains BF16 V and bias/RoPE writes BF16 directly; 36 processes still only 0.777×–0.906× with Qwen Max/RMS 0.485/0.110 | online model track closed; retained primitives need independent consumers |
| Complete-training HIP Graph | smoke-tested | 24-process FP32/BF16 staged audit; dynamic Storage is rejected before driver invalidation and same-Stream recovery passes; AdamW captures 21 nodes but replay does not advance host step | graph-wide liveness/workspace plan and device-owned optimizer step before any full-step performance claim |
| rocprofv3 workflow | smoke-tested | kernel/HIP API/memory/full trace generated | release artifact retention |
| hipBLASLt matmul | smoke-tested | FP32/BF16, rank-N strided batches, four transpose contracts, Model-S and T≥256 Attention forward/backward | workspace-aware candidate enumeration/timing |
| BF16 solution tuning | smoke-tested | eight T512 shapes, 24 processes, 1,536 complete-output candidates and two rejected model policies | stable cross-process winner plus both-model 1.05 gate before persistence |
| Matmul tuning registry | smoke-tested | exact persistent key plus complete-output gate, Event/wall P50/P95 and explicit acceptance | solution-index enumeration and automatic model regression |
| RCCL two-GPU baseline | smoke-tested | XGMI average and global-batch parameter equivalence | buckets/4 GPU/failure timing |
| DataParallelTrainer | smoke-tested | 3-step two-rank/global-batch equivalence, rank diff 0, stage trace | one-process-per-GPU/gradient-ready overlap |
| RCCL gradient buckets | smoke-tested | 1MB payload with 64/4/1 bucket matrix | overlap with backward |
| RCCL four-GPU | draft | 3 stable init failures and debug root cause | environment with >87MB /dev/shm |
| RCCL compute overlap | smoke-tested | 3 runs, 30–33% synthetic overlap gain | bucket readiness during real backward |
