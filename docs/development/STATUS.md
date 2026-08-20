# Evidence status

States: `draft`, `implemented`, `smoke-tested`, `reference-trained`, `released`.

| Component | State | Current evidence | Missing gate |
|---|---|---|---|
| CPU configuration | smoke-tested | framework-only main configure/build; 161/161 CPU CTest | CI matrix |
| CPU code coverage | smoke-tested | 83.9% lines, 90.9% functions, 66.6% branches over `src/` + `include/` | split CPU/HIP reports and add justified thresholds |
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
| MI300X precision capabilities | smoke-tested | dedicated gfx942 gate; FP32/FP16/BF16/FP8 hipBLASLt execution and Event speedup | INT8 probe and packed INT4 software path |
| FP8 training/inference | smoke-tested | FNUZ kernels, scaled GEMM, FP32 master/backward, Transformer Linear policy and KV decode | dynamic amax/history and full training curve |
| Qwen2.5-0.5B | smoke-tested | official FP32 plus single-representation BF16 FFN; full logits, exact tokens, 31.7% current-memory reduction | BF16 prefill parity/tool chat/multi-step SFT |
| DeepSeek-R1-Distill-Qwen-1.5B | smoke-tested | official 339 tensors; BF16 FFN exact 8 tokens, 32.5% current-memory reduction | PyTorch BF16 decode/prefill parity, longer reasoning/SFT |
| Operator context | smoke-tested | explicit Stream ordering and mismatch tests | low-level C descriptor |
| CPU Transformer Autograd | smoke-tested | dedicated graph construction tests, finite differences, PyTorch full-graph gradients | more dtypes |
| HIP Autograd | smoke-tested | CPU/HIP full Transformer gradient comparison; zero host transfers during graph execution | optimized reductions/more dtypes |
| SGD/AdamW | smoke-tested | PyTorch parity plus zero-transfer device-native HIP AdamW and real Qwen update | mixed precision/scaler |
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
| Weight/state API | smoke-tested | independent state_dict, atomic strict/non-strict load, Qwen-style transpose mapping | streaming load/model-specific architecture validation |
| safetensors | smoke-tested | F32/BF16/F16, shards/index/corruption/CPU-HIP plus official Python package two-way interop | FP8/quantized formats/memory mapping |
| C++ training CLI | smoke-tested | CPU save/resume fixture and Model-S HIP real-text steps | validation/report CLI |
| Model-S TinyStories smoke | smoke-tested | immutable 1MiB train prefix, 10 HIP steps | full train/validation curve |
| Tiny Transformer training | smoke-tested | 40-step overfit and finite gradients | validation split/Model-S |
| SFT response masking | smoke-tested | CPU/HIP ignored targets and tiny response-loss curve | Model-S instruction corpus/run |
| CPU/HIP KV cache | smoke-tested | request-bounded preallocation, stable address, every MHA/GQA prefix, zero-transfer MI300X decode; score 0.885816→1.167931 | batching, fused long-context attention |
| Device greedy sampling | smoke-tested | deterministic two-stage 151936 argmax (-96.7% Kernel), 4-byte/token D2H, robust score 1.770568 | stochastic device top-k/RNG |
| HIP exact-size allocator | smoke-tested | steady-state pool plus 16-block shared retirement Events; score 2.470863 | size classes and explicit multi-Stream ownership |
| Fused cached decode Attention | smoke-tested | FP32 MHA/GQA 1/32/128/512 + fallback; repeated-process score 1.752183; long decode up to +57.9% | one-token regression, prefill/backward/BF16 |
| Fused Q/K bias + split-half RoPE | smoke-tested | CPU/HIP/PyTorch forward+backward; 1,120 fewer profiled launches; paired generation +13.7%/+6.6%; score 1.784147 | interleaved/low-precision variants and remaining launch fusion |
| Fused cached residual + RMSNorm | smoke-tested | pair-output oracle, 532 fewer launches, 512-thread wide path; DeepSeek +9.6%; score 1.845199 | broader width matrix and training graph fusion |
| BF16 GEMM/autograd/FFN inference | smoke-tested | operator island plus transactional single-representation official policy; 18 exact-token rows, 1.115×/1.051× decode vs micro FP32 | 3/4 PyTorch full-BF16 performance rows below parity; mixed training |
| Token generation | smoke-tested | deterministic sampling and cache-backed length/bounds | trained text report |
| Stable model failure | smoke-tested | low-loss cycle breaks beyond training context | rebuttal experiments |
| PyTorch Custom Ops | smoke-tested on CPU | Torch 2.13 add/multiply via dispatcher | build/run with PyTorch ROCm |
| PyTorch correctness oracle | smoke-tested on CPU | 30 Tensor APIs, 29 graph APIs, 25 test files audited; forward/backward/shape/model/optimizer parity | direct PyTorch ROCm run |
| PyTorch ROCm environment | draft | matching 2.11/7.13 wheel Bus error on import | working Torch ROCm environment |
| C ABI v1 | smoke-tested | pure C CPU/HIP create/copy/ops/error client | zero-copy external views |
| Python ctypes API | smoke-tested | CPU/HIP Tensor/ops/error unittest | packaging/broader ops |
| External TensorView ops | smoke-tested | caller-owned CPU/HIP buffers and Stream | Torch build validation |
| In-process profiling | smoke-tested | TraceSession/TraceTimer, values/operator/layer passes, CPU/HIP tests | async Event completion/rocprof markers/Python decorator |
| Cross-framework alignment | smoke-tested | CPU and MI300X both pass 58/58 forward/loss/all-parameter-gradient checkpoints, plus op/layer/backward timings | Qwen/DeepSeek runners/direct PyTorch ROCm |
| Profiling/autotune | smoke-tested | rocprofv3, hipBLASLt, exact-shape registry | general persistent arch/version candidate cache |
| Micro-benchmark harness | smoke-tested | CPU/HIP Event+wall JSONL and error gate | PyTorch operator timing/more shapes |
| Engine allocation tracker | smoke-tested | CPU/HIP current/peak/total accounting | external allocator integration |
| End-to-end benchmark | smoke-tested | matched Python/PyTorch ROCm built-in and official-HF JSONL ratios | longer contexts, repeated official training and tuned comparison |
| Single-GPU model matrix | smoke-tested | MI300X tiny/Model-S/Model-M plus official Qwen0.5B/DeepSeek-Distill1.5B train+generate, parameters, time, tokens/s and engine peak bytes | repeated contexts/dtypes and external board-level memory sampling |
| Python/PyTorch ROCm performance matrix | smoke-tested | matched FP32 built-ins plus official Qwen/DeepSeek Distill, raw Python data and automatic throughput/memory ratios | repeated official-model training and PyTorch version matrix |
| Optimization experiment journal | implemented | FP32 main track plus BF16 shape/model/FFN tracks, generated figures and repeated raw evidence | whole-model BF16, HIP Graph and long-context tracks |
| rocprofv3 workflow | smoke-tested | kernel/HIP API/memory/full trace generated | release artifact retention |
| hipBLASLt matmul | smoke-tested | FP32 CPU comparison, shape matrix, Model-S e2e | batched/workspace/autotune cache |
| Matmul tuning registry | smoke-tested | exact shape override/clear and availability gates | persistence/arch key |
| RCCL two-GPU baseline | smoke-tested | XGMI average and global-batch parameter equivalence | buckets/4 GPU/failure timing |
| DataParallelTrainer | smoke-tested | 3-step two-rank/global-batch equivalence, rank diff 0, stage trace | one-process-per-GPU/gradient-ready overlap |
| RCCL gradient buckets | smoke-tested | 1MB payload with 64/4/1 bucket matrix | overlap with backward |
| RCCL four-GPU | draft | 3 stable init failures and debug root cause | environment with >87MB /dev/shm |
| RCCL compute overlap | smoke-tested | 3 runs, 30–33% synthetic overlap gain | bucket readiness during real backward |
