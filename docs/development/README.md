# Development records

This directory is the chronological record requested for the main repository.
Each milestone records scope, decisions, commands, evidence, known failures, and
the next gate. Records are append-only except for factual corrections.

- [ROADMAP.md](ROADMAP.md): planned development sequence and acceptance gates.
- [NEXT_STEPS.md](NEXT_STEPS.md): explicit gaps for external weights, real-model
  compatibility, registry/profiler maturity, and release evidence.
- [STATUS.md](STATUS.md): current evidence state by subsystem.
- [2026-08-19-m0-n0-bootstrap.md](2026-08-19-m0-n0-bootstrap.md): first engine
  bootstrap and CPU Tensor vertical slice.
- [2026-08-19-m1-hip-runtime.md](2026-08-19-m1-hip-runtime.md): HIP allocation,
  transfer, Stream, and Event boundary.
- [2026-08-19-m1-cpu-reference-ops.md](2026-08-19-m1-cpu-reference-ops.md): readable
  CPU oracle for the first Transformer operator set.
- [2026-08-19-m1-hip-basic-ops.md](2026-08-19-m1-hip-basic-ops.md): readable HIP
  elementwise kernels and naive batched matmul.
- [2026-08-19-m1-hip-transformer-ops.md](2026-08-19-m1-hip-transformer-ops.md):
  readable HIP Transformer operator set and CPU/HIP conformance.
- [2026-08-19-m1-op-context.md](2026-08-19-m1-op-context.md): explicit Stream and
  workspace context plus the N1 CPU/HIP artifact.
- [2026-08-19-m2-autograd-core.md](2026-08-19-m2-autograd-core.md): eager reverse-mode
  graph, gradient accumulation, and finite differences.
- [2026-08-19-m2-transformer-backward.md](2026-08-19-m2-transformer-backward.md):
  backward paths needed by Transformer training.
- [2026-08-19-m2-optimizers.md](2026-08-19-m2-optimizers.md): SGD, AdamW, and
  optimizer-state continuation equivalence.
- [2026-08-19-m2-checkpoint.md](2026-08-19-m2-checkpoint.md): versioned complete
  training state and multi-step resume equivalence.
- [2026-08-19-m3-attention-primitives.md](2026-08-19-m3-attention-primitives.md):
  differentiable causal masking and contiguous graph materialization.
- [2026-08-19-m3-model-config.md](2026-08-19-m3-model-config.md): executable Model-S
  and Model-M parameter budgets.
- [2026-08-19-m3-gqa-repeat.md](2026-08-19-m3-gqa-repeat.md): differentiable K/V
  head expansion for grouped-query Attention.
- [2026-08-19-m3-transformer-model.md](2026-08-19-m3-transformer-model.md): trainable
  Decoder-only MHA/GQA Transformer composition.
- [2026-08-19-m3-data-pipeline.md](2026-08-19-m3-data-pipeline.md): byte tokenizer
  and deterministic resumable token batches.
- [2026-08-19-m3-tiny-overfit.md](2026-08-19-m3-tiny-overfit.md): complete tiny
  Transformer training loop and measured loss trajectory.
- [2026-08-19-m3-kv-cache.md](2026-08-19-m3-kv-cache.md): real per-layer MHA/GQA
  K/V caching and cached/full logit equivalence.
- [2026-08-19-m3-generation.md](2026-08-19-m3-generation.md): deterministic greedy
  and sampled generation over the real KV cache.
- [2026-08-19-m3-trained-generation-failure.md](2026-08-19-m3-trained-generation-failure.md):
  low training loss with a stable beyond-context generation failure.
- [2026-08-19-m3-model-s-forward.md](2026-08-19-m3-model-s-forward.md): real
  15.6M-parameter Model-S construction and CPU forward smoke.
- [2026-08-19-m3-model-s-train-step.md](2026-08-19-m3-model-s-train-step.md): full
  Model-S backward and AdamW update smoke.
- [2026-08-19-m3-model-s-hip-forward.md](2026-08-19-m3-model-s-hip-forward.md):
  complete Model-S weight transfer and readable HIP forward comparison.
- [2026-08-19-m3-hip-strided-copy.md](2026-08-19-m3-hip-strided-copy.md): generic
  HIP view materialization required by backward and multi-token inference.
- [2026-08-19-m3-hip-training.md](2026-08-19-m3-hip-training.md): first complete
  GPU-resident Transformer training trajectory and its host-reference boundaries.
- [2026-08-19-m4-c-api.md](2026-08-19-m4-c-api.md): stable versioned C ABI and
  pure C CPU/HIP integration client.
- [2026-08-19-m4-python-api.md](2026-08-19-m4-python-api.md): dependency-free ctypes
  API consuming the same C ABI.
- [2026-08-19-m4-external-tensor-view.md](2026-08-19-m4-external-tensor-view.md):
  zero-copy caller-owned TensorView operators and explicit Stream interop.
- [2026-08-19-m4-torch-custom-ops.md](2026-08-19-m4-torch-custom-ops.md): optional
  PyTorch CPU/ROCm registration source and missing-environment evidence.
- [2026-08-19-m5-micro-benchmark.md](2026-08-19-m5-micro-benchmark.md): reproducible
  operator timing schema and first CPU/gfx942 measurements.
- [2026-08-19-m5-allocation-tracker.md](2026-08-19-m5-allocation-tracker.md): engine
  current/peak/total allocation accounting.
- [2026-08-19-m5-end-to-end-benchmark.md](2026-08-19-m5-end-to-end-benchmark.md):
  train/generate tokens/s, peak memory, and a stable GPU-slower failure.
- [2026-08-19-m5-rocprof-trace.md](2026-08-19-m5-rocprof-trace.md): rocprofv3
  runtime trace confirming host copies and launch count as the current bottleneck.
- [2026-08-19-m5-hipblaslt.md](2026-08-19-m5-hipblaslt.md): shape-aware hipBLASLt
  candidate, Model-S speedup, and setup-time regression.
- [2026-08-19-m5-tuning-registry.md](2026-08-19-m5-tuning-registry.md): safe exact-shape
  implementation override seam for offline tuning skills.
- [2026-08-19-m6-rccl-equivalence.md](2026-08-19-m6-rccl-equivalence.md): real two-GPU
  all-reduce and single/global-batch training-step equivalence.
- [2026-08-19-m6-buckets-and-four-rank-failure.md](2026-08-19-m6-buckets-and-four-rank-failure.md):
  bucket timing matrix and reproducible four-rank shared-memory failure.
- [2026-08-19-m6-async-collective.md](2026-08-19-m6-async-collective.md): enqueue/wait
  collective split required for communication-compute overlap.
- [2026-08-19-m6-overlap-experiment.md](2026-08-19-m6-overlap-experiment.md):
  two-Stream overlap measurements and multi-GPU device-selection bug.
- [2026-08-19-course-and-evidence.md](2026-08-19-course-and-evidence.md): N0–N8,
  PA0–PA2, unified artifact parser, and CPU CI.
- [2026-08-19-model-m-hip-train-step.md](2026-08-19-model-m-hip-train-step.md):
  actual 128MB-tier forward/backward/AdamW step on MI300X.
- [2026-08-19-bpe-and-tinystories.md](2026-08-19-bpe-and-tinystories.md): self-contained
  BPE and licensed immutable TinyStories loader source.
- [2026-08-19-sft-response-masking.md](2026-08-19-sft-response-masking.md): CPU/HIP
  ignored-target loss and tiny SFT trajectory.
- [2026-08-19-training-cli-real-text-smoke.md](2026-08-19-training-cli-real-text-smoke.md):
  pure C++ save/resume CLI and Model-S TinyStories HIP trajectory.
- [2026-08-19-torch-cpu-validation.md](2026-08-19-torch-cpu-validation.md): optional
  dispatcher binding compiled and run with an isolated Torch CPU wheel.
- [2026-08-19-torch-rocm-environment-failure.md](2026-08-19-torch-rocm-environment-failure.md):
  matching AMD wheel fails before Custom Op build.
- [2026-08-19-verification-summary.md](2026-08-19-verification-summary.md): final build/test
  matrix for this development branch and remaining release blockers.
- [2026-08-19-device-native-autograd.md](2026-08-19-device-native-autograd.md):
  self-written eager graph engine running the Transformer forward/backward path on HIP
  without host transfers.
- [2026-08-19-pytorch-parity-and-graph-tests.md](2026-08-19-pytorch-parity-and-graph-tests.md):
  dedicated graph construction tests, full PyTorch operator/model/optimizer oracle, and
  machine-enforced test-file coverage.
- [2026-08-19-branch-separation.md](2026-08-19-branch-separation.md): framework on
  `main`, beginner course on `tutorial/beginner-course`, and independent verification
  boundaries.
- [2026-08-19-weight-api.md](2026-08-19-weight-api.md): named model state,
  safetensors/shards/index, official-package two-way interop, Qwen-style mappings,
  corruption tests, and direct HIP load.
- [2026-08-19-repository-presentation.md](2026-08-19-repository-presentation.md):
  professional README, developer documentation hierarchy, CMake presets, and explicit
  compiler/ROCm environment matrix.
- [2026-08-19-alignment-infrastructure.md](2026-08-19-alignment-infrastructure.md):
  four-pass microLLM/PyTorch forward/loss/gradient/timing traces, automatic comparison, and
  complete experiment manifests.
- [2026-08-19-operator-property-matrix.md](2026-08-19-operator-property-matrix.md):
  deterministic multi-rank shape/edge tests and randomized finite-difference gradients.
- [2026-08-19-cpu-code-coverage.md](2026-08-19-cpu-code-coverage.md): measured
  GCC/gcovr line, function, and branch coverage plus the remaining blind spots.
- [2026-08-19-low-precision-tensor-storage.md](2026-08-19-low-precision-tensor-storage.md):
  real FP16/BF16 storage, cast/view/device-copy semantics, and MI300/MI350 boundaries.
- [2026-08-19-mi300x-precision-capabilities.md](2026-08-19-mi300x-precision-capabilities.md):
  dedicated hardware-format gates and measured FP32/FP16/BF16 Matrix acceleration.
- [2026-08-19-fp8-training-inference.md](2026-08-19-fp8-training-inference.md):
  FNUZ quantization kernels, scaled GEMM, FP32-master training op, and measured ratios.
- [2026-08-19-qwen25-architecture.md](2026-08-19-qwen25-architecture.md): pinned HF
  config, Q/K/V bias, split-half RoPE, exact parameter count, and remaining real-model gates.
- [2026-08-19-deepseek-distill.md](2026-08-19-deepseek-distill.md): official dense
  Distill-Qwen checkpoint, reasoning chat, complete logits, KV tokens, and resource evidence.
- [2026-08-19-device-native-adamw.md](2026-08-19-device-native-adamw.md): zero-transfer
  HIP AdamW and an official Qwen2.5 loss/parameter-update comparison with PyTorch.
- [2026-08-19-data-parallel-trainer.md](2026-08-19-data-parallel-trainer.md):
  reusable two-rank training step, bucketed gradient average, global-batch equivalence,
  stage profiling, and production reducer gaps.
- [2026-08-20-course-only-branch.md](2026-08-20-course-only-branch.md): removed the
  duplicated engine from the tutorial branch, added a course-only CI boundary, and
  extended the curriculum through official HF models and FP8.
- [2026-08-20-single-gpu-model-matrix.md](2026-08-20-single-gpu-model-matrix.md):
  MI300X training/generation throughput and engine-owned memory for tiny, Model-S and
  Model-M, with a CTest schema gate.
- [2026-08-20-single-gpu-hf-model-matrix.md](2026-08-20-single-gpu-hf-model-matrix.md):
  official Qwen2.5-0.5B and DeepSeek-R1-Distill-Qwen-1.5B inference/training time,
  throughput, memory and missing-checkpoint semantics.
- [2026-08-20-pytorch-performance-comparison.md](2026-08-20-pytorch-performance-comparison.md):
  independent Python/PyTorch ROCm built-in and official-model matrices with matched
  workload enforcement and measured ratios.
- [2026-08-20-optimization-log.md](2026-08-20-optimization-log.md): living 0→1
  optimization blog, experiment protocol, measured step contracts and generated
  autoresearch-style progress/bottleneck SVGs.
- [2026-08-20-bf16-autograd-policy.md](2026-08-20-bf16-autograd-policy.md): retained
  FP32-master BF16 autograd primitive, rejected official-model policy and raw evidence.
- [2026-08-20-fused-bias-rope.md](2026-08-20-fused-bias-rope.md): first-class fused
  Q/K bias+RoPE forward/backward, paired official-model medians and profiler evidence.
- [2026-08-20-fused-residual-rmsnorm.md](2026-08-20-fused-residual-rmsnorm.md): cached
  pair-output residual+Norm fusion, launch reduction and recorded DeepSeek regression.
- [2026-08-20-wide-residual-norm.md](2026-08-20-wide-residual-norm.md): measured
  256/512-thread width policy that resolves the preceding DeepSeek regression.
- [2026-08-20-batched-retirement-events.md](2026-08-20-batched-retirement-events.md):
  shared completion Events, allocator safety gates and four-workload speedup.
- [2026-08-20-bf16-output-gemm.md](2026-08-20-bf16-output-gemm.md): explicit
  BF16-output/FP32-accumulate GEMM foundation for activation islands.
- [2026-08-20-bf16-ffn-island.md](2026-08-20-bf16-ffn-island.md): continuous
  BF16 FFN, real decode-shape fallback, 36-run matrix and generated evidence chart.
- [2026-08-20-parallel-cross-entropy.md](2026-08-20-parallel-cross-entropy.md):
  Experiment 001 block-parallel CE, large-vocabulary/PyTorch gates, 66.1% score gain,
  profiler before/after and next-hotspot handoff.
