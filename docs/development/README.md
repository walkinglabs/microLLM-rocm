# Development records

- [2026-08-26: Autograd external gradient buffer](2026-08-26-autograd-external-gradient-buffer.md)
- [2026-08-26: PyTorch zero-copy backward matrix](2026-08-26-pytorch-zero-copy-backward.md)
- [2026-08-26: PyTorch zero-copy RoPE/Embedding/loss](2026-08-26-pytorch-zero-copy-sequence-loss.md)
- [2026-08-26: PyTorch zero-copy MHA/GQA Attention](2026-08-26-pytorch-zero-copy-attention.md)
- [2026-08-26: PyTorch zero-copy operator matrix](2026-08-26-pytorch-zero-copy-operator-matrix.md)
- [2026-08-26: PyTorch zero-copy FP16/BF16](2026-08-26-pytorch-zero-copy-low-precision.md)
- [2026-08-26: PyTorch ROCm zero-copy Tensor](2026-08-26-pytorch-zero-copy-tensor.md)
- [2026-08-26: PyTorch ROCm native Stream interop](2026-08-26-pytorch-native-stream-interop.md)
- [2026-08-26: Explicit Python Stream isolation](2026-08-26-python-stream-isolation.md)
- [2026-08-26: Python asynchronous HIP Event completion](2026-08-26-python-hip-event-completion.md)
- [2026-08-26: Python/ROCTX/GPU clock calibration](2026-08-26-python-roctx-gpu-clock.md)
- [2026-08-26: CMake Config component and relocation gates](2026-08-26-cmake-config-component-gates.md)
- [2026-08-26: Unified ROCTX/GPU Perfetto](2026-08-26-unified-rocprof-perfetto.md)
- [2026-08-26: rocprof to Perfetto correlation merge](2026-08-26-rocprof-perfetto-merge.md)
- [2026-08-26: Optional ROCTX TraceTimer ranges](2026-08-26-roctx-trace-ranges.md)
- [2026-08-26: Python Perfetto export](2026-08-26-python-perfetto-export.md)
- [2026-08-26: Python profile decorator](2026-08-26-python-profile-api.md)
- [2026-08-26: Qwen tool chat template](2026-08-26-qwen-tool-chat.md)
- [2026-08-26: Safetensors mmap visits](2026-08-26-safetensors-mmap.md)
- [2026-08-26: Indexed HIP weight streaming](2026-08-26-indexed-streaming.md)
- [2026-08-26: Multi-shard HIP weight streaming](2026-08-26-multishard-streaming.md)
- [2026-08-26: Grouped cached-decode cleanup](2026-08-26-grouped-decode-cleanup.md)
- [2026-08-26: Decode rows2 grouped gate/up](../optimization-log/experiments/325-grouped-gate-up-row2.md)
- [2026-08-26: Native128 finalize cleanup](2026-08-26-native128-finalize-cleanup.md)
- [2026-08-26: Native128 finalize rejection](2026-08-26-native128-finalize-reject.md)
- [2026-08-26: Native128 finalize infrastructure](2026-08-26-native128-finalize-infrastructure.md)
- [2026-08-26: Finalize architecture gap audit](../optimization-log/experiments/323-finalize-architecture-gap.md)
- [2026-08-26: Clean long-context baseline and profile](2026-08-26-clean-long-context-profile.md)
- [2026-08-26: FP32 FFN down rejection](2026-08-26-fp32-ffn-down-reject.md)
- [2026-08-26: FP32 FFN down runner](2026-08-26-fp32-ffn-down-runner.md)
- [2026-08-26: Rejected FFN route cleanup](2026-08-26-prefill-ffn-route-cleanup.md)
- [2026-08-26: Down after exact gate/up](2026-08-26-post-exact-gate-up-down.md)
- [2026-08-26: Post-exact gate/up FFN runner](2026-08-26-post-exact-gate-up-runner.md)
- [2026-08-26: All-exact FFN rejection](2026-08-26-prefill-ffn-all-exact-reject.md)
- [2026-08-26: All-batch exact FFN runner](2026-08-26-prefill-ffn-all-exact-runner.md)
- [2026-08-26: Selective prefill FFN model rejection](2026-08-26-prefill-ffn-selective-reject.md)
- [2026-08-26: Scoped prefill FFN model gate](2026-08-26-prefill-ffn-scope-and-model-runner.md)
- [2026-08-26: FP32 FFN row-invariance result](2026-08-26-fp32-ffn-row-invariance-result.md)
- [2026-08-26: FP32 FFN row-invariance runner](2026-08-26-fp32-ffn-row-invariance-runner.md)
- [2026-08-26: FFN gate/up first-drift result](2026-08-26-prefill-ffn-gate-up-result.md)
- [2026-08-26: Complete-value prefill FFN runner](2026-08-26-prefill-ffn-stage-runner.md)
- [2026-08-26: Cached-prefill FFN detail trace](2026-08-26-prefill-ffn-detail-trace.md)
- [2026-08-26: Exact-stack model rejection](2026-08-26-prefill-exact-stack-reject.md)
- [2026-08-26: Batch-selective exact-stack model gate](2026-08-26-prefill-exact-stack-gate-infrastructure.md)
- [2026-08-26: Why the scoped O model candidate was rejected](2026-08-26-prefill-o-model-reject.md)
- [2026-08-26: Scoped O complete model-gate runner](2026-08-26-prefill-o-model-gate-infrastructure.md)
- [2026-08-26: FFN output after exact O](2026-08-26-post-exact-o-ffn-output.md)
- [2026-08-26: Post-exact-O block trace runner](2026-08-26-post-exact-o-trace-infrastructure.md)
- [2026-08-26: Scoped prefill O projection](2026-08-26-prefill-o-projection-scope.md)
- [2026-08-26: O projection after exact core](2026-08-26-post-exact-core-o-projection.md)
- [2026-08-26: Post-exact-core block trace runner](2026-08-26-post-exact-core-trace-infrastructure.md)
- [2026-08-26: Batch-selective Attention rejection](2026-08-26-batch-selective-attention-reject.md)
- [2026-08-26: Batch-selective Attention gate runner](2026-08-26-batch-selective-attention-gate.md)
- [2026-08-26: Prefill Attention model rejection](2026-08-26-prefill-attention-model-reject.md)
- [2026-08-26: Prefill Attention model-gate runner](2026-08-26-prefill-attention-model-gate-infrastructure.md)
- [2026-08-26: Scoped cached-prefill Attention solutions](2026-08-26-prefill-attention-scopes.md)
- [2026-08-26: Attention batch solution result](2026-08-26-attention-batch-solution-result.md)
- [2026-08-26: Real Attention batch-invariance harness](2026-08-26-attention-batch-invariance-infrastructure.md)
- [2026-08-26: T2048 prefill Attention core matrix](2026-08-26-prefill-attention-core-matrix.md)
- [2026-08-26: Filtered binary TraceSession export](2026-08-26-binary-trace-export.md)
- [2026-08-26: Prefill Attention core diagnostics](2026-08-26-prefill-attention-core-diagnostics.md)
- [2026-08-26: Symmetric CPU, HIP, and RCCL SDK presets](2026-08-26-cmake-sdk-presets.md)
- [2026-08-25: Cross-batch precision-island isolation](2026-08-25-cross-batch-precision-isolation.md)
- [2026-08-25: Cross-batch complete-logit audit](2026-08-25-cross-batch-logit-audit.md)
- [2026-08-25: Serving batch scale, explained simply](2026-08-25-serving-batch-scale.md)
- [2026-08-25: Why exact GQA value reuse was still slower](2026-08-25-exact-gqa-value-reuse-reject.md)
- [2026-08-25: Why split-P*V model precision failed](2026-08-25-split-pv-model-reject.md)
- [2026-08-25: Explicit split-P*V model route](2026-08-25-split-pv-model-route.md)
- [2026-08-25: Exact softmax and split P*V, explained simply](2026-08-25-exact-softmax-split-pv.md)
- [2026-08-25: Why fewer finalize threads did not help](2026-08-25-finalize-thread-mapping-discard.md)
- [2026-08-25: Post-materialized profile, explained simply](2026-08-25-post-materialized-profile.md)
- [2026-08-25: Materialized-score automatic policy result](../optimization-log/experiments/288-materialized-score-auto-policy.md)
- [2026-08-25: Materialized-score scoped auto policy](2026-08-25-materialized-auto-policy.md)
- [2026-08-25: Materialized-score default boundary](../optimization-log/experiments/287-materialized-score-model-boundary.md)
- [2026-08-25: Materialized-score multi-model matrix infrastructure](2026-08-25-materialized-model-matrix-infrastructure.md)
- [2026-08-25: Materialized-score official model result](../optimization-log/experiments/286-materialized-score-model.md)
- [2026-08-25: Materialized-score explicit model route](2026-08-25-materialized-score-model-route.md)
- [2026-08-25: Materialized-score operator result](../optimization-log/experiments/285-materialized-score-attention.md)
- [2026-08-25: Materialized-score matrix infrastructure](2026-08-25-materialized-score-matrix-infrastructure.md)
- [2026-08-25: Materialized-score exact-order cached Attention](2026-08-25-materialized-score-cached-attention.md)
- [2026-08-25: Split-sequence model rejection](../optimization-log/experiments/284-cached-attention-split-model-reject.md)
- [2026-08-25: Split-sequence official model-gate infrastructure](2026-08-25-split-sequence-model-gate-infrastructure.md)
- [2026-08-25: Split-sequence explicit model route](2026-08-25-split-sequence-model-route.md)
- [2026-08-25: Split-sequence operator result](../optimization-log/experiments/283-cached-attention-split-search.md)
- [2026-08-25: Split-sequence matrix infrastructure](2026-08-25-split-sequence-matrix-infrastructure.md)
- [2026-08-25: Split-sequence cached Attention](2026-08-25-split-sequence-cached-attention.md)
- [2026-08-25: Cached Attention stage-matrix result](../optimization-log/experiments/282-cached-attention-stage-matrix.md)
- [2026-08-25: Cached Attention stage-matrix infrastructure](2026-08-25-cached-attention-stage-matrix-infrastructure.md)
- [2026-08-25: Cached Attention context oracle](2026-08-25-cached-attention-context-oracle.md)
- [2026-08-25: Current DeepSeek T2048 profile](2026-08-25-current-deepseek-t2048-profile.md)
- [2026-08-25: Current cached-decode profile runner](2026-08-25-current-cached-profile-runner.md)
- [2026-08-25: Current DeepSeek T2048 baseline](2026-08-25-current-deepseek-t2048-baseline.md)
- [2026-08-25: Cached Attention score oracle](2026-08-25-cached-attention-score-oracle.md)
- [2026-08-25: Next long-context profile audit](2026-08-25-long-context-profile-audit.md)
- [2026-08-25: Evidence status refresh](2026-08-25-evidence-status-refresh.md)
- [2026-08-25: Ranked gather-scale result](2026-08-25-ranked-gather-scale-result.md)
- [2026-08-25: Ranked persistent gather-scale infrastructure](2026-08-25-ranked-gather-scale-infrastructure.md)
- [2026-08-25: Ranked ready-bucket weighting result](2026-08-25-ranked-bucket-weighting-result.md)
- [2026-08-25: Ranked ready-bucket weighting infrastructure](2026-08-25-ranked-bucket-weighting-infrastructure.md)
- [2026-08-25: Ranked weighted-overlap result](2026-08-25-ranked-weighted-overlap-result.md)
- [2026-08-25: Ranked weighted-overlap matrix infrastructure](2026-08-25-ranked-weighted-overlap-matrix.md)
- [2026-08-25: Ranked token-weighted overlap infrastructure](2026-08-25-ranked-weighted-overlap-infrastructure.md)
- [2026-08-25: Ranked Model-S uneven-input result](2026-08-25-ranked-model-s-input-weighting-result.md)
- [2026-08-25: Ranked uneven-input weighting result](2026-08-25-ranked-input-weighting-result.md)
- [2026-08-25: Ranked uneven-input weighting infrastructure](2026-08-25-ranked-input-weighting-infrastructure.md)
- [2026-08-25: Ranked RCCL preflight result](2026-08-25-ranked-rccl-preflight-result.md)
- [2026-08-25: Ranked RCCL debug and resource preflight](2026-08-25-ranked-rccl-preflight-infrastructure.md)
- [2026-08-25: Ranked world-size result](2026-08-25-ranked-world-size-result.md)
- [2026-08-25: Ranked world-size infrastructure](2026-08-25-ranked-world-size-infrastructure.md)
- [2026-08-25: Ranked Model-S checkpoint result](2026-08-25-ranked-model-s-checkpoint-result.md)
- [2026-08-25: Ranked Model-S checkpoint infrastructure](2026-08-25-ranked-model-s-checkpoint-infrastructure.md)
- [2026-08-25: Ranked checkpoint ownership result](2026-08-25-ranked-checkpoint-result.md)
- [2026-08-25: Ranked checkpoint ownership infrastructure](2026-08-25-ranked-checkpoint-infrastructure.md)
- [2026-08-25: Ranked overlap context-scale result](2026-08-25-ranked-overlap-context-result.md)
- [2026-08-25: Ranked overlap context-scale infrastructure](2026-08-25-ranked-overlap-scale-infrastructure.md)
- [2026-08-25: Ranked gradient-ready overlap result](2026-08-25-ranked-gradient-overlap-result.md)
- [2026-08-25: Ranked gradient-ready overlap infrastructure](2026-08-25-ranked-gradient-overlap-infrastructure.md)
- [2026-08-25: Ranked gradient-as-bucket view result](2026-08-25-ranked-gradient-view-result.md)
- [2026-08-25: Ranked gradient-as-bucket view infrastructure](2026-08-25-ranked-gradient-view-infrastructure.md)
- [2026-08-25: Ranked persistent bucket result](2026-08-25-ranked-persistent-bucket-result.md)
- [2026-08-25: Ranked persistent bucket infrastructure](2026-08-25-ranked-persistent-bucket-infrastructure.md)
- [2026-08-25: Ranked Model-S steady reducer result](2026-08-25-ranked-steady-reducer-result.md)
- [2026-08-25: Ranked multi-step cold/steady timing infrastructure](2026-08-25-ranked-multistep-timing-infrastructure.md)
- [2026-08-25: Ranked Model-S natural-bucket result](2026-08-25-ranked-model-s-bucket-result.md)
- [2026-08-25: Ranked Model-S bucket measurement infrastructure](2026-08-25-ranked-model-s-bucket-infrastructure.md)
- [2026-08-25: Ranked gradient bucket result](2026-08-25-ranked-gradient-bucket-result.md)
- [2026-08-25: Ranked gradient bucket infrastructure](2026-08-25-ranked-gradient-bucket-infrastructure.md)
- [2026-08-25: One-process-per-GPU result](2026-08-25-one-process-per-gpu-result.md)
- [2026-08-25: One-process-per-GPU infrastructure](2026-08-25-one-process-per-gpu-infrastructure.md)
- [2026-08-25: Gradient-ready Event overlap result](2026-08-25-gradient-ready-overlap-result.md)
- [2026-08-25: Gradient-ready Event overlap infrastructure](2026-08-25-gradient-ready-overlap-infrastructure.md)
- [2026-08-25: Gradient-ready audit result](2026-08-25-gradient-ready-audit-result.md)
- [2026-08-25: Gradient-ready audit infrastructure](2026-08-25-gradient-ready-audit-infrastructure.md)
- [2026-08-25: Rejected scoped Autograd producer cleanup](2026-08-25-scoped-autograd-gradient-producer-cleanup.md)
- [2026-08-25: Scoped Autograd producer result](2026-08-25-scoped-autograd-gradient-producer-result.md)
- [2026-08-25: Scoped Autograd gradient producer](2026-08-25-scoped-autograd-gradient-producer.md)
- [2026-08-25: Caller-owned weight-gradient producer result](2026-08-25-gradient-producer-out-result.md)
- [2026-08-25: Caller-owned weight-gradient producer](2026-08-25-gradient-producer-out-infrastructure.md)
- [2026-08-25: Rejected direct-gradient route cleanup](2026-08-25-data-parallel-direct-gradient-cleanup.md)
- [2026-08-25: Direct bucket-gradient result](2026-08-25-data-parallel-direct-gradient-result.md)
- [2026-08-25: Direct bucket-gradient infrastructure](2026-08-25-data-parallel-direct-gradient-infrastructure.md)
- [2026-08-25: Gradient-as-bucket view result](2026-08-25-data-parallel-gradient-view-result.md)
- [2026-08-25: Gradient-as-bucket view infrastructure](2026-08-25-data-parallel-gradient-view-infrastructure.md)
- [2026-08-25: Persistent gradient-bucket result](2026-08-25-data-parallel-persistent-bucket-result.md)
- [2026-08-25: Persistent gradient-bucket infrastructure](2026-08-25-data-parallel-persistent-bucket-infrastructure.md)
- [2026-08-25: CMake Config package completion](2026-08-25-cmake-config-package.md)
- [2026-08-25: CMake Config and README usability audit](2026-08-25-cmake-config-readme-audit.md)
- [2026-08-24: Public CMake package consumer](2026-08-24-public-cmake-consumer.md)
- [2026-08-24: CMake SDK and README cleanup](2026-08-24-cmake-sdk-readme.md)
- [2026-08-24: Training add plus RMSNorm Autograd](2026-08-24-training-add-rms-norm-autograd.md)
- [2026-08-24: Multi-tensor AdamW primitive](2026-08-24-multi-tensor-adamw.md)
- [2026-08-24: Training BF16 shared activation](2026-08-24-training-bf16-shared-activation.md)
- [2026-08-24: Post-training-micro saturation](2026-08-24-post-training-micro-saturation.md)
- [2026-08-24: Direct BF16 Q/K into BTHD Attention](2026-08-24-inference-bthd-bf16-qk.md)
- [2026-08-24: Direct BF16 Q/K shape expansion](2026-08-24-inference-bthd-bf16-qk-shapes.md)
- [2026-08-24: 128-thread causal-softmax counterexample](2026-08-24-causal-softmax-128-discard.md)
- [2026-08-24: BF16 V cast/repeat counterexample](2026-08-24-bf16-repeat-fusion-discard.md)
- [2026-08-24: Inference micro-fusion saturation](2026-08-24-inference-micro-fusion-saturation.md)

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
- [2026-08-23-matmul-registry-exact-key.md](2026-08-23-matmul-registry-exact-key.md):
  isolates choices by dtype, layout, architecture, versions, mode and workspace.
- [2026-08-23-matmul-persistent-cache.md](2026-08-23-matmul-persistent-cache.md):
  deterministic JSONL save/load, atomic replacement and environment-version invalidation.
- [2026-08-23-matmul-correctness-before-timing.md](2026-08-23-matmul-correctness-before-timing.md):
  complete-output gates before HIP Event P50/P95 and explicit acceptance.
- [2026-08-23-block-reduction-determinism.md](2026-08-23-block-reduction-determinism.md):
  fixes scratch reuse before all lanes read the previous Attention reduction result.
- [2026-08-23-adamw-correctness-before-timing.md](2026-08-23-adamw-correctness-before-timing.md):
  exact AdamW registry/cache, full state before timing, MI300 shape matrix and retained Scalar Auto.
- [2026-08-23-cooperative-bias-gradient.md](2026-08-23-cooperative-bias-gradient.md):
  contiguous-column 2D reduction, 32-row crossover and same-revision official training win.
- [2026-08-23-post-bias-training-profile.md](2026-08-23-post-bias-training-profile.md):
  phase subtraction removes load-only false hotspots and selects exact training GEMM solutions.
- [2026-08-23-bf16-training-solutions.md](2026-08-23-bf16-training-solutions.md):
  1,536 complete-output solution screens and rejected all/selective model policies.
- [2026-08-23-tied-embedding-sparse-add.md](2026-08-23-tied-embedding-sparse-add.md):
  source-aware gradient attribution and retained 8.11% Qwen peak-memory reduction.
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
- [2026-08-21-streaming-safetensors-load.md](2026-08-21-streaming-safetensors-load.md):
  strict metadata preflight, bounded low-precision staging, direct cast/transpose into
  parameter Storage, and 30–48× pinned load speedups.
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
- [2026-08-23-repeatable-cpu-coverage.md](2026-08-23-repeatable-cpu-coverage.md):
  removes stale runtime profiles and proves three consecutive coverage summaries are identical.
- [2026-08-23-fp8-layer-leave-one-out-runner.md](2026-08-23-fp8-layer-leave-one-out-runner.md):
  complete-logit one-FP32-block screening after the critical-block counterfactual failed.
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
- [2026-08-20-single-representation-bf16-ffn.md](2026-08-20-single-representation-bf16-ffn.md):
  one-way model preparation, official exact tokens, transactional peak and PyTorch BF16 gaps.
- [2026-08-20-prefill-allocator-boundary.md](2026-08-20-prefill-allocator-boundary.md):
  independent workload timing and 1.54×–1.64× official prefill improvement.
- [2026-08-20-deepseek-bf16-decode-profile.md](2026-08-20-deepseek-bf16-decode-profile.md):
  decode-only rocprof evidence and the bounded BF16 Attention handoff.
- [2026-08-20-bf16-attention-shared-cast.md](2026-08-20-bf16-attention-shared-cast.md):
  transactional Q/K/V/O weights, rejected per-Linear cast and retained shared cast.
- [2026-08-20-bf16-plan-cache.md](2026-08-20-bf16-plan-cache.md): scoped immutable
  BF16 hipBLASLt plan reuse and four-row PyTorch BF16 performance pass.
- [2026-08-20-bf16-fp32-master-training.md](2026-08-20-bf16-fp32-master-training.md):
  full-model STE gradients, official multi-step updates and internal performance failure.
- [2026-08-20-bf16-training-qkv-discard.md](2026-08-20-bf16-training-qkv-discard.md):
  paired cast profile and a measured shared-QKV graph rejection.
- [2026-08-20-parallel-cross-entropy.md](2026-08-20-parallel-cross-entropy.md):
  Experiment 001 block-parallel CE, large-vocabulary/PyTorch gates, 66.1% score gain,
  profiler before/after and next-hotspot handoff.
- [2026-08-21-bf16-kv-cache.md](2026-08-21-bf16-kv-cache.md): FP32/BF16 Cache
  policy, exact half-size Storage, B2 T4097 fallback, complete-logit matrix and retained
  DeepSeek RMSE failure.
- [2026-08-21-fused-prefix-pair-discard.md](2026-08-21-fused-prefix-pair-discard.md):
  zero-D2D prefix candidate, clean local profile, stable Qwen T2048 B8 regression and rollback.
- [2026-08-21-mixed-layer-kv-policy.md](2026-08-21-mixed-layer-kv-policy.md): per-layer
  FP32/BF16 Cache API, sensitive-layer search, 12-shape strict precision pass and long-batch cost.
- [2026-08-21-targeted-prefix-pair-discard.md](2026-08-21-targeted-prefix-pair-discard.md):
  same-binary one-FP32-layer fusion retry, reduced D2D and failed prepare/E2E gate.
- [2026-08-21-same-binary-kv-policy.md](2026-08-21-same-binary-kv-policy.md):
  alternating uniform/strict policy runner and correction of a cross-window performance claim.
- [2026-08-21-kv-policy-prompt-robustness.md](2026-08-21-kv-policy-prompt-robustness.md):
  deterministic prompt challenges, one-layer counterexample and first-four robust-strict policy.
- [2026-08-21-qwen-kv-prompt-failure.md](2026-08-21-qwen-kv-prompt-failure.md):
  constant-context complete-logit failure and required all-FP32 fallback.
- [2026-08-21-reference-serving-scheduler.md](2026-08-21-reference-serving-scheduler.md):
  delayed requests, independent state/Cache/RNG, CPU/HIP oracle and serial serving baseline.
- [2026-08-21-static-batch-generation.md](2026-08-21-static-batch-generation.md):
  public compatible-request batch API, row correctness and 1–8 request CPU/HIP scaling.
- [2026-08-21-admission-batch-scheduler.md](2026-08-21-admission-batch-scheduler.md):
  stable compatibility buckets, singleton fallback, late admission and B4 throughput plateau.
- [2026-08-21-request-cancellation.md](2026-08-21-request-cancellation.md): terminal
  cancellation, idempotency, immediate KV Cache release and cancelled-row batch exclusion.
- [2026-08-21-expanded-inference-matrix.md](2026-08-21-expanded-inference-matrix.md):
  named short/long-context suites, batch efficiency and per-request/device-memory metrics.
- [2026-08-21-serving-last-logit-prefill.md](2026-08-21-serving-last-logit-prefill.md):
  explicit full/last logits semantics, matched PyTorch path and removal of historical-token output projection.
- [2026-08-21-folded-gqa-discard.md](2026-08-21-folded-gqa-discard.md): faster and
  smaller K/V-free grouped layout rejected by official complete-logit error.
- [2026-08-21-register-softmax.md](2026-08-21-register-softmax.md): bit-identical
  register-cached exponentials, alternating binary evidence and no-spill profile.
- [2026-08-21-readable-fused-attention-discard.md](2026-08-21-readable-fused-attention-discard.md):
  no-drop-in-FMHA inventory and a 0.360x readable fused route rejection.
- [2026-08-21-inplace-causal-softmax.md](2026-08-21-inplace-causal-softmax.md):
  internal score/probability Storage reuse with exact T² peak reduction and bit-exact logits.
- [2026-08-21-stop-token-early-completion.md](2026-08-21-stop-token-early-completion.md):
  explicit stop IDs, variable batch row lengths, completion reasons and immediate B1 Cache release.
- [2026-08-21-kv-cache-clear-row.md](2026-08-21-kv-cache-clear-row.md): device-native
  full-capacity row clearing, other-row preservation and explicit shared-position boundary.
- [2026-08-21-kv-cache-per-row-positions.md](2026-08-21-kv-cache-per-row-positions.md):
  per-row position metadata, strict ambiguous-read failure and reset/advance transitions.
- [2026-08-21-inference-shape-memory-matrix.md](2026-08-21-inference-shape-memory-matrix.md):
  boundary contexts, output-length sweeps, batch efficiency and strict KV/peak-memory accounting.
- [2026-08-21-deepseek-steady-profile-d2h-discard.md](2026-08-21-deepseek-steady-profile-d2h-discard.md):
  T2048 hotspot trace and a device token-history candidate rejected by B8 allocator regression.
- [2026-08-21-immediate-default-stream-pool.md](2026-08-21-immediate-default-stream-pool.md):
  phase-independent exact-size reuse with strict default/non-default Stream safety gates.
- [2026-08-21-bf16x2-key-load-discard.md](2026-08-21-bf16x2-key-load-discard.md):
  small-test pass rejected by official T2048 complete-logit and token failures.
- [2026-08-21-raw-packed-key-load-discard.md](2026-08-21-raw-packed-key-load-discard.md):
  public-scalar reconstruction reproduces the same official failure and closes local pair loads.
- [2026-08-21-device-token-history.md](2026-08-21-device-token-history.md):
  caller-owned argmax outputs and one final greedy-history D2H after allocator stabilization.
- [2026-08-21-normalize-cached-probabilities-discard.md](2026-08-21-normalize-cached-probabilities-discard.md):
  bit-exact shared normalization rejected by neutral-negative alternating performance.
- [2026-08-21-bf16-paired-value-load-discard.md](2026-08-21-bf16-paired-value-load-discard.md):
  bit-exact paired Value accumulation rejected and local scalar Attention search closed.
- [2026-08-21-official-continuous-serving.md](2026-08-21-official-continuous-serving.md):
  official Qwen/DeepSeek short/long 2/4-slot serving, exact request-bound KV bytes, memory and token gates.
- [2026-08-21-fixed-request-slot-sweep.md](2026-08-21-fixed-request-slot-sweep.md):
  fair 1/2/4/8-slot efficiency, full-row Storage recycle fix and DeepSeek cross-slot failure.
- [2026-08-21-deepseek-prefill-divergence.md](2026-08-21-deepseek-prefill-divergence.md):
  top-2 margin diagnostics, prefill-only counterfactual and PyTorch-based no-rollback decision.
- [2026-08-21-b2-prefill-row-audit.md](2026-08-21-b2-prefill-row-audit.md):
  explicit prompt offsets, swapped/duplicate B2 rows and row-copy hypothesis rejection.
- [2026-08-21-prefill-layer-drift.md](2026-08-21-prefill-layer-drift.md):
  complete values for embedding, 28 blocks, final norm and official-model logits.
- [2026-08-21-block0-drift.md](2026-08-21-block0-drift.md):
  block-zero norm/QKV/RoPE/Attention/residual/FFN split and first-drift isolation.
- [2026-08-21-bf16-ffn-drift.md](2026-08-21-bf16-ffn-drift.md):
  low-precision trace repair and gate/up/SwiGLU/down drift isolation.
- [2026-08-21-bf16-algorithm-inventory.md](2026-08-21-bf16-algorithm-inventory.md):
  M32/M64 solution-index, workspace and intersection evidence.
- [2026-08-22-bf16-same-algorithm.md](2026-08-22-bf16-same-algorithm.md):
  version-local strict registry, exact logits and no-trace performance cost.
- [2026-08-22-length-bucketed-kv-cache.md](2026-08-22-length-bucketed-kv-cache.md):
  shared-weight fixed-capacity pools, CPU/HIP/CLI gates and the measured Cache/TTFT/throughput tradeoff.
- [2026-08-22-bucket-pareto.md](2026-08-22-bucket-pareto.md): idle-gated 1/2/4-bucket
  sweep, contaminated-window rejection and the current two-B4 balanced point.
- [2026-08-22-arrival-skew-infrastructure.md](2026-08-22-arrival-skew-infrastructure.md):
  logical arrivals, focus tail latency, skew suites and a real post-process idle-gate rejection.
- [2026-08-23-traffic-skew-tail.md](2026-08-23-traffic-skew-tail.md): clean 36-process
  skew matrix, median/tail split and the bounded compatible-overflow follow-up contract.
- [2026-08-23-compatible-overflow.md](2026-08-23-compatible-overflow.md): route-contract
  failure, pending double-count fix and the formal short-heavy P95 recovery.
- [2026-08-23-slot-ratio-sweep.md](2026-08-23-slot-ratio-sweep.md): 2:6/4:4/6:2
  matrix, workload-matched static optima and the dynamic-capacity handoff.
- [2026-08-23-mi300-precision-roofline.md](2026-08-23-mi300-precision-roofline.md):
  20 executed dtype/size rows, achieved TFLOPS and FP8 peak-utilization boundary.
- [2026-08-23-large-precision-roofline.md](2026-08-23-large-precision-roofline.md):
  explicit FP32 GPU reference and 2048/4096 FP8 speedup/peak-utilization evidence.
- [2026-08-23-int8-executed-probe.md](2026-08-23-int8-executed-probe.md):
  raw hipBLASLt INT8 execution, exact sample gates and explicit non-model boundary.
- [2026-08-23-official-fp8-static-scale.md](2026-08-23-official-fp8-static-scale.md):
  single-representation official weights, unsupported-shape fallback and four precision failures.
- [2026-08-23-fp8-global-scale-grid.md](2026-08-23-fp8-global-scale-grid.md):
  fixed-before-run official scale grid, complete-logit selection and an explicit upper-boundary gap.
- [2026-08-23-fp8-scale-boundary.md](2026-08-23-fp8-scale-boundary.md):
  one-dimensional boundary expansion that improves RMS without passing the model gate.
- [2026-08-23-fp8-scale-turn.md](2026-08-23-fp8-scale-turn.md):
  DeepSeek error turn, Qwen open boundary and a top-token rejection counterexample.
- [2026-08-23-qwen-fp8-scale-closure.md](2026-08-23-qwen-fp8-scale-closure.md):
  finite-search boundary, diminishing returns and the handoff to per-Tensor weight scales.
- [2026-08-23-fp8-tensor-amax-weight.md](2026-08-23-fp8-tensor-amax-weight.md):
  explicit API, transactional preparation, one-time scan and zero-transfer prepared hot path.
- [2026-08-23-fp8-activation-range.md](2026-08-23-fp8-activation-range.md):
  all-layer Linear-input ranges, a rejected missing-boundary trace and the device-amax handoff.
- [2026-08-23-fp8-device-activation-amax.md](2026-08-23-fp8-device-activation-amax.md):
  host-optional scaled Tensor contract, dynamic quantize/dequantize and zero-transfer model path.
- [2026-08-23-fp8-activation-row-range.md](2026-08-23-fp8-activation-row-range.md):
  filtered full-value diagnostics and evidence for FFN-only row-scale design.
- [2026-08-23-hipblaslt-outer-vector-scale.md](2026-08-23-hipblaslt-outer-vector-scale.md):
  version-local native row-scale API and row-major/column-major descriptor mapping.
- [2026-08-23-installable-cmake-package.md](2026-08-23-installable-cmake-package.md):
  relocatable and build-tree package configs, external CPU/HIP/RCCL consumer gates,
  and isolation from repository-only compiler flags.
- [2026-08-23-fp8-ffn-outer-row-policy.md](2026-08-23-fp8-ffn-outer-row-policy.md):
  evidence-routed FFN-only model policy and explicit runtime fallback counters.
- [2026-08-23-fp8-device-weight-amax.md](2026-08-23-fp8-device-weight-amax.md):
  zero-D2H weight preparation contract and separated host/device scan evidence.
- [2026-08-23-hf-cli-binary-contract.md](2026-08-23-hf-cli-binary-contract.md):
  stale binary detection, fresh-build compile failure and repaired CLI evidence gate.
- [2026-08-23-fp8-multiblock-amax.md](2026-08-23-fp8-multiblock-amax.md):
  two-stage device reduction, late-partition maximum test and zero-transfer contract.
- [2026-08-23-fp8-shared-activation-quantization.md](2026-08-23-fp8-shared-activation-quantization.md):
  caller-owned ScaledTensor reuse across QKV/gate-up and machine call counters.
- [2026-08-23-fp8-selective-fp32-blocks.md](2026-08-23-fp8-selective-fp32-blocks.md):
  validated mixed block construction and official counterfactual API.
- [2026-08-23-fp8-error-source-diagnostics.md](2026-08-23-fp8-error-source-diagnostics.md):
  weight-only/activation-only inference counterfactuals, CLI schema and CPU/HIP gates.
- [2026-08-23-fp8-both-roundtrip-diagnostic.md](2026-08-23-fp8-both-roundtrip-diagnostic.md):
  both-operands FP8 rounding with FP32 GEMM for native-GEMM attribution.
- [2026-08-23-fp8-native-roundtrip-runner.md](2026-08-23-fp8-native-roundtrip-runner.md):
  direct full/both-roundtrip/FP32 complete-logit comparison and order rotation.
- [2026-08-23-fp8-output-column-scale-operator.md](2026-08-23-fp8-output-column-scale-operator.md):
  device per-column weight quantization and native scalar-GEMM post-scale algebra.
- [2026-08-23-fp8-output-channel-model-policy.md](2026-08-23-fp8-output-channel-model-policy.md):
  single-representation model preparation, CLI counters and CPU/HIP hot-path gates.
- [2026-08-23-fp8-output-column-native-probe.md](2026-08-23-fp8-output-column-native-probe.md):
  cached A-side outer-vector capability probe and scalar post-scale fallback.
- [2026-08-23-fp8-weight-reconstruction-audit.md](2026-08-23-fp8-weight-reconstruction-audit.md):
  external scalar/column weight-error audit and family aggregation contract.
- [2026-08-23-fp8-output-head-only-scope.md](2026-08-23-fp8-output-head-only-scope.md):
  rejected tied/untied routing experiment and subsequent public-API removal.
- [2026-08-23-fp8-attention-only-scope.md](2026-08-23-fp8-attention-only-scope.md):
  rejected Q/K/V/O routing experiment and removal after O-only dominance.
- [2026-08-23-fp8-attention-output-scope.md](2026-08-23-fp8-attention-output-scope.md):
  O-projection-only routing and long-context counterfactual contract.
- [2026-08-23-fp8-clipped-dynamic-quantization.md](2026-08-23-fp8-clipped-dynamic-quantization.md):
  explicit amax fraction, finite saturation and E4M3/E5M2 format maxima.
- [2026-08-23-fp8-clipped-activation-model.md](2026-08-23-fp8-clipped-activation-model.md):
  rejected ModelConfig/CLI fraction and removal after coarse/fine grids.
- [2026-08-23-fp8-fraction-pilot-runner.md](2026-08-23-fp8-fraction-pilot-runner.md):
  archived single-oracle pilot and removal after the search direction closed.
- [2026-08-23-fp8-mixed-e5-activation-probe.md](2026-08-23-fp8-mixed-e5-activation-probe.md):
  native E5M2-activation/E4M3-weight execution and explicit fallback counters.
- [2026-08-23-fp8-e5-activation-model.md](2026-08-23-fp8-e5-activation-model.md):
  ModelConfig/CLI format, mixed autograd and CPU/HIP Transformer gates.
- [2026-08-21-divergent-row-cache-reference.md](2026-08-21-divergent-row-cache-reference.md):
  unequal-position shared-Storage B1 oracle with CPU/HIP and reset-prefix gates.
- [2026-08-24-unique-gradient-inplace-add.md](2026-08-24-unique-gradient-inplace-add.md):
  exclusive-owner accumulation, real allocation savings and a default-off model rebuttal.
- [2026-08-24-hip-graph-runtime.md](2026-08-24-hip-graph-runtime.md):
  explicit-Stream capture/replay, submission crossover and honest model-readiness blockers.
- [2026-08-24-hip-graph-gemm.md](2026-08-24-hip-graph-gemm.md):
  caller-owned hipBLASLt output, capture conformance and repeated-GEMM rejection.
- [2026-08-24-scoped-model-stream-discard.md](2026-08-24-scoped-model-stream-discard.md):
  complete-logit corruption from routing model Kernels without temporary Storage lifetime.
- [2026-08-24-deferred-hip-deallocation.md](2026-08-24-deferred-hip-deallocation.md):
  explicit lifetime queue, overflow safety, synchronization reduction and pending-byte cost.
- [2026-08-24-scoped-deferred-model-stream.md](2026-08-24-scoped-deferred-model-stream.md):
  bit-exact model-wide Stream/lifetime routing, official matrix rejection and allocator handoff.
- [2026-08-24-per-device-hipblaslt-handles.md](2026-08-24-per-device-hipblaslt-handles.md):
  restores rank-local vendor handle ownership and all two-rank model correctness gates.
- [2026-08-24-stream-ordered-allocator.md](2026-08-24-stream-ordered-allocator.md):
  explicit HIP async allocation/Graph conformance and measured policy rejection.
- [2026-08-24-activation-arena.md](2026-08-24-activation-arena.md):
  stable backing, aligned two-slot liveness and allocation-free Graph replay.
- [2026-08-24-arena-ffn.md](2026-08-24-arena-ffn.md):
  external Storage and the first official-shape heterogeneous FFN Graph region.
- [2026-08-24-bf16-arena-ffn.md](2026-08-24-bf16-arena-ffn.md):
  caller-owned BF16 workspace, explicit fallback and official shape matrix.
- [2026-08-24-bf16-ffn-arena-model.md](2026-08-24-bf16-ffn-arena-model.md):
  one workspace shared across blocks and the complete-model universal-policy rejection.
- [2026-08-24-bf16-ffn-arena-selective.md](2026-08-24-bf16-ffn-arena-selective.md):
  rows≥512 selection, exact short-path bypass and two-model long-prefill keep.
- [2026-08-24-bf16-qkv-arena-discard.md](2026-08-24-bf16-qkv-arena-discard.md):
  caller-owned QKV, allocation reduction and complete-model performance rejection.
- [2026-08-24-allocation-source-attribution.md](2026-08-24-allocation-source-attribution.md):
  thread-local source×size diagnostics and deterministic T512 target selection.
- [2026-08-24-attention-core-arena-discard.md](2026-08-24-attention-core-arena-discard.md):
  exact core liveness, caller-owned Attention and model-level rejection.
- [2026-08-24-fp32-attention-solutions.md](2026-08-24-fp32-attention-solutions.md):
  four exact QK/PV inventories with complete-output-before-timing selection.
- [2026-08-24-fp32-attention-model-gate.md](2026-08-24-fp32-attention-model-gate.md):
  exact versioned registry, accumulated-error counterexample and 24-process default rejection.
- [2026-08-24-bf16-grouped-qkv.md](2026-08-24-bf16-grouped-qkv.md):
  phase-delta target selection, pointer-stable grouped plans and two-model rejection.
- [2026-08-24-bf16-grouped-qkv-expanded.md](2026-08-24-bf16-grouped-qkv-expanded.md):
  64-candidate recovery, device user arguments, steady keep and setup-gate rejection.
- [2026-08-24-bf16-grouped-qkv-prewarm.md](2026-08-24-bf16-grouped-qkv-prewarm.md):
  explicit model prewarm, first-request timing and admission lifecycle boundary.
- [2026-08-24-hipblaslt-preload.md](2026-08-24-hipblaslt-preload.md):
  beginner-friendly cold-start explanation and the rejected all-kernel preload shortcut.
- [2026-08-24-bf16-exact-startup.md](2026-08-24-bf16-exact-startup.md):
  beginner-friendly explanation of why a faster local GEMM does not make startup or the model fast.
- [2026-08-24-bf16-grouped-gate-up.md](2026-08-24-bf16-grouped-gate-up.md):
  beginner-friendly grouped gate/up capability and stable-address requirement.
- [2026-08-24-bf16-grouped-gate-up-model.md](2026-08-24-bf16-grouped-gate-up-model.md):
  beginner-friendly shared-kernel/per-block-plan integration and official model result.
- [2026-08-24-bf16-grouped-composition.md](2026-08-24-bf16-grouped-composition.md):
  beginner-friendly four-policy proof that the two grouped registries compose safely.
- [2026-08-24-bf16-grouped-shape-matrix.md](2026-08-24-bf16-grouped-shape-matrix.md):
  beginner-friendly rows256/1024 capability and single-process counterexample correction.
- [2026-08-24-bf16-grouped-shape-models.md](2026-08-24-bf16-grouped-shape-models.md):
  beginner-friendly sequence/batch model gate and real CLI batch-logit fix.
- [2026-08-24-bf16-grouped-composed-profile.md](2026-08-24-bf16-grouped-composed-profile.md):
  beginner-friendly post-composition hotspot and exact submission accounting.
- [2026-08-24-hf-strided-copy-sources.md](2026-08-24-hf-strided-copy-sources.md):
  beginner-friendly exact source attribution for every remaining layout copy.
- [2026-08-24-inference-bthd-attention.md](2026-08-24-inference-bthd-attention.md):
  beginner-friendly copy-free inference Attention island and fallback domain.
- [2026-08-24-inference-bthd-shape-models.md](2026-08-24-inference-bthd-shape-models.md):
  beginner-friendly sequence/batch extension and source-aware residual copy boundary.
- [2026-08-24-inference-bthd-profile.md](2026-08-24-inference-bthd-profile.md):
  beginner-friendly post-BTHD trace and next cast-boundary selection.
- [2026-08-24-bf16-adamw-moments.md](2026-08-24-bf16-adamw-moments.md): opt-in
  BF16 AdamW state, checkpoint v2 compatibility, corrected timing and official-model evidence.
- [2026-08-24-hybrid-bf16-adamw.md](2026-08-24-hybrid-bf16-adamw.md): thresholded
  small-Tensor merge, 16M counterexample and retained 1M Auto policy.
- [2026-08-24-post-hybrid-training-profile.md](2026-08-24-post-hybrid-training-profile.md):
  load-subtracted proof that GEMM is the next training architecture boundary.
- [2026-08-24-grouped-weight-gradient-discard.md](2026-08-24-grouped-weight-gradient-discard.md):
  eight-case FP32 GroupedGemm capability failure before any Autograd route.
- [2026-08-24-packed-weight-gradient-discard.md](2026-08-24-packed-weight-gradient-discard.md):
  complete-output proof that D2D pack plus one large GEMM is slower in all four cases.
- [2026-08-24-fp32-weight-gradient-solutions.md](2026-08-24-fp32-weight-gradient-solutions.md):
  rank-2 exact registry, stable operator winners and end-to-end rejection.
- [2026-08-24-training-graph-capture-boundary.md](2026-08-24-training-graph-capture-boundary.md):
  beginner-friendly staged capture, allocation-safe recovery and optimizer host-state boundary.
- [2026-08-24-adamw-device-step-graph.md](2026-08-24-adamw-device-step-graph.md):
  beginner-friendly device step ownership, checkpoint synchronization and measured Graph boundary.
- [2026-08-24-adamw-stable-descriptor-multi-graph.md](2026-08-24-adamw-stable-descriptor-multi-graph.md):
  beginner-friendly immutable pointer table, two-node replay and real-gradient address blocker.
- [2026-08-24-gradient-address-stability.md](2026-08-24-gradient-address-stability.md):
  beginner-friendly shape-versus-address explanation and model/context-specific evidence.
- [2026-08-24-optimizer-graph-model-preflight.md](2026-08-24-optimizer-graph-model-preflight.md):
  beginner-friendly Stream/allocator conflict, snapshot safety gate and zero-launch rejection.
- [2026-08-24-quiescent-allocator-handoff.md](2026-08-24-quiescent-allocator-handoff.md):
  beginner-friendly device-wide completion proof and reversible default/Graph phases.
- [2026-08-24-optimizer-graph-model-gate.md](2026-08-24-optimizer-graph-model-gate.md):
  beginner-friendly explanation of why fewer submissions can still lose on real model shapes.
- [2026-08-24-cmake-c-only-consumer.md](2026-08-24-cmake-c-only-consumer.md):
  proves that both build-tree and relocated install-tree Config packages work from a genuinely C-only project.
- [2026-08-25-rocwmma-qk-tile.md](2026-08-25-rocwmma-qk-tile.md):
  beginner-friendly matrix-core capability, complete-output matrix and long-context counterexample.
- [2026-08-25-rocwmma-online-attention.md](2026-08-25-rocwmma-online-attention.md):
  beginner-friendly online softmax, MFMA QK/PV, caught race and real GQA evidence.
- [2026-08-25-rocwmma-online-operator.md](2026-08-25-rocwmma-online-operator.md):
  beginner-friendly public contract, exact routing counters, batch and fallback evidence.
- [2026-08-25-rocwmma-online-model-discard.md](2026-08-25-rocwmma-online-model-discard.md):
  beginner-friendly full-model counterexample, complete logits, memory and cast-cost explanation.
- [2026-08-25-rocwmma-direct-bf16-model-discard.md](2026-08-25-rocwmma-direct-bf16-model-discard.md):
  beginner-friendly rebuttal that removes all three casts and closes the model track.
- [2026-08-25-current-inference-profile.md](2026-08-25-current-inference-profile.md):
  beginner-friendly load-subtracted reprofile and next-hotspot selection.
- [2026-08-25-fp32-attention-t1024-discard.md](2026-08-25-fp32-attention-t1024-discard.md):
  beginner-friendly descriptor mismatch and cross-model exact-solution rejection.
- [2026-08-25-bf16-swiglu-vector-discard.md](2026-08-25-bf16-swiglu-vector-discard.md):
  beginner-friendly operator-versus-model speedup counterexample.
- [2026-08-25-bf16-grouped-swish-discard.md](2026-08-25-bf16-grouped-swish-discard.md):
  beginner-friendly grouped epilogue capability and full-model rejection.
- [2026-08-25-bf16-rms-norm-output.md](2026-08-25-bf16-rms-norm-output.md):
  beginner-friendly direct-BF16 final store and corrected GPU reference.
- [2026-08-25-bf16-ffn-norm-model.md](2026-08-25-bf16-ffn-norm-model.md):
  beginner-friendly default route, exact allocation reduction and fallback fix.
- [2026-08-25-post-bf16-ffn-norm-profile.md](2026-08-25-post-bf16-ffn-norm-profile.md):
  beginner-friendly load-subtracted map after the retained default change.
- [2026-08-25-bf16-attention-norm-model.md](2026-08-25-bf16-attention-norm-model.md):
  beginner-friendly Attention/QKV Arena route and exact peak reduction.
- [2026-08-25-post-bf16-attention-norm-profile.md](2026-08-25-post-bf16-attention-norm-profile.md):
  beginner-friendly map of the final per-layer cast pair.
- [2026-08-25-bf16-pv-output-discard.md](2026-08-25-bf16-pv-output-discard.md):
  beginner-friendly backend capability rejection before timing.
- [2026-08-25-bf16-value-pv-discard.md](2026-08-25-bf16-value-pv-discard.md):
  closes the opposite mixed-dtype P×V boundary with the same evidence gate.
- [2026-08-25-inference-local-saturation.md](2026-08-25-inference-local-saturation.md):
  explains the measured local-search ceiling without confusing it with the whole roadmap.
- [2026-08-25-current-training-profile-runner.md](2026-08-25-current-training-profile-runner.md):
  pins the retained B1T512 training profile contract in one reproducible command.
- [2026-08-25-current-training-profile.md](2026-08-25-current-training-profile.md):
  records the four-process current training map and unchanged hotspot order.
- [2026-08-25-bf16-weight-gradient-benchmark.md](2026-08-25-bf16-weight-gradient-benchmark.md):
  defines a cast-inclusive low-precision weight-gradient operator gate before graph integration.
- [2026-08-25-bf16-weight-gradient-shapes.md](2026-08-25-bf16-weight-gradient-shapes.md):
  records four rejected shapes, two admitted shapes and the default-off Autograd boundary.
- [2026-08-25-bf16-weight-gradient-model.md](2026-08-25-bf16-weight-gradient-model.md):
  records the wiring rebuttal and the passing short official-model gate.
- [2026-08-25-training-trajectory-evidence.md](2026-08-25-training-trajectory-evidence.md):
  defines timed loss export and complete temporary gate/up parameter comparison.
- [2026-08-25-bf16-weight-gradient-trajectory-discard.md](2026-08-25-bf16-weight-gradient-trajectory-discard.md):
  records the long-run rebuttal, complete parameter failure and candidate cleanup.
- [2026-08-25-bf16-weight-gradient-allocation-attribution.md](2026-08-25-bf16-weight-gradient-allocation-attribution.md):
  proves the exact two-cast allocation identity without restoring the rejected route.
- [2026-08-25-bf16-weight-gradient-workspace-discard.md](2026-08-25-bf16-weight-gradient-workspace-discard.md):
  rejects a public workspace after separate Event and wall gates.
- [2026-08-25-training-local-saturation.md](2026-08-25-training-local-saturation.md):
  closes local training policy retuning and selects the next architecture scale.
- [2026-08-25-current-data-parallel-audit.md](2026-08-25-current-data-parallel-audit.md):
  separates current two-GPU training, communication and host verification costs.
- [2026-08-25-data-parallel-verification-interval.md](2026-08-25-data-parallel-verification-interval.md):
  makes every-step, sparse and disabled host parameter audits explicit.
- [2026-08-25-data-parallel-verification-matrix.md](2026-08-25-data-parallel-verification-matrix.md):
  records rotated performance, exact losses and the explicit optimizer completion fix.
- [2026-08-25-data-parallel-bucket-matrix-runner.md](2026-08-25-data-parallel-bucket-matrix-runner.md):
  fixes a final-step-audited real bucket-count sweep before overlap work.
- [2026-08-25-data-parallel-bucket-matrix.md](2026-08-25-data-parallel-bucket-matrix.md):
  rejects artificial tiny-model overlap and hands off to Model-S.
- [2026-08-25-model-s-data-parallel-workload.md](2026-08-25-model-s-data-parallel-workload.md):
  adds the first natural multi-bucket model workload and measured memory/stage evidence.
- [2026-08-25-model-s-bucket-matrix-runner.md](2026-08-25-model-s-bucket-matrix-runner.md):
  pins the natural 1/4/25 MiB reducer-baseline matrix.
- [2026-08-25-data-parallel-model-s-buckets.md](2026-08-25-data-parallel-model-s-buckets.md):
  selects the 3-bucket baseline and records its memory tradeoff.
- [2026-08-25-data-parallel-bucket-copy-stats.md](2026-08-25-data-parallel-bucket-copy-stats.md):
  adds exact pack/unpack/average temporary and backend-allocation statistics.
- [2026-08-25-data-parallel-bucket-copy-attribution.md](2026-08-25-data-parallel-bucket-copy-attribution.md):
  admits persistent reducer work with an exact Model-S allocation identity.
- [2026-08-25-data-parallel-inplace-average.md](2026-08-25-data-parallel-inplace-average.md):
  adds address-stable averaging and a same-binary Model-S gate.
- [2026-08-25-data-parallel-inplace-average-result.md](2026-08-25-data-parallel-inplace-average-result.md):
  records the retained 1.107x Model-S reducer improvement.
- [2026-08-26-cmake-config-public-gate.md](2026-08-26-cmake-config-public-gate.md):
  exposes one copy-paste CTest preset for the complete build/install/relocation/consumer package contract.
- [2026-08-26-int8-weight-contract.md](2026-08-26-int8-weight-contract.md):
  records the one-byte Tensor, explicit scale, CPU/HIP/PyTorch and official-safetensors boundary.
- [2026-08-26-int8-weight-matmul-baseline.md](2026-08-26-int8-weight-matmul-baseline.md):
  fixes the complete-output explicit-dequantize Linear control before performance work.
- [2026-08-26-int8-fused-decode.md](2026-08-26-int8-fused-decode.md):
  keeps an explicit memory-first M=1 route while preserving the resident-GEMM counterexample.
- [2026-08-26-model-s-int8-inference.md](2026-08-26-model-s-int8-inference.md):
  records transactional whole-model preparation, decode gain and the prefill counterexample.
- [2026-08-26-official-int8-device-amax.md](2026-08-26-official-int8-device-amax.md):
  keeps zero-D2H preparation while rejecting official Qwen on complete logits and tokens.
- [2026-08-26-official-int8-column-scale.md](2026-08-26-official-int8-column-scale.md):
  retains column primitives but closes the current official weight-only INT8 precision line.
