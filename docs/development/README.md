# Development records

This directory is the chronological record requested for the main repository.
Each milestone records scope, decisions, commands, evidence, known failures, and
the next gate. Records are append-only except for factual corrections.

- [ROADMAP.md](ROADMAP.md): planned development sequence and acceptance gates.
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
