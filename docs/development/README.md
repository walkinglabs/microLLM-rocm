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
