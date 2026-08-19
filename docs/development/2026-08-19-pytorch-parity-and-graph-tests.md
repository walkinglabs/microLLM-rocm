# 2026-08-19 — PyTorch parity and dedicated graph tests

## Goal

Turn “the implementation looks similar” into explicit evidence that a new learner can
follow and a future contributor cannot accidentally bypass.

## Design documentation

- `docs/DESIGN_FOR_BEGINNERS.zh-CN.md` explains Storage, Tensor, stride, device,
  Stream/Event, CPU/HIP operators, graph construction, backward, Transformer, training,
  checkpoint, KV cache, and RCCL using small examples and everyday language.
- `docs/OPERATOR_CONTRACTS.zh-CN.md` fixes input/output shapes, errors, PyTorch oracle,
  and FP32 tolerances for every current forward/backward math operator.

## Dedicated graph test boundary

`tests/graph/` is separate from operator smoke tests:

- `graph_construction_test.cpp` checks operation names, parent edges, shared nodes,
  topological ordering, `requires_grad` pruning, root shape, and every public graph op;
- `graph_gradient_alignment_test.cpp` checks hand-valued composite graphs, fork
  accumulation, repeated backward, view logical order, and seed-shape errors;
- `hip_graph_alignment_test.cpp` compares CPU/HIP topology, loss, and every named
  Transformer parameter gradient while requiring zero host/device transfers during the
  GPU graph.

The graph engine now exposes a read-only `GraphSnapshot`. Snapshot node IDs are in
topological order, so tests and teaching tools can inspect a graph without mutating it.

## PyTorch oracle

`tests/torch/operator_oracle.cpp` executes deterministic microLLM cases. The independent
`python/tests/test_operator_parity.py` rebuilds those forward and backward graphs using
PyTorch APIs and autograd, then checks values, shapes, and fixed tolerances.

The gate includes:

- all public forward and backward math operators;
- 24 invalid shape/dtype contracts;
- reshape/transpose/contiguous graph gradients;
- SGD parameter parity;
- two AdamW steps, parameters, first moments, and second moments;
- one-layer GQA Transformer logits, loss, and every named parameter gradient.

## Coverage enforcement

`tests/coverage_manifest.json` maps the public API to PyTorch and shape cases.
`scripts/audit_test_coverage.py` compares it with `ops.h`, `autograd.h`, every discovered
native/Python test file, and CMake/CTest registration. Current audit result:

```text
tensor APIs = 30
graph and Value APIs = 29
registered test files = 25
```

Adding a public API without a gate, or adding a test file without registration, fails
`Coverage.OperatorAndTestFiles`.

## Measured gates

```text
normal CPU CTest                 99/99 pass
CPU ASan/UBSan                   97/97 pass
MI300X/gfx942 HIP                23/23 pass
two-rank RCCL                     7/7 pass
PyTorch 2.13 CPU binding/oracle   2/2 pass
```

## Honest boundary

PyTorch CPU is the direct external oracle. Each HIP primitive is independently compared
with the same CPU implementation, producing a PyTorch-to-CPU-to-HIP evidence chain.
Direct PyTorch ROCm comparison is still blocked because the available matching wheel
fails during `import torch`; it is not claimed as measured. BF16/FP16 tolerances also
remain future work because the current training contract is FP32.
