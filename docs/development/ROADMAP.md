# Development roadmap

The roadmap follows engine dependencies. Course chapters consume milestone tags;
they do not determine source-module boundaries.

| Milestone | Engine result | Evidence gate |
|---|---|---|
| M0 / N0 | CPU build, Storage, Tensor/View, PPM | unit, random shape, ASan/UBSan, checksum |
| M1 / N1 | HIP runtime, copy, add, matmul | CPU/HIP conformance, async failure timeline |
| M2 / N2–N3 | CPU autograd and resumable checkpoint | finite difference and resume trajectory |
| M3 / N4–N5 | Model-S train, SFT, generate, KV cache | overfit, train/val, cached logits |
| M4 | C/Python/PyTorch bindings and Model-M | zero-copy/stream integration tests |
| M5 / N6 | profiling, tuned ops, autotune prototype | raw traces and end-to-end regression |
| M6 / N7 | RCCL single-node 2/4 GPU | single/multi equivalence and failure handling |
| M7 / N8 | unified evidence and course release | reproducible report and failure atlas |

## Six-month delivery rhythm

1. Month 1: M0 plus HIP runtime and three end-to-end operators.
2. Month 2: complete Model-S forward operator set and micro-benchmarks.
3. Month 3: autograd, optimizer, checkpoint, tokenizer, Model-S training.
4. Month 4: bindings, Radeon validation, Model-M.
5. Month 5: synchronous RCCL correctness and distributed failure handling.
6. Month 6: overlap, complete benchmarks, compatibility guide, and course.

Every merged milestone updates `STATUS.md` from evidence. Planned code or an empty
directory does not advance status.
