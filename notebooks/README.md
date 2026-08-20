# microLLM-rocm course notebooks

These Markdown notebooks are executable guides for the engine on the repository's
`main` branch. This course branch intentionally contains no engine copy. Before running
commands, set `MICROLLM_ENGINE_DIR` to an absolute path of a separate `main` checkout as
described in the [course README](../README.md).

| Notebook | Engine evidence |
|---|---|
| [N0](N0_storage_tensor.md) | Storage, Tensor, view, PPM, sanitizer |
| [N1](N1_cpu_hip.md) | HIP runtime, Stream/Event, CPU/HIP conformance |
| [N2](N2_autograd.md) | eager reverse mode and finite differences |
| [N3](N3_checkpoint.md) | complete-state resume equivalence |
| [N4](N4_transformer.md) | MHA/GQA Transformer and tiny overfit |
| [N5](N5_training_generation.md) | Model-S, generation, evaluation boundary |
| [N6](N6_performance.md) | Event timing, rocprofv3, hipBLASLt |
| [N7](N7_multi_gpu.md) | RCCL equivalence, buckets, overlap, failure |
| [N8](N8_evidence_atlas.md) | unified pass/fail/unverified evidence atlas |
| [N9](N9_huggingface_models.md) | official Qwen/DeepSeek config, tokenizer, weights, logits, train step |
| [N10](N10_low_precision.md) | FP16/BF16/FP8 storage, scaled GEMM, accuracy and speed boundaries |

Each notebook starts from a concrete old method, adds one failure condition, defines
a task contract before implementation, and retains at least one limitation or failure.

N0–N8 form the original “one allocation to multi-GPU” path. N9 and N10 are extension
lessons that use the same engine to cross from a teaching model to official external
weights and then to measured low-precision execution.

Commands use CMake presets from `main`:

```bash
cmake --build "$MICROLLM_ENGINE_DIR/build/cpu-debug" --parallel
cmake --build "$MICROLLM_ENGINE_DIR/build/hip-release" --parallel
cmake --build "$MICROLLM_ENGINE_DIR/build/rccl-release" --parallel
```

Run them from the engine checkout or use the absolute paths shown in each notebook.
