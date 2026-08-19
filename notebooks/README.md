# microLLM-rocm course notebooks

These Markdown notebooks are executable guides over the same engine source. They do
not contain a second teaching-only implementation.

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

Each notebook starts from a concrete old method, adds one failure condition, defines
a task contract before implementation, and retains at least one limitation or failure.
