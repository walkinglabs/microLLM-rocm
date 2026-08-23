# Tied-embedding sparse accumulation evidence

Experiment 161 attributes Qwen/DeepSeek gradient accumulation and strided-copy layouts,
then replaces one Qwen-only dense embedding-gradient add with sparse token-row updates.

- `training.jsonl`: dense/sparse × Qwen/DeepSeek × three fresh processes;
- `summary.json`: model, diagnostics, profile and keep gates;
- `dense-kernel-stats.csv` / `sparse-kernel-stats.csv`: same T512 workload;
- `verification.json`: compact machine-readable decision.

Qwen's tied embedding/output head receives a dense `matmul_right` gradient first and an
`embedding_backward` contribution second. The old path allocates and clears a 136,134,656
element mostly-zero Tensor, then adds all 544 MB. The candidate atomically adds only the
512 token rows into the uniquely owned dense head gradient.

Peak engine memory falls by 1,055,989,760 bytes (8.11%), throughput is neutral-positive,
and DeepSeek's untied model never enters the sparse path. The explicit
`--tied-embedding-sparse-add true/false` option exists for same-binary A/B; default is true.
