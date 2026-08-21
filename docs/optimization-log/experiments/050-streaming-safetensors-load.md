# Experiment 050 — streaming safetensors load

## Failure

After Experiment 045 removed random initialization and host transpose, DeepSeek still took
about 65 seconds to load. The loader decoded every BF16 tensor into `vector<float>`, sent
4-byte FP32 to the GPU, retained a complete GPU StateDict, and then built a second prepared
parameter set.

## Design

The fast path is deliberately narrow: an uninitialized model on HIP loading one
safetensors file.

1. Inspect only the header and validate strict names, mappings, ranks and shapes before
   any payload transfer.
2. Visit tensors in file-offset order with one reusable host byte buffer.
3. Transfer the original BF16/F16 payload, not decoded FP32.
4. Reuse one device staging allocation per source dtype.
5. Cast identity weights directly into existing parameter Storage.
6. Fuse low-precision cast and 2D transpose into the target parameter Storage.
7. Synchronize, release staging, then mark the model initialized.

An already initialized model still uses the old prepare-then-commit StateDict path. If
streaming I/O fails midway, the uninitialized model remains unable to run forward.

## Load result

| Model | Before | After | Speedup | H2D bytes | Load peak / weights |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 17.659 s | 0.580 s | 30.45× | 0.988 GB | 1.138× |
| DeepSeek Distill 1.5B | 65.100 s | 1.356 s | 48.02× | 3.554 GB | 1.066× |

DeepSeek PyTorch's same-window median load is 2.084 seconds, so microLLM is `1.537×`
faster for this pinned single-file path. Current engine bytes after load equal exactly the
FP32 model weights; the temporary peak is bounded by the largest raw low-precision tensor.

![Streaming safetensors load](../assets/streaming-safetensors-load.svg)

## Training non-regression

The complete DeepSeek four-shape matrix was rerun. Relative to Experiment 045, throughput
ratios are `0.999×、1.001×、1.000×、0.996×`; all training peaks are identical. Losses are
finite, parameters update and optimizer payload transfer remains zero.

## Remaining boundary

The optimized path does not yet stream multiple explicit files or an index. Those APIs
retain their atomic StateDict behavior. Extending them requires a global metadata preflight
across all shards before the first write.
