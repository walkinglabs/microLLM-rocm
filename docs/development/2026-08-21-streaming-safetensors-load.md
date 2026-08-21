# 2026-08-21 — streaming safetensors load

## Problem in plain language

The file already stores BF16 numbers in two bytes. The old loader first turned every
number into a four-byte FP32 value on the CPU, copied those larger values to the GPU, kept
the whole imported StateDict, and then made model parameter copies. Loading DeepSeek took
about 65 seconds even though one measured training step took less than a second.

## Boundary chosen

Only an uninitialized model on HIP loading one safetensors file uses the new path. This is
important:

- an uninitialized model cannot run forward, so a mid-file I/O failure cannot expose a
  partly updated usable model;
- an already initialized model keeps the old prepare-then-commit atomic replacement;
- shards and index files keep the old path until they have a global metadata preflight.

## Implementation

- `inspect_safetensors()` reads names, dtypes, shapes and byte counts without payloads;
- `visit_safetensors()` visits raw tensors in file-offset order with a bounded byte buffer;
- strict names/mappings/shapes are checked before the first H2D transfer;
- one low-precision GPU staging Tensor is sized to the largest required source;
- `cast_out_` and `cast_transpose_2d_out_` write existing FP32 parameter Storage;
- current bytes, peak bytes and load transfer counts are reported by the training CLI.

## Evidence

```text
Qwen load       17.659 s → 0.580 s   30.45×
DeepSeek load   65.100 s → 1.356 s   48.02×
DeepSeek PyTorch same-window load     2.084 s

Qwen H2D        988,065,536 bytes
DeepSeek H2D    3,554,176,000 bytes
load D2H        0 calls
```

Those H2D totals equal exactly two bytes per model parameter. DeepSeek load current bytes
equal its 7,108,352,000 FP32 weight bytes; peak is 7,575,099,392 bytes, or one largest raw
staging allocation above the weights.

The full DeepSeek four-shape, two-framework, three-process matrix also passed. Relative to
the pre-streaming loader, training throughput changed by at most 0.4% and measured training
peak memory did not change.

Raw data, graph and machine-checked contracts live in
[Experiment 050](../optimization-log/experiments/050-streaming-safetensors-load.md).
