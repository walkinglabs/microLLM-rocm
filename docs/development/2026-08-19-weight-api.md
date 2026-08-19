# 2026-08-19 — external weight and safetensors API

## Goal

Create the weight boundary required before real Qwen/DeepSeek work, without claiming
that file compatibility alone implements those model architectures.

## Public API

- independent named `TransformerModel::state_dict()`;
- strict atomic and non-strict reporting `load_state_dict()`;
- `load_safetensors`, `load_safetensors_files`, and sharded index loading;
- model convenience load/save methods;
- F32, BF16, and F16 file conversion to the current FP32 engine tensors;
- `WeightMapping` with identity or 2D transpose transforms;
- Qwen-style parameter name mapping for an architecturally compatible model;
- CPU or direct HIP target placement.

## Safety and correctness decisions

- strict load validates every source before changing any parameter;
- returned state dicts do not alias model storage;
- load clears gradients on replaced parameters;
- header size, JSON structure, dtype, shape, byte count, data ranges, file bounds,
  duplicate shards, missing index entries, and unsafe relative paths are checked;
- safetensors output uses deterministic map order, padded JSON headers, little-endian
  offsets, and temporary-file replacement;
- Qwen mapping does not hide unsupported Q/K/V biases, QK-Norm, explicit head width,
  MLA, MoE, or quantization parameters: strict mode reports them.

## Measured tests

```text
framework CPU                 109/109 pass
CPU ASan/UBSan                107/107 pass
MI300X/gfx942 HIP              24/24 pass
PyTorch CPU binding/oracle       2/2 pass
two-rank RCCL                    7/7 pass
coverage audit             28 test files pass
```

Focused weight tests cover independent snapshots, atomic failure, non-strict reports,
forward reproduction, Qwen naming/transposes, tied embeddings, single/multiple/indexed
files, three source dtypes, corruption, duplicate weights, unsafe index paths, direct
GPU load, and GPU model placement.

## External format interoperability follow-up

The initial round-trip used the microLLM writer and reader together. A later optional
CTest closes that shared-bug gap by exchanging F32, BF16, and F16 fixtures in both
directions with the official `safetensors` Python package. It checks keys, declared
dtype, scalar/vector/matrix shapes, and every converted value.

Measured environment and result:

```text
PyTorch       2.13.0+cpu
safetensors   0.6.2
NumPy         2.3.2
packaging     25.0
official interop CTest  1/1 pass
directions              C++ -> Python, Python -> C++
file dtypes              F32, BF16, F16
```

The gate is enabled explicitly with `MICROLLM_SAFETENSORS_PYTHON`; the ordinary build
does not download external Python dependencies.

## Honest boundary

The implementation currently materializes a complete StateDict before model copy, so
peak host/device memory is not suitable for flagship models. Streaming/memory-mapped
load, official tokenizer/config parsing, Qwen architecture differences, FP8/INT8/INT4,
and rank-local expert/tensor placement remain explicit work in `NEXT_STEPS.md`.
