# Exclusive gradient accumulation experiment

## Why this node exists

The current T512 profile still contains dense gradient additions. A generic earlier
copy-on-write idea had no measured owner evidence. New diagnostics now separate the target
operation, shape, contribution source and Storage owner count, making one narrow experiment
possible.

## Implemented boundary

- `ops::add_in_place_` writes an FP32 contiguous destination on CPU or HIP;
- shape, dtype, device, contiguity and partial-overlap errors are explicit;
- Autograd only selects it when the node is the sole persistent Storage owner;
- shared gradients fall back to allocating `ops::add`;
- candidate and executed calls/elements are separately reported;
- the CLI exposes a default-off boolean for same-binary experiments;
- the installed CMake package consumer links the new API.

## Measured outcome

Qwen and DeepSeek execute all 72/84 eligible additions and save 144/168 allocation calls
over two measured steps. Peak is unchanged, parameters pass, and throughput is only
`1.0042×/0.9952×`, below the declared two-model `1.01×` gate.

rocprofv3 shows the reason: across three Qwen steps, 216 engine allocation/cache-reuse
events disappear, while backend allocation calls, HIP allocation/free calls, total Kernel
launches and add Kernel launches are unchanged. Therefore the policy defaults false. The
safe primitive and diagnostics stay available for larger liveness/buffer-planning work.

Full report: [Experiment 172](../optimization-log/experiments/172-unique-gradient-inplace-add-discard.md).
