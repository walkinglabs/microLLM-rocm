# 2026-08-19 — M2 versioned checkpoint

## Contract

Persist everything needed to compute the same next training step: named parameters,
AdamW configuration and moments, optimizer step, global step, data cursor, caller-
serialized RNG state, and model/data configuration summaries. Reject corruption and
contract mismatches before mutating live parameters.

## Format

The first binary format contains:

- fixed magic and format version;
- explicit endian marker and payload size;
- lightweight payload integrity value;
- length-delimited strings and collections;
- CPU float32 Tensor shapes and values;
- named parameters in stable order;
- AdamW configuration, step, first moments, and second moments;
- experiment state fields.

Readers limit ranks, collection counts, and payload-derived lengths before allocating.

## Verification

Checkpoint-focused tests pass 3/3:

1. complete state round-trip preserves every metadata field, parameter, and optimizer
   step;
2. a restored run matches uninterrupted parameter values for the next three AdamW
   updates exactly;
3. a changed payload byte is rejected, as are duplicate parameter names.

## Boundary

The current writer targets contiguous CPU float32 training state and writes directly
to the requested path. Atomic temporary-file replacement and direct GPU Tensor
serialization remain follow-up reliability work.
