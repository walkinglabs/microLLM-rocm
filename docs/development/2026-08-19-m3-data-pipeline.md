# 2026-08-19 — M3 tokenizer and deterministic token batches

## Contract

Provide a dependency-free smoke tokenizer and a resumable contiguous-token dataset.
Keep the interface separate from the model so a future BPE implementation can replace
the tokenizer without changing Transformer code.

## Implementation and evidence

- byte tokenizer maps every byte to one of 256 IDs and round-trips all 256 values;
- TokenDataset emits `B x T` int32 inputs and one-token-shifted targets;
- cursor advancement is deterministic and wraps across valid start positions;
- restoring cursor into an equivalent dataset produces the same subsequent batch;
- insufficient token data, negative tokens, invalid batch size, and bad cursor fail
  explicitly.

The byte tokenizer is a smoke/reference path. Model-S dataset selection and BPE remain
separate evidence-gated work.
