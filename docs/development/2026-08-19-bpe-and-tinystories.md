# 2026-08-19 — self-contained BPE and immutable TinyStories source

## BPE

The tokenizer begins with all 256 byte tokens, counts adjacent pairs, chooses a
deterministic most-frequent pair, appends a merged piece, and rewrites the sequence.
Encoding replays learned merges in order; decoding expands stored byte pieces.

Tests prove frequent-pair compression, arbitrary byte round-trip, serialization
round-trip, invalid vocabulary rejection, and invalid merge-reference rejection. This
is a readable trainer, not an optimized tokenizer for multi-gigabyte corpora.

## Dataset source

The TinyStories Hugging Face data card records license `cdla-sharing-1.0`. The registry
pins revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` and official train/validation
filenames. `scripts/fetch_tinystories_smoke.sh` range-downloads an immutable validation
prefix; a 4096-byte loader check succeeded.

The range prefix may end within a story and is only loader/training smoke data. It is
not a validation benchmark. State advances from `planned` to `loader-ready`, not
`reference-trained`.
