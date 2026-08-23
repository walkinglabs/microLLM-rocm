# Interleaved-head Attention P×V evidence

Experiment 163 verifies that hipBLASLt can read and write `[B,T,H,D]` without first
materializing `[B,H,T,D]`. The candidate uses matrix leading dimension `H*D` and head-batch
stride `D`.

- `raw.jsonl`: five shapes × materialized/interleaved × three fresh processes;
- `summary.json`: medians, complete-output errors and official T512 speed gates;
- `coverage-summary.json`: complete post-change CPU coverage;
- `verification.json`: cross-framework and full-suite evidence.

Every one of the 30 rows compares all output elements with the materialized GPU path.
Maximum and RMS error are both zero; timed regions report zero host payload transfers.
Qwen T512 Event/wall speedups are `1.415×/1.330×`; DeepSeek T512 reaches
`2.200×/2.136×`. This is an operator building block, not yet an end-to-end model claim.
