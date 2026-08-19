# 2026-08-19 — course and unified evidence audit

## Delivered course surface

- N0–N8 all consume the same engine targets and executable evidence;
- PA0 includes a runnable manual-gradient counterexample;
- PA1 consumes benchmark/rocprof artifacts;
- PA2 enforces a falsifiable failure-driven proposal;
- no notebook duplicates the engine in a separate implementation.

## Unified verifier

`scripts/verify_evidence.sh` checks all notebook/PA artifacts, parses every committed
JSON/JSONL result, builds the selected tree, and runs CPU, HIP, or RCCL-labelled tests.
It makes missing course files and malformed benchmark evidence first-class failures.

GitHub CPU CI runs the sanitizer build plus the CPU evidence verifier. GPU/RCCL CI is
not declared on generic hosted runners; local hardware commands remain explicit.

## Still unverified external deliverables

- PyTorch ROCm Custom Op runtime because the matching temporary wheel fails on import;
- real-corpus Model-S pretraining and SFT because the dataset registry is planned;
- Radeon compatibility because no Radeon device is available;
- four-rank RCCL because the current container shared-memory limit blocks init.

These are retained in STATUS/N8 rather than converted to release claims.
