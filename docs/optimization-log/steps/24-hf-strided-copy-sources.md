# Step 24 — strided-copy source attribution

Status: complete

## Evidence

- six fresh composed T512 processes;
- Qwen 96 calls / 100.7 MB;
- DeepSeek 112 calls / 205.5 MB;
- 100% attributed to attention.layout and attention.core;
- exact three records per model and deterministic process equality;
- disabled diagnostics retain no extra records.

## Decision

Implement an inference BTHD Attention island. Do not optimize the generic copy kernel.
