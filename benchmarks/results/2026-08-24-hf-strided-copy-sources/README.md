# Composed T512 strided-copy source attribution

Experiment 201 adds the existing model AllocationSource to the strided-copy
diagnostic aggregation key and exposes the records through hf_infer.

Three fresh composed-policy processes per model produce identical records:

| Model | Calls | Bytes | attention.layout | attention.core |
|---|---:|---:|---:|---:|
| Qwen | 96 | 100,663,296 | 72 calls / 56,623,104 B | 24 / 44,040,192 B |
| DeepSeek | 112 | 205,520,896 | 84 / 117,440,512 B | 28 / 88,080,384 B |

No copy is attributed to embedding, FFN, output or unspecified code.

Every block performs four copies:

1. query BTHD→BHTD;
2. key BTHD→BHTD;
3. value BTHD→BHTD;
4. context BHTD→BTHD.

The three exact records per model distinguish query, the two equal K/V
layouts, and context. This selects an inference BTHD Attention island as the
next candidate. Optimizing the generic strided-copy kernel would preserve all
100.7/205.5 MB of avoidable traffic.

Diagnostics remain off by default. Allocation source scopes activate when
either allocation or strided diagnostics is enabled and remain a fast no-op
when both are disabled.

Files: raw.jsonl, summary.json and verification.json.
