# T1024 exact FP32 Attention solution screening

Four QK/PV shapes for Qwen H14/D64 and DeepSeek H12/D128 were screened in
three fresh processes with up to 64 hipBLASLt algorithms. All 64 candidates per
shape were common and correctness-passing.

Local winners:

| Shape | Index | Event speedup |
|---|---:|---:|
| Qwen QK | 304680 | 1.538× |
| Qwen PV | 294867 | 1.476× |
| DeepSeek QK | 310758 | 1.060× |
| DeepSeek PV | 296917 | 1.103× |

These are operator candidates only. The current BTHD PV descriptor does not
consume the generic PV keys, and the QK full-model gate rejects the remaining
policy.
