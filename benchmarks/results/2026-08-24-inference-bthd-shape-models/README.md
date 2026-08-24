# Inference BTHD sequence and batch matrix

Experiment 203 extends the explicit BTHD island to B1/T256, B1/T1024 and
B2/T512 on both official models.

Thirty-six uninstrumented performance processes and six isolated candidate
diagnostic processes produce:

| Model | Case | Baseline | BTHD | Speedup | Peak saved |
|---|---|---:|---:|---:|---:|
| Qwen | B1/T256 | 67,212 | 76,761 tok/s | 1.1421× | 2 MiB |
| Qwen | B1/T1024 | 114,528 | 125,849 | 1.0989× | 8 MiB |
| Qwen | B2/T512 | 139,500 | 151,382 | 1.0852× | 8 MiB |
| DeepSeek | B1/T256 | 36,834 | 40,322 | 1.0947× | 3.5 MiB |
| DeepSeek | B1/T1024 | 63,042 | 68,805 | 1.0914× | 14 MiB |
| DeepSeek | B2/T512 | 67,916 | 74,186 | 1.0923× | 14 MiB |

All complete logits are bit-exact and every batch row preserves top-1.
Attention layout/core copies are zero in all six diagnostic cases.

B2 retains one small unspecified copy, 7,168/12,288 bytes, when selecting the
last hidden row from the batch/sequence Tensor. This is outside the Attention
island. B1 cases have no residual copy.

Decision: keep the explicit BTHD policy for all measured sequence/batch cases.
Cached prefill and trace-value routes remain on the old fallback.

Files: performance-raw.jsonl, diagnostic-raw.jsonl, summary.json and
verification.json.
