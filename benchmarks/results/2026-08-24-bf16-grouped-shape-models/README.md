# BF16 grouped sequence and batch model matrix

Experiment 199 gates the rows256/1024 operator evidence on three distinct
complete-model workloads:

- B1/T256, flattened rows 256;
- B1/T1024, flattened rows 1024;
- B2/T512, also flattened rows 1024.

Each model/case runs three fresh baseline and three fresh composed-grouped
processes with two warm-ups and five measured prefills.

| Model | Case | Baseline | Both | Speedup | Peak ratio |
|---|---|---:|---:|---:|---:|
| Qwen | B1/T256 | 60914 | 67459 tok/s | 1.1075× | 1.00176 |
| Qwen | B1/T1024 | 111466 | 114589 tok/s | 1.0280× | 1.00644 |
| Qwen | B2/T512 | 135291 | 139499 tok/s | 1.0311× | 1.00657 |
| DeepSeek | B1/T256 | 34202 | 36786 tok/s | 1.0755× | 1.00088 |
| DeepSeek | B1/T1024 | 61700 | 63006 tok/s | 1.0212× | 1.00338 |
| DeepSeek | B2/T512 | 66322 | 67804 tok/s | 1.0223× | 1.00340 |

All 36 complete outputs are finite, inside the BF16 0.25/0.05 envelope and
preserve top-1 independently for every batch row. Combined setup is
208.1–212.2 ms and remains an explicit serving warm-up cost.

## CLI bug found by the gate

The first formal run stopped before producing a summary: the model returned
both B2 rows, but logits-output wrote only B0. The CLI now exports all batch
rows in last mode and gathers each batch row's last token in full mode. A
real tiny-HF fixture proves B2 last/full both contain 2×vocab values and match
two B1 rows. The invalid partial run is not retained.

The different B1/T1024 and B2/T512 ratios prove that equal flattened projection
shape does not imply equal end-to-end behavior.

Decision: keep explicit rows256/1024 composed policies. Defaults remain
unchanged and backend-local indices stay outside source policy.

Environment: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, HIP
runtime/driver 71399004, hipBLASLt 1.3.0. Files: raw.jsonl, summary.json and
verification.json.
