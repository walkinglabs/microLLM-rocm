# Qwen3 checkpoint resume audit

One process saves checkpoint v2 immediately after step1 and continues through step3. A fresh process restores that exact checkpoint and performs steps2–3. The two branches compare losses, 310 parameters, 620 AdamW moments and step/global step.

| Precision | Loss Max | Parameter Max | Moment Max | Result |
|---|---:|---:|---:|---|
| FP32 | `0` | `1.86e-9` | `2.98e-8` | 5/5 pass |
| BF16 | `0` | `1.49e-8` | `1.79e-7` | 5/5 pass |

The first attempted control reran step1 in a separate process and failed a bitwise claim (`2.38e-7` parameter, `5.36e-7` moment), because it mixed checkpoint behavior with GPU algorithm scheduling. The shared-state branch isolates restore. Losses are bitwise; tied/atomic paths retain tiny non-bitwise parameter/moment differences under fixed `1e-7/1e-5` gates.

Raw files contain 930 records per precision. Checkpoint plus four state exports total about 21.46GB per precision and were deleted.
