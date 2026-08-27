# Qwen3 three-step complete-state trajectory

Three consecutive B1/T32 steps compare every loss, then 310 final parameters, 620 final AdamW moments and strict step=3.

| Precision | Loss max | Parameter Max/RMS | Moment Max/RMS | Decision |
|---|---:|---:|---:|---|
| FP32 | `4.470e-6` | `3.227e-5 / 5.161e-8` | `7.258e-5 / 4.584e-8` | 6/6 pass |
| BF16 | `0.01167` | `5.993e-5 / 3.995e-6` | `0.1407 / 8.677e-5` | loss, parameter RMS, moment Max fail |

FP32 loss falls `2.3768→0.7796→0.3520` in both frameworks. BF16 remains finite but diverges; its worst final moment moves to block2 FFN down. Raw files contain 930 records per precision. Four temporary exports total 14,305,423,728 bytes per precision and were deleted.
