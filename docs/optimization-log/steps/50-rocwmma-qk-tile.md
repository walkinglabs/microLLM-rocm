# Step 50 — rocWMMA QK tile capability

Status: complete, admit prototype only

## Decision

48个新进程覆盖T16–2048、D64/128，完整BF16×BF16→FP32输出全部对齐。筛出的tile32/wave1
在T512比同二进制hipBLASLt快1.654×–1.784×，但T2048 D128只有0.688×。底层能力与反例都
成立，因此只进入online Attention原型；模型路由保持关闭。
