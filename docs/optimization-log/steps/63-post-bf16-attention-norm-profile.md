# Step 63 — reprofile after both Norm fusions

Status: complete

## Decision

Kernel时间8.069/14.489 ms，cast为48/56，每层各剩一次FP32→BF16和BF16→FP32。
下一节先归因，不盲目融合。
