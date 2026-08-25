# Step 61 — reprofile after FFN Norm fusion

Status: complete

## Decision

Kernel时间8.315→8.208 ms / 14.862→14.659 ms，cast调用96→72 / 112→84，GEMM占60.9%/68.2%。
下一有界问题是Attention Norm直入QKV Arena。
