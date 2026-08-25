# Step 67 — Reprofile current retained training

Status: complete

当前 B1T512 BF16 路径的稳定 Kernel/step 为 31.327/71.873 ms，GEMM 占
58.56%/63.43%，AdamW 占 13.22%/18.16%。热点排序与 Experiment 216 一致；
下一节点必须改变训练 GEMM 或 graph-wide 架构。

