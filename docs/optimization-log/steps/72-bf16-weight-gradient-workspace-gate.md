# Step 72 — Allocating versus preallocated weight-gradient cost

Status: complete, workspace API rejected

扩展独立算子benchmark，同时测公共allocating API和已有preallocated组成路径。固定Qwen/DeepSeek
gate/up shape，三个新进程，Event与wall分开。只有wall median至少1.01×且minimum不回退，
才设计workspace API；否则记录“逻辑分配可见但物理分配为零”的反例并关闭此track。

结果：Qwen/DeepSeek wall为0.986×/0.889×，0/2过门，不增加workspace API。
