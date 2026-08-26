# Step 143 — Clean DeepSeek fixed-workload local saturation

Status: completed by Experiment 327

当前T2048/B2/N64为1.1393×PyTorch。finalize六条结构路线关闭；GEMM rows2 operator 1.814×但整模
1.00968×未过门；cast只占4.11%，即使免费删除上限也仅1.043×，相邻cast/Arena路线已有证据。

因此停止该固定workload的局部策略搜索。下一工作必须改变尺度：serving并发与相同resident policy、训练
架构，或Radeon/ROCm版本矩阵。
