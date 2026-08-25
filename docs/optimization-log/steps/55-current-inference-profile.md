# Step 55 — reprofile retained B1T1024 inference

Status: complete, select exact Attention GEMM screening

## Decision

4个rocprof进程用`(6−1)/5`得到steady prefill：GEMM占Qwen/DeepSeek 59.7%/66.8%，softmax
14.8%/9.2%。旧softmax局部候选理论整步不足0.3%，online track已关闭；下一节点只筛T1024
QK/PV exact solution，不修改默认。
