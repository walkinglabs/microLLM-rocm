# 2026-08-24 — Hybrid AdamW 后的训练 profile

使用相同二进制分别采集 Qwen/DeepSeek 的 load+1-step 与 load+3-step Kernel stats。差分得到
Qwen/DeepSeek 每步 32.117/72.906 ms，GEMM 占 59.33%/63.81%，AdamW 降到 12.82%/17.61%。

本节点同时强化 `profile_step_delta.py`：负 call delta 直接失败，softmax/repeat 独立分类，
并增加注册测试。下一工作边界是训练 GEMM，不再扩大 AdamW 阈值。

详细证据见
[Experiment 216](../optimization-log/experiments/216-post-hybrid-training-profile.md)。

发布门：CPU 325/325、ASan/UBSan 323/323、PyTorch 299/299、CPU/HIP 509/509
（3 个条件跳过、HIP 标签 173/173）、RCCL 14/14。干净覆盖率为
79.8% lines、87.7% functions、60.4% branches；覆盖清单注册 88 个测试文件。
