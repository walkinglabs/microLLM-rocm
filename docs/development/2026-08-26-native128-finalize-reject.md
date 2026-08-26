# 2026-08-26 — native128数值正确，但只有约1.003×

DeepSeek T512/T2048、B1/B2、FP32/BF16 cache共16个fresh process全部通过完整输出，最大误差不超过
3.73e-9，且没有backend allocation或payload transfer。数值不位级相同，符合新归约顺序预期。

T2048四个case Event/wall只在约1.003×附近，0个达到1.05×/1.02×门。candidate拒绝，finalize局部线
关闭。下一提交删除native128 API/kernel/runner，只保留raw证据；之后审计33.25%的GEMM类别。
