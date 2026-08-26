# Step 149 — zero-stride scalar-seed backward

Status: keep

`sum()`的output gradient是4-byte storage加全0 stride。旧bridge物化完整FP32 Tensor。新窄路由
直接读取设备标量；64K/1M Event改善1.164×/1.081×，peak从263,680/4,195,840B降到1,536B，
完整梯度过门。mean/weighted/general gradient保持旧路径。

详细记录见[Experiment 333](../experiments/333-pytorch-swiglu-scalar-seed.md)。

