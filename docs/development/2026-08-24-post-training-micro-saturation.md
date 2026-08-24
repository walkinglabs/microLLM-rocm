# 2026-08-24 — 训练微优化饱和审计

## 结论

当前默认B1/T512训练中，去除加载与初始化后，GEMM和AdamW合计占Qwen/DeepSeek Kernel时间的
72.71%/83.77%。训练侧单cast、单add、单Norm和只减少launch的搜索空间暂时关闭。

## 测量方法

每个模型用相同二进制分别profile“加载+一步”和“加载+三步”。按完整Kernel名称相减后，所有
类别的调用差都非负，差值恰好对应两个step。这样不会把权重转置、BF16 mirror准备或library
首次setup误写成训练热点。

## 下一版本边界

- GEMM：需要新的精确shape算法或真正的grouped backward；
- AdamW：需要减少实际读写流量，而不是继续合并launch；
- 图执行：需要完整liveness plan和稳定地址后再进入异构HIP Graph。

详细证据见[实验213](../optimization-log/experiments/213-post-training-micro-saturation.md)。
