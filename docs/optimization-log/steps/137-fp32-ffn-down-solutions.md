# Step 137 — FP32 FFN down row-invariant solutions

Status: planned after route cleanup

Experiment 320把第一处差异移动到down。真实descriptor family为：

```text
M = 2048 / 4096 / 8192 / 16384
K = 8960
N = 1536
transpose = NN
dtype = FP32
```

先复用通用row-invariance工具做inventory交集、CPU sentinel、完整重复block bitwise、同进程default Event
和每M 0.95门。只有operator通过才考虑新的独立down scope；不得复用或保留已失败的gate/up模型route。
