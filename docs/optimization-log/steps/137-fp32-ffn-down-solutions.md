# Step 137 — FP32 FFN down row-invariant solutions

Status: completed by Experiment 321; rejected, track closed

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

结果：15个共同候选中只有296100 block exact，但逐M speedup仅
0.506/0.758/0.686/0.863×。不新增scope、不跑模型门，down vendor-solution线关闭。详见
[`Experiment 321`](../experiments/321-fp32-ffn-down-reject.md)。
