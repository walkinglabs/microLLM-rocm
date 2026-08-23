# Experiment 125 data

第三段activation边界扩展。weight候选、官方模型、T8 prompt和完整logits门保持不变。

```text
activation scales 0.4, 0.8
weight scales     0.00125, 0.0025, 0.005, 0.01
workers           18 fresh processes
precision         complete 151936-value last-token logits
```

目录保存实际命令、三次0/0 GPU2预检、18条raw、summary、空stderr和结论门。DeepSeek在保留
top token的候选中出现转折；Qwen最佳点仍在0.8上边界。
