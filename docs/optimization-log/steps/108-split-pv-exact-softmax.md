# Step 108 — Exact softmax with split P×V

Status: complete; operator admitted

Step 107证明仅减少物理线程没有用，因为每个output column仍按T串行读取value并累加。Step 108只
拆这一段：

```text
parallel score                     保持Experiment 285
exact max + denominator            保持原256-lane归约树
normalized probabilities           写FP32全局buffer
split P×V partials                 多block并行不同position区间
ordered partial combine            输出context
```

合同：

1. score必须与current位级相同；
2. maximum、exp与denominator必须保持exact-order；
3. 只允许P×V累加树改变，并报告完整context Max/RMS与位级状态；
4. 搜索S1/2/4/8/16，覆盖Qwen/DeepSeek、T512/T2048、B1/B2、FP32/BF16；
5. S1必须作为额外launch/partial buffer反例；
6. operator winner至少Event 1.05x、wall 1.02x才进入DeepSeek完整logits门；
7. 模型门沿用303,872 logits、64 token、peak/KV与三对fresh process。

如果P×V-only在operator快但模型精度仍失败，则不再继续序列split；下一路线转向跨GQA heads的value
复用或更大图融合。

## 实测结果

160进程、80个candidate、16个case全部通过。S1全部位级相同且全部更慢；S16赢16/16，Event
1.2749x–2.9549x、wall 1.2372x–2.6373x，context Max/RMS最多3.90e-9/1.09e-9。
Step 109进入DeepSeek完整logits门，模型和Auto仍未改变。
