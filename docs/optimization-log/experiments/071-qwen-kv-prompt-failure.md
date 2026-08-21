# Experiment 071 — Qwen uniform BF16也会被换prompt推翻

Qwen uniform BF16在repeat T32/512/2048、B1/B8曾6/6通过。使用Experiment 070的prompt
挑战后：repeat 3/3、rotated 3/3、ramp 5/5通过，但constant 0/3通过。

```text
constant T32   max_abs 0.681 / RMSE 0.146
constant T512  max_abs 0.619 / RMSE 0.146
constant T2048 max_abs 0.449 / RMSE 0.087
```

四步token尚未变化，但完整logits门已失败，所以不能称为精度同步。

## context反驳

constant T512中，前2层FP32刚好通过：`max_abs 0.241 / RMSE 0.0487`，Cache仍缩小
1.846×。把同一策略搬到T2048后，RMSE跳到`3.141`并发生token分叉。

继续测前4/8/12层FP32：T2048 RMSE仍约`3.144–3.148`，token仍错；只有全部24层FP32
恢复0误差。

![Qwen KV prompt failure](../assets/qwen-kv-prompt-failure.svg)

## 决定

- Qwen uniform BF16继续作为显式速度/显存实验路径，不再写成多prompt严格安全；
- 当前没有一个保持低精度收益、又覆盖long constant的Qwen robust-strict层集合；
- 对该稳定失败必须fallback到全FP32 Cache；
- repeat/ramp成功和constant失败都保留，不能用“普通文字不长这样”删除反例。

这不证明所有自然文本都会失败，也不证明constant是常见生产输入。它只证明“Qwen BF16
Cache普遍精度同步”的宽泛结论不成立。
