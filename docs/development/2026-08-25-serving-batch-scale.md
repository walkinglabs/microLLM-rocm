# Batch变大：总吞吐提高，但每个请求并没有免费加速

## 把Batch想成同时服务多位同学

B1一次只处理一个请求，B8一次处理八个。GPU可以同时做更多独立工作，所以总tokens/s会上升。
理想情况是B8达到B1的8倍；实际通常达不到，因为显存、Kernel和调度也会增加。

我们用`实际扩展 / batch`定义效率：

```text
Qwen B8: 6.585 / 8 = 82.3%
DeepSeek B8: 6.282 / 8 = 78.5%
```

因此B8总吞吐很高，但64-token生成延迟也从Qwen约479ms升到582ms，DeepSeek约696ms升到887ms。
服务端要同时看总吞吐和单请求等待时间。

## 显存为什么每请求反而下降

模型权重只保存一份，batch里的请求共同使用；KV Cache按请求增加。所以总peak上升，但除以batch后
每请求承担的权重份额下降。Qwen microLLM从B1每请求1.48GiB降到B8约0.45GiB，DeepSeek从
4.52GiB降到约0.87GiB。

## 为什么还不能自动选择B8

Qwen所有batch与PyTorch的64 token相同。DeepSeek B2/B4相同，B1/B8却从第3个token分叉。由于
microLLM是混合BF16权重/FP32边界，PyTorch是全BF16，这可能是精度政策，也可能是batch实现问题。

下一步不继续跑速度，而是让microLLM B1和B2/B4/B8导出同一步完整logits，逐行比较。只有框架自身
跨batch一致，scheduler才有资格选择batch区间。
