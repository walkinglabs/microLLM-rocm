# Experiment 126：Qwen还在下降，但停止跨模型盲搜

## 结果

activation 1.6/3.2、原四个weight scale。GPU2三次0/0预检后，9/9 fresh worker成功，8/8
top token相同，0/8通过完整logits门。

最佳`activation=3.2, weight=0.0025`：

```text
TPS  2026.16
max  1.00196
RMS  0.21674
gate 0.2 / 0.05 / top-equal → fail
```

相对Exp125最佳，RMS再降28.5%，曲线没有字面转弯；但仍是RMS门的4.33倍。

![Qwen FP8 scale closure](../assets/qwen-fp8-scale-closure.svg)

## 为什么现在停止

停止不是声称“所有可能的小数都被证明失败”。理由是：

1. DeepSeek保留top token的谷底在0.2附近，Qwen到3.2仍在下降，一个全局值无法成为跨模型策略；
2. 从0.8到3.2把activation scale放大4倍，只把RMS从0.303降到0.217，仍差4.33倍；
3. weight最佳值也随阶段变化，提示不同Tensor的数值范围正在互相妥协；
4. 继续枚举全局数字不会回答哪一层饱和、哪一层量化太粗。

因此工程上关闭“跨模型全局scale搜索”，不把它写成数学证明。

## 下一步

实现显式`tensor-amax` weight policy：每个Linear权重用自己的`max(abs(weight))/240`，报告实际
scale范围和准备成本。activation暂时保持固定，确保一次只改变weight尺度；随后仍用官方完整
logits判断它是否真正缩小误差。
