# Experiment 124：把上边界放大四倍，误差还在下降

## 为什么继续测

Experiment 123中Qwen最好的activation scale是最大候选0.05。像只量到山坡半腰一样，这时停止会
误判最低点。Exp124只扩大activation范围，weight候选、模型、prompt和完整logits门全部不变。

## 结果

GPU2三次预检均为use 0%、VRAM 0%。18/18个fresh worker执行成功，16个FP8候选仍为0个过门。

| 模型 | 最佳a/w | max/RMS | RMS / 0.05门 | top相同 | TPS |
|---|---:|---:|---:|---|---:|
| Qwen2.5-0.5B | 0.2 / 0.0025 | 4.263 / 0.669 | 13.38× | 是 | 2076 |
| DeepSeek Distill Qwen 1.5B | 0.2 / 0.005 | 6.806 / 1.170 | 23.40× | 是 | 1377 |

相对Exp123最佳RMS，Qwen下降65.2%，DeepSeek下降54.0%。这说明原固定scale确实严重限制了
数值范围，但误差仍远高于门。

![FP8 scale boundary](../assets/fp8-scale-boundary.svg)

## 反例怎样改变结论

原假设是0.05附近已经能代表全局scale的最好水平。0.2把两个模型误差都明显降低，推翻了这个
解释。新的最佳点又同时落在0.2上边界，因此本轮仍不能宣布“全局scale方向关闭”。

## 下一步

只再测0.4和0.8，继续沿相同完整logits门寻找曲线转折。如果误差开始反弹或仍无法接近门，停止
扩大数字，转向每个weight Tensor自己的amax scale。无论速度多快，0/16精度结果都不能成为默认。
