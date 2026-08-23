# Experiment 125：DeepSeek转弯了，Qwen还没有

## 结果

固定activation 0.4/0.8与原四个weight scale。GPU2三次0/0预检后，18/18 fresh worker执行
成功，16个FP8候选仍然0个通过完整logits门。

| 模型 | 固定规则最佳a/w | max/RMS | 与Exp124 RMS比较 | top相同 | TPS |
|---|---:|---:|---:|---|---:|
| Qwen2.5-0.5B | 0.8 / 0.005 | 1.580 / 0.303 | -54.7% | 是 | 1840 |
| DeepSeek Distill Qwen 1.5B | 0.4 / 0.00125 | 6.753 / 1.235 | +5.6% | 是 | 1374 |

DeepSeek在0.8/0.01虽然RMS降到0.628，但top token已经改变，不能被选为可用候选。它证明只看
平均误差会隐藏生成决策翻转。

![FP8 scale turn](../assets/fp8-scale-turn.svg)

## 结论

- DeepSeek的top-equal误差谷底已经在0.2附近出现，而且仍为精度门23倍；停止搜索DeepSeek全局scale。
- Qwen在0.8仍改善且位于边界，只再测1.6/3.2一次。
- 即使Qwen找到谷底，单个全局值也不能同时成为两个模型的策略；下一设计仍是weight per-tensor scale。
- 0/16候选不改变“FP8非默认、官方低精度模型未通过”的状态。
