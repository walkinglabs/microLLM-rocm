# 两个答案只差一点点：怎样定位 DeepSeek 的跨 slot 分叉

## 1. 问题是什么

同一批 DeepSeek 请求使用1/2个slot时，第6条请求的第5个生成token是`23606`；使用4/8个slot时
却变成`1196`。前四个token完全相同，之后因为语言模型会把自己的答案继续作为输入，整段输出
沿两条路分开。

这不一定说明某个Kernel写错。BF16只保留有限精度；矩阵的batch shape改变时，库可能选择另一种
计算顺序。两个候选本来就非常接近，小数点后的变化足以交换第一名。

## 2. 新的诊断接口记录什么

`ContinuousBatchConfig::capture_selection_diagnostics`默认关闭。显式开启后，每次选token都会记录：

- scheduler step、request ID和slot；
- 这是请求的第几个生成token；
- 当时Cache position；
- logits来自prefill、uniform decode还是positions-aware decode；
- 产生logits的真实batch大小；
- GPU实际选择的token；
- top-1、top-2 token及其logit；
- 第一名减第二名的margin。

诊断会把logits复制到CPU，不能用它测性能。普通服务默认不打开，因此原性能路径没有额外D2H。

## 3. 首个分叉点看见了什么

18个fresh process全部稳定：

| 路径 | source / batch | top-1 | logit | top-2 | logit | margin |
|---|---|---:|---:|---:|---:|---:|
| S1 | uniform / B1 | 23606 | 18.917286 | 1196 | 18.901663 | 0.015623 |
| S2 | positions / B2 | 23606 | 18.915491 | 1196 | 18.904139 | 0.011353 |
| S4 | positions / B4 | 1196 | 18.890232 | 23606 | 18.889563 | 0.000669 |
| S8 | positions / B8 | 1196 | 18.890232 | 23606 | 18.889563 | 0.000669 |

S4/S8的margin只有约0.00067，而且两个候选正好互换。GPU argmax与CPU检查的top-1始终相同，
所以不是argmax挑错了；送进argmax的logits已经不同。

## 4. 怎样只改变一个条件

S1/S2中，第6条请求单独做B1 prefill；S4/S8中，它与另一条32-token prompt组成B2 prefill。
但decode batch也分别是1/2/4/8，所以第一张表还不能证明是哪一段造成差异。

新增一个只用于实验的开关：

```text
batch_equal_length_prefill = false
```

它保持S4/S8的slot、请求、positions-aware decode和decode batch不变，只把等长prompt逐条prefill。
结果如下：

| S4/S8条件 | prefill | decode | 第5个token | margin |
|---|---|---|---:|---:|
| 默认 | B2 | B4/B8 | 1196 | 0.000669 |
| 反驳实验 | B1 | B4/B8 | 23606 | 0.011353 |

反驳组的完整输出回到S1/S2，且该点logits逐值等于S2。这推翻了“decode batch是主要原因”，支持
“prefill batch shape先引入小差异，低margin自回归把它放大”。

## 5. 为什么不把默认改成串行prefill

PyTorch full-BF16 sequential reference是外部token参考。在原分叉请求上：

- 默认B2 prefill与PyTorch相同，选择`1196`；
- 强制B1 prefill反而选择`23606`，新增一个外部差异；
- 默认S4与PyTorch只剩第8条请求在生成位置14不同；
- serial S4同时在第6条位置4和第8条位置14不同。

因此“所有slot必须逐字节得到S1答案”不是比外部参考更高的正确性标准。默认批量prefill既更快，
在这条请求上也更接近PyTorch，所以保留。serial开关只用于反驳实验，不作为生产默认。

## 6. 现在能得出什么

- scheduler、Cache refill和argmax不是这次分叉的原因；
- prefill B1/B2是因果变量；
- 当前证据更像BF16/hipBLASLt不同M shape的数值差异，但还没有排除B2 local-row/copy问题；
- 下一实验应交换B2两行顺序并放入重复prompt，检查差异是否跟随row索引；
- 任何吞吐比较都必须使用关闭诊断的普通路径。

![DeepSeek slot divergence](../optimization-log/assets/continuous-divergence.svg)

完整记录见[Experiment 104](../optimization-log/experiments/104-deepseek-prefill-divergence.md)。
