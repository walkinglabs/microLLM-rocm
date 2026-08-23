# Experiment 138：FFN输出误差不大，相加后突然放大

| 模型/关键层 | attention output rel-L2 | FFN gate/up | FFN output | block output | 最后一步Δ |
|---|---:|---:|---:|---:|---:|
| Qwen21 | 11.16% | 4.57%/6.93% | **1.74%** | **21.21%** | +19.47pp |
| Deep27 | 8.58% | 4.31%/3.88% | **3.24%** | **11.50%** | +8.26pp |

![FP8 block detail](../assets/fp8-block-detail.svg)

gate/up、activated和FFN output都没有在最后一步爆炸。大跳发生于`attention_residual + ffn.output`
得到block output时，符合残差两项部分抵消、已有小误差被相对放大的解释。

但当前detail没有显式block input，且相对误差会受参考Tensor范数影响，所以只写“支持”而非“证明”。
下一反驳实验：记录input；再让关键block用FP32而输入仍来自前面FP8层。若最终误差显著下降，说明
本block计算是主因；若仍大，说明上游误差与残差几何关系主导。
