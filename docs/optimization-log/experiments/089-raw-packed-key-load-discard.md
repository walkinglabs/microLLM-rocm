# Experiment 089 — 不用内部vector类型，为什么误差一个数字都没变？

Experiment 088把两个公开`hip_bfloat16`重解释成内部`__hip_bfloat162`，官方完整logits失败。它留下
两个解释：类型重解释不等价，或pair循环改变了Release长规约代码生成。

## 只反驳第一个解释

这次不使用内部vector类型：

```text
一次读取32-bit原始字
→ lower 16 bits写回公开hip_bfloat16 first.data
→ upper 16 bits写回公开hip_bfloat16 second.data
→ 仍按first、second顺序累加
```

Query仍标量，Value/softmax/block布局不变，odd width走原路径。4个focused HIP tests仍全部通过。

## 官方结果与Experiment 088完全相同

| Shape | max-abs | RMSE | token |
|---|---:|---:|---|
| T2048 B1 | 0.0564956665 | 0.0132278599 | 相同 |
| T2048 B8 | 11.9780039787 | 1.5284578662 | 第3个开始分叉 |

两个实验的logit数量、max、RMSE和suffix逐项相同。显式恢复公开scalar没有改变失败，因此“只有内部
vector转换错了”被推翻。剩余更强的解释是：把`column++`循环改写成pair循环后，Release codegen
或规约轨迹改变；当前证据还不足以区分循环重排、lane顺序和编译器融合的具体责任。

![Raw packed BF16 Key load discard](../assets/raw-packed-key-load-discard.svg)

## 决定

候选不计时并完整回退。连续两个相同的官方失败已经关闭“本地BF16 pair Key load”这条搜索，除非
先建立能证明逐position dot位级一致的独立反汇编/中间结果门。

下一节点不继续排列相同pair写法。allocator已经由Experiment 087稳定，Experiment 090回到此前
被allocator干扰的GPU token-history/D2H候选，验证它现在是否可以安全保留。

数据见[`089-data`](089-data/)。
