# Experiment 088 — 两个BF16一起读，小测试为什么看不出错？

Experiment 086证明cached Attention约占DeepSeek T2048 steady decode的60%。当前score阶段让每个
position线程串行读取width128个Key标量。相邻BF16只有2 bytes，候选希望一次读取两个，减少读取
和转换指令。

## 只改什么

- 只改BF16 fused cached Attention的Key dot；
- Query使用`float2`，Key把两个公开`hip_bfloat16`重解释为`__hip_bfloat162`；
- 两个乘加仍按x再y写出；
- Value、softmax、block 256 threads、shared score和FP32路径完全不变；
- odd width继续走scalar fallback。

## 小算子门给了假安全感

现有HIP cached Attention覆盖MHA/GQA、BF16 batch和长fallback，4个focused tests全部通过。它们的
width和数值规模很小，只能证明“没有崩溃、误差在宽松小shape门内”，不能证明官方模型轨迹。

## 官方完整cached logits立即否决

| Shape | logit数量 | max-abs | RMSE | 8-token suffix |
|---|---:|---:|---:|---|
| T2048 B1 | 151,936 | 0.05650 | 0.01323 | 相同 |
| T2048 B8 | 1,215,488 | 11.9780 | 1.52846 | 第3个token开始分叉 |

B1再次说明top-1相同不是精度通过。B8候选恰好生成更接近当前PyTorch的一条suffix，也不能证明
候选正确；它相对仓库冻结baseline的完整logits已经大幅改变。

## 两种仍待区分的解释

1. 公开`hip_bfloat16`与内部`__hip_bfloat162`的重解释/转换合同并不等价；
2. Release编译器对vector load后的算术做了不同组合，长规约把细小变化放大。

这个实验不能仅凭结果区分两者。下一次若继续pair load，必须使用一个32-bit原始整数读取，再把
两个16-bit公开BF16值显式恢复；不能继续依赖内部vector类型重解释。

![BF16x2 Key load discarded](../assets/bf16x2-key-load-discard.svg)

## 决定

不做性能测量，候选完整回退。性能再快也不能越过官方完整logits门。Experiment 089若继续这条
路线，只允许改变“读取两元素的方式”，仍保持scalar公开BF16转换和原累加顺序。

数据见[`088-data`](088-data/)。
