# 2026-08-26 — Autograd外部梯度缓冲区

## 先用一句话说明

以前，自动求导自己决定把参数梯度放在哪里。现在，调用者可以先准备一块连续内存，
再明确告诉一个叶子参数：“你的梯度只能写到这里。”

这项能力是外部训练框架、梯度池和通信bucket复用同一块显存的基础。它目前只是显式接口，
没有成为默认训练策略；完整Tiny/Model-S模型还要单独通过数值、地址、显存和时间门。

## 为什么需要它

把梯度想成每位学生的作业纸。普通Autograd在收到第一份答案时，直接拿那张纸保存；收到
第二份答案时，再找一张新纸相加。这样容易实现，但纸放在哪里每次可能不同。

外部系统可能已经准备好一张大表格：每个参数都有固定的一格，RCCL或PyTorch也认识这些
地址。若Autograd另外申请一份梯度，后面还要复制一次。外部梯度缓冲区让叶子参数直接向
指定格子累加。

```text
一块调用者拥有的内存
├── 参数 A 的梯度切片
├── 参数 B 的梯度切片
└── 参数 C 的梯度切片

Value::bind_grad_buffer(切片)
             │
             └── backward 的每次贡献都原地加到同一地址
```

## 接口和不变量

```cpp
parameter.bind_grad_buffer(buffer);       // 默认先清零
loss.backward();                          // 原地累加
parameter.zero_grad();                    // 清零，但仍绑定
parameter.unbind_grad_buffer();           // 明确解除绑定
```

接口只接受满足全部条件的Tensor：

- 参数必须需要梯度；
- 参数必须是没有父节点的叶子，不能给中间结果偷偷换存储；
- shape和device必须与参数完全一致；
- dtype必须是FP32；
- 缓冲区必须连续；
- 调用者必须让外部内存活到全部CPU/GPU工作完成。

绑定后，`set_grad`和第二次绑定会报错。`zero_grad`只清零，不改变地址；解除绑定后，叶子恢复
普通Autograd规则。Embedding重复token使用稀疏行累加，普通分支使用原地dense add。

## 这一步验证了什么

CPU测试覆盖：

- 一个叶子经两条分支收到`2 + 3 = 5`的梯度；
- 外部地址在两次backward和一次`zero_grad`之后不变；
- Embedding重复索引会把同一行累加两次；
- 非叶子、错误shape、重复绑定和绑定后的`set_grad`均被拒绝；
- 解除绑定后不再把旧缓冲区当作当前梯度。

MI300X HIP测试覆盖：

- backward期间没有H2D或D2H payload传输；
- GPU结果与CPU reference一致；
- 清零和第二次backward仍使用原来的显存地址。

后续完整模型门已经完成。18个新进程覆盖Tiny T8、Model-S T8/T32并轮换先后顺序：Tiny
21/21、Model-S 57/57地址稳定，15,586,176个Model-S梯度元素Max/RMS均为0。但Event中位数
只有0.871×/0.814×/0.792×，Model-S测量区峰值增加6.75–10.69MiB。因此它只保留为显式
互操作接口，不进入默认训练策略。完整证据见
[Experiment 328](../optimization-log/experiments/328-external-gradient-pool-discard.md)。
