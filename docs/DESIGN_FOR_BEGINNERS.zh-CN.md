# microLLM-rocm 设计说明：把一串数字变成会学习的小模型

这份文档写给第一次接触框架内部结构的人。读者只需要知道：数组可以保存数字，函数可以接收输入并返回输出。

我们不会一上来背术语。先看这个仓库到底在做什么：

```text
一块内存
  ↓ 加上“谁负责释放”
Storage
  ↓ 加上“这些数字排成什么形状”
Tensor
  ↓ 加上“怎样计算”
Operator / Kernel
  ↓ 加上“结果是从谁算出来的”
Autograd Graph
  ↓ 把许多算子按固定结构连接
Transformer
  ↓ 反向计算梯度，再更新参数
训练
```

框架的核心不是一个神秘的大类。它只是在每一步多保存一点必要信息，并守住每一步的规则。

## 1. 先认识一块内存

假设桌上有六张写着数字的卡片：

```text
[1, 2, 3, 4, 5, 6]
```

计算机里，这些卡片连续放在内存中。`float*` 像第一张卡片的地址。只拿到这个地址，我们不知道：

- 一共有几张卡片；
- 它们是不是 `2×3` 的表格；
- 卡片在 CPU 内存还是 GPU 显存；
- 最后应由谁把它们收走；
- 另一个对象是不是也在看同一叠卡片。

裸指针能找到数据，却不能回答这些问题。

## 2. Storage：内存的保管员

`Storage` 只负责保管一块内存。它记录：

```text
地址 + 字节数 + 设备 + 释放办法
```

可以把它想成仓库管理员。管理员知道箱子在哪里、箱子有多大、箱子属于 CPU 还是 GPU，并保证最后只释放一次。

多个 Tensor 可以共享一个 Storage。共享时，最后一个使用者离开后，Storage 才释放内存。这样 view 不会提前把原数据释放，也不会释放两次。

Storage 不知道矩阵、图片或模型。它只知道“我保管多少字节”。这种职责越单纯，生命周期越容易检查。

## 3. Tensor：同一块内存的阅读说明

Tensor 在 Storage 上增加四类信息：

```text
shape   每个方向有多少格
stride  沿某个方向走一步，要跨过多少个元素
dtype   每个格子用什么数字格式
offset  第一个逻辑元素离 Storage 开头多远
```

六个数既可以解释成一排 `shape=[6]`，也可以解释成两行三列：

```text
shape  = [2, 3]
stride = [3, 1]

1 2 3
4 5 6
```

索引 `[row,column]` 对应 Storage 中的位置：

```text
offset + row × stride[0] + column × stride[1]
```

例如 `[1,2]` 的位置是 `0+1×3+2×1=5`，读到第六个数 `6`。

## 4. View：不搬数据，只换阅读方法

转置 `2×3` Tensor 时，不必复制六个数字。只交换 shape 和 stride：

```text
原来：shape=[2,3], stride=[3,1]
转置：shape=[3,2], stride=[1,3]
```

两者共享 Storage。修改底层数据，两边都会看到变化。

这也带来一条规则：算子必须明确自己是否接受不连续 Tensor。当前 readable HIP Kernel 通常要求连续输入；遇到转置 view，先调用 `contiguous()`，由 stride-copy Kernel 按逻辑顺序整理一份连续数据。

## 5. Device：数字到底放在哪里

本项目使用两种设备：

```text
cpu:0  普通内存，CPU 循环可以直接读取
hip:0  AMD GPU 0 的显存，HIP Kernel 可以读取
```

CPU 指针不能直接当成 GPU 指针使用。`Tensor::to(device)` 会明确复制方向。

我们不让算子偷偷迁移设备。如果 add 的左边在 CPU、右边在 GPU，算子会报错。调用者必须先决定数据放在哪里。

## 6. Stream 与 Event：任务队列和完成标记

CPU 调用 GPU Kernel 时，通常只是把任务放进队列，然后继续执行：

```text
CPU:  提交复制 → 提交 Kernel → 继续走
GPU:             复制 → Kernel
```

Stream 是有顺序的任务队列。同一个 Stream 中，后一个任务等前一个任务。Event 是队列中的完成标记，可以用于计时或让另一条 Stream 等待。

“提交完成”不等于“GPU 计算完成”。测试读取 GPU 结果前要等待；性能代码则不能为了省事到处做全局同步。

## 7. Operator 与 Kernel 有什么不同

Operator 是框架看到的数学接口，例如：

```text
add(left, right) -> output
```

Kernel 是 GPU 实际运行的函数，例如一个线程负责一个元素：

```text
output[i] = left[i] + right[i]
```

一次 Operator 调用会做四件事：

1. 检查 dtype、shape、device 和连续性；
2. 为结果申请 Tensor；
3. CPU 输入走参考循环，HIP 输入启动 Kernel；
4. 返回拥有明确 shape 的结果。

每个算子都必须先写契约：输入有几个维度、哪些维度必须相等、输出 shape 怎样计算、支持什么 dtype 和 device、允许与 PyTorch 相差多少、哪些错误必须被拒绝。

## 8. 为什么保留 CPU reference

GPU 一次运行很多线程，错误可能只在某个 shape 出现。CPU 循环较慢，但容易逐项阅读。

因此同一个问题保留三条路径：

```text
CPU reference        最容易理解，作为正确性 oracle
readable HIP Kernel  自己实现，证明 GPU 路径
optimized path       例如明确选择 hipBLASLt，追求速度
```

优化实现不能删除 CPU reference。Kernel 变快后，仍要回到 CPU 和 PyTorch 检查数值。

## 9. 自动求导图：给每个结果留下来路

考虑：

```text
a = [1, 2]
b = [3, 4]
c = a × b
loss = sum(c)
```

前向得到 `c=[3,8]` 和 `loss=11`。要知道 `a` 怎样影响 loss，框架必须记住 `c` 来自 `a` 和 `b`。

本项目的 `Value::Node` 保存：

```text
data          本节点的 Tensor
gradient      loss 对本节点的梯度
parents       本结果由哪些节点产生
backward      收到上游梯度后，怎样计算父节点梯度
requires_grad 是否需要梯度
```

图不是预先画好的。每次执行 `add(Value,Value)` 等操作时，结果节点立即记录父节点和 backward 函数。这叫 eager graph。

## 10. 图的构建和 backward 必须单独测试

图测试不能混在某个 Kernel smoke 中。本项目使用独立的 `tests/graph/`，检查：

1. 操作后是否建立正确父边；
2. 不需要梯度的节点是否不会进入反向图；
3. 深度访问能否得到合法拓扑顺序；
4. 同一节点在分叉图中是否只执行一次；
5. 同一参数从多条路收到的梯度是否相加；
6. repeated backward 是否只累加叶子，不复用旧中间梯度；
7. reshape、transpose、contiguous 后梯度逻辑顺序是否正确；
8. 每个图操作的前向值和梯度是否与 PyTorch 一致；
9. 完整 Transformer 图每个命名参数的梯度是否一致；
10. GPU 图执行期间是否发生隐式 host copy。

从标量 loss 开始，图引擎：

```text
把 loss 梯度设为 1
→ 访问父节点建立拓扑顺序
→ 反向逐个执行 backward closure
→ 分叉处累加梯度
→ 叶子参数留下最终梯度
```

## 11. 自己写的反向 Kernel 怎样接入图

图引擎决定“现在轮到哪个节点”，算子负责自己的数学公式。

以 Softmax 为例。保存前向输出 `y`，收到上游梯度 `g` 后，一行的反向是：

```text
dot = sum(g[i] × y[i])
dx[i] = y[i] × (g[i] - dot)
```

CPU reference 用普通循环。HIP backward Kernel 在 GPU 上执行同一公式。图节点调用 `softmax_backward(output,gradient)`，不把 Tensor 转成 host vector。

当前完整 tiny GQA Transformer 的前向开始到 backward 结束，专门测试要求 host→device 和 device→host 调用数都等于 0。

## 12. Transformer 是怎样拼出来的

一个 Decoder block 的数据流：

```text
token id
  ↓ Embedding
hidden
  ├─ RMSNorm → Q/K/V → RoPE → causal Attention → output projection ─┐
  └─────────────────────────────────────────────────────────────────── add
                                                                      ↓
  ├─ RMSNorm → gate/up → SwiGLU → down projection ──────────────────┐
  └─────────────────────────────────────────────────────────────────── add
                                                                      ↓
                                                                  next hidden
```

最后经过 RMSNorm 和输出矩阵，得到每个位置对词表中每个 token 的 logits。

每条箭头都是 Tensor。每个方框都是要独立对照 PyTorch 的算子。每个分叉都要求 autograd 累加梯度。

## 13. MHA 和 GQA

MHA 中，每个 query head 有自己的 key/value head。GQA 中，多个 query head 共享较少的 key/value head。

框架用 repeat-interleave 把 K/V 展开到 query head 数量；反向时，把重复 head 的梯度加回原来的 K/V head。

必须满足：

```text
dimension % heads == 0
heads % kv_heads == 0
```

## 14. Loss 和训练

CrossEntropy 接收：

```text
logits  [..., classes]
targets [...]
```

targets 的 shape 必须等于 logits 去掉最后一维后的 shape。`-100` 表示这一行不参加 loss，SFT 用它屏蔽 prompt，只训练 response。

一次训练 step：

```text
清空旧梯度 → forward → loss → backward → AdamW 更新参数和动量
```

当前 FP32 Transformer 前向/反向图已经是 device-native。指标输出和 AdamW 仍是 correctness-first host 路径，所以“图不回 CPU”不能扩大成“整个训练 step 不回 CPU”。

## 15. Checkpoint 为什么不只是权重

恢复后想继续同一次实验，需要保存模型参数、AdamW 动量、step、数据游标、随机数状态以及模型和数据配置。缺少任何一项，恢复后的下一步都可能改变。

## 16. KV Cache 为什么只用于生成

生成第 100 个 token 时，前 99 个 token 的 K/V 已经算过。KV Cache 把每层过去的 K/V 留在内存中，下一步只追加一格。

验收不是“看起来更快”，而是逐个位置比较：

```text
cached logits ≈ full-prefix logits
```

## 17. 两张 GPU 怎样学习同一个模型

两张卡各看一部分 batch，会得到不同梯度。若各自直接更新，就会变成两个模型。RCCL all-reduce 汇总并发回相同梯度：

```text
GPU0 gradient ─┐
               ├─ 求和/平均 ─→ GPU0、GPU1 得到相同结果
GPU1 gradient ─┘
```

先证明两卡参数一致，再讨论 bucket 和通信重叠。

## 18. 正确性的四层证据

每个算子都要经过：

```text
手算小例子 → CPU reference → PyTorch oracle → HIP 对照
```

反向还要增加有限差分。测试同时覆盖正常 shape、边界 shape、非法 shape、极端数值、device 不匹配、不连续 view、前向精度和反向精度。

## 19. 精度不是一句“差不多”

比较两个数时使用：

```text
|actual-reference| <= atol + rtol × |reference|
```

`atol` 管接近 0 的误差，`rtol` 管数值变大后的相对误差。每个阈值都写在算子契约和 PyTorch parity 测试里，不能测试失败后随意放大。

## 20. 怎样安全增加一个新算子

1. 写输入、输出、shape、dtype、device 契约；
2. 写手算例子和非法输入；
3. 写 CPU reference；
4. 加 PyTorch oracle 和固定误差；
5. 写 readable HIP Kernel；
6. 做 CPU/HIP/PyTorch 对照；
7. 如果可求导，写 backward 和有限差分；
8. 接进模型；
9. 最后才优化；
10. 加入覆盖清单，否则审计失败。

## 21. 代码地图

```text
include/microllm/core       Storage、Tensor、View 接口
src/core                    CPU/HIP Tensor 实现
include/microllm/ops        算子公开接口
src/ops/ops.cpp             CPU reference 和设备分派
src/ops/hip                 readable HIP kernels
src/autograd                自研 eager reverse-mode 图引擎
src/model                   Decoder-only Transformer
src/training                Trainer、优化器、Checkpoint
src/inference               KV Cache 和生成
src/multi_gpu               RCCL 和梯度 bucket
tests/graph                 图构建、前向和反向专门测试
tests                       其余单元、错误和集成测试
python/tests                PyTorch oracle 与绑定测试
docs/development            里程碑证据和已知边界
```

## 22. 最后记住三句话

1. Tensor 是“内存 + 阅读方法”。
2. 自动求导图是“结果从哪里来”的记录，backward 按相反顺序把影响传回去。
3. 自己写 Kernel 不等于正确；hand value、CPU、PyTorch、HIP 和反例共同通过，才允许说这个 shape 已经验证。
