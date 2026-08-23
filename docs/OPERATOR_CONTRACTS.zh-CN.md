# 算子契约、PyTorch 对照与测试门

这份文档是算子的验收合同。教学文档解释“为什么”，这里规定“什么输入合法、输出长什么样、怎样证明一致”。

## 共同规则

- 当前训练主路径只接受 FP32 数据；token/target 使用 Int32。
- 同一个算子的 Tensor 输入必须位于同一设备。
- readable HIP 算子要求连续输入；view 先显式 materialize。
- 不自动广播。`add` 和 `multiply` 要求 shape 完全相等。
- 数值比较使用 `abs(actual-ref) <= atol + rtol*abs(ref)`。
- 默认元素级阈值是 `atol=1e-6, rtol=1e-5`。
- PyTorch 支持更多广播、dtype 和维度；microLLM 第一版故意更窄。测试比较共同支持域，并验证契约外输入会被拒绝。

## 前向算子

| microLLM | 输入和输出 shape | PyTorch oracle | FP32 阈值 | 必测非法输入 |
|---|---|---|---|---|
| `fill_` | 任意 Tensor，shape 不变 | `torch.full` | 精确 | 非 FP32 |
| `add` | `S,S -> S` | `torch.add` | 默认 | shape/device 不同、HIP 非连续 |
| `add_bias` | input `[...,D]`、bias `[D]`，输出不变 | `input+bias` | 默认 | rank/shape/device 错 |
| `multiply` | `S,S -> S` | `torch.mul` | 默认 | shape/device 不同、HIP 非连续 |
| `scale` | `S,scalar -> S` | `x*scalar` | 默认 | 非 FP32、HIP 非连续 |
| `matmul` | `[...,M,K] × [...,K,N] -> [...,M,N]`，batch 维完全相同 | `torch.matmul` | `2e-4,2e-4` | rank<2、rank/batch/inner 不同 |
| `embedding` | weight `[V,D]`，index `S`，输出 `S+[D]` | `F.embedding` | 默认 | weight 非二维、index 非 Int32/越界、设备不同 |
| `softmax` | `[...,D] -> [...,D]`，仅最后一维 | `torch.softmax(x,-1)` | `2e-6,2e-5` | 空最后维、非最后维 |
| `rms_norm` | input `[...,D]`，weight `[D]` | `F.rms_norm` | `2e-4,2e-4` | weight shape 错、epsilon<=0 |
| `add_rms_norm` | left/right `[...,D]`，weight `[D]`，返回 sum/norm 两个 Tensor | `s=x+y; (s,F.rms_norm(s))` | `2e-4,2e-4` | shape/dtype/device/weight/epsilon 错 |
| `silu` | `S -> S` | `F.silu` | 默认 | 非 FP32 |
| `swiglu` | gate/up 都是 `S` | `F.silu(gate)*up` | `2e-6,2e-5` | shape/device 不同 |
| `rope` | `[...,T,...,H] -> same`，H 是偶数 | PyTorch `sin/cos` 组合 | `2e-5,2e-5` | rank<2、H 奇数、sequence_dim 错、offset/base 错 |
| `rope_split_half_bias` | input `[B,H,T,D]`、bias `[H*D]`，D 偶数 | `rope_split_half(x+bias.view(1,H,1,D))` | `3e-5,3e-5` | 非 FP32、rank/shape/device 错、offset/base 错 |
| `cross_entropy` | logits `S+[C]`，targets `S`，输出 scalar | `F.cross_entropy(ignore_index=-100)` | `2e-5,2e-5` | target shape/dtype/device 错、无有效 target |
| `reduce_sum` | `S -> scalar` | `torch.sum` | `2e-5,2e-5` | 非 FP32、HIP 非连续 |
| `broadcast_scalar` | scalar + 目标 shape `S -> S` | `scalar.expand(S).clone()` | 精确 | source 不是单元素 |
| `causal_softmax` | `[...,T,T] -> same` | 上三角 mask 后 softmax | `2e-6,2e-5` | rank<2、末两维不等、T=0 |
| `repeat_interleave` | 维 d 从 D 变为 `D×repeats` | `torch.repeat_interleave` | 精确 | dim 越界、repeats<=0、溢出 |

`matmul_with_implementation` 的 `Readable` 和 `HipBLASLt` 是同一数学契约的不同执行办法；二者必须通过同一个 oracle。选择器和注册表只决定实现，不能改变 shape 或数值含义。

## 反向原语

反向 reference 由 PyTorch 对对应前向执行 autograd 得到，不复制一份 microLLM 公式充当答案。

| microLLM backward | 返回 shape | PyTorch reference | FP32 阈值 | 必测非法输入 |
|---|---|---|---|---|
| `embedding_backward` | `[V,D]` | `F.embedding(...).backward(seed)` | `2e-5,2e-5` | gradient/index 数量错、V<=0、dtype/device 错 |
| `embedding_backward_add_` | 在调用者 `[V,D]` 中累加token行 | dense `embedding_backward + add` | `2e-5,2e-5` | destination/gradient/index shape、dtype、device或连续性错 |
| `bias_gradient` | `[...,D] -> [D]`，前面各维求和 | `(input+bias).backward(seed)` | Max `3e-5`、RMS `1e-5` | scalar、非 FP32、HIP 非连续 |
| `softmax_backward` | 与 output 相同 | `softmax(...).backward(seed)` | `2e-5,2e-5` | output/seed shape/device 不同 |
| `rms_norm_backward` | input grad `[...,D]`，weight grad `[D]` | `F.rms_norm(...).backward(seed)` | `3e-4,3e-4` | input/weight/seed 契约错 |
| `silu_backward` | 与 input 相同 | `F.silu(...).backward(seed)` | `2e-5,2e-5` | shape/device 不同 |
| `swiglu_backward` | gate/up 各一个梯度 | `(F.silu(gate)*up).backward(seed)` | `2e-5,2e-5` | 三者 shape/device 不同 |
| `rope_backward` | 与 seed 相同 | PyTorch RoPE 图 `.backward(seed)` | `3e-5,3e-5` | 与前向相同的配置错误 |
| `cross_entropy_backward` | 与 logits 相同 | `F.cross_entropy(...).backward(seed)` | `3e-5,3e-5` | seed 非 scalar、target 契约错 |
| `causal_softmax_backward` | 与 output 相同 | masked softmax `.backward(seed)` | `3e-5,3e-5` | output/seed shape 错、非方阵 |
| `repeat_interleave_backward` | 原 input shape | `repeat_interleave(...).backward(seed)` | `2e-5,2e-5` | gradient 与推导 shape 不同 |

## Autograd 图操作

| 图操作 | 前向 shape | backward 重点 |
|---|---|---|
| `add` | 两输入同 shape | 两边收到上游梯度；分叉要累加 |
| `multiply` | 两输入同 shape | `g*right` 与 `g*left` |
| `scale` | shape 不变 | 梯度乘 scalar |
| `matmul` | 遵守 matmul 契约 | 两边分别使用转置矩阵乘 |
| `sum` | 任意 shape 到 scalar | scalar seed 广播回原 shape |
| `mean` | 非空 shape 到 scalar | sum backward 再除元素数 |
| `reshape` | 元素总数不变 | 梯度恢复原 shape |
| `transpose` | 交换两维 | backward 再交换相同两维 |
| `embedding` | `S -> S+[D]` | 重复 index 必须 scatter-add |
| `softmax` | shape 不变 | 使用保存的前向 output |
| `rms_norm` | shape 不变 | input 与 weight 都有梯度 |
| `silu` | shape 不变 | 极大正负输入仍有限 |
| `swiglu` | shape 不变 | gate/up 两条父边都有梯度 |
| `rope` | shape 不变 | 旋转矩阵的转置作用于 seed |
| `rope_split_half_bias` | `[B,H,T,D]` 不变 | input 收到逆旋转梯度；bias 对 B/T 求和 |
| `cross_entropy` | logits 到 scalar | ignored row 梯度必须全 0 |
| `contiguous` | shape 不变 | view 的逻辑顺序不能改变 |
| `causal_softmax` | 方阵不变 | future 前向和反向都为 0 |
| `repeat_interleave` | 一维扩大 | 重复位置梯度求和回原元素 |

独立的 `tests/graph/` 还要用相同权重构建 CPU、PyTorch 和 HIP 图，比较 loss、节点 shape、全部命名参数梯度，并拒绝 GPU 图中的隐式 host copy。整图门使用 `atol=2e-3, rtol=2e-3`，不能拿这个较宽阈值代替单算子门。

## Tensor/View shape 门

| API | 合法条件 | 必须保持 |
|---|---|---|
| `transpose(d0,d1)` | 两维存在 | Storage 共享、numel 不变、stride 正确 |
| `slice(dim,start,end,step)` | dim/range 合法且 step>0 | offset/shape/stride 正确 |
| `reshape(shape)` | 元素数相同且输入连续 | Storage 共享 |
| `contiguous()` | 任意合法 view | 逻辑值顺序不变，结果连续 |
| `unsqueeze(dim)` | 插入位置合法 | numel和值不变 |
| `squeeze(dim)` | 指定维大小为 1 | numel和值不变 |
| `to(device)` | 目标设备存在 | shape/dtype/逻辑值不变 |

## 优化器和模型级 PyTorch 门

- SGD：一步更新与 `torch.optim.SGD` 对齐。
- AdamW：至少比较前两步参数、一阶动量、二阶动量和恢复后的下一步。
- AdamW 优化实现：Scalar/Vectorized 必须对齐参数、两组 moment、BF16 mirror 和非 4
  倍数尾部；显式 Vectorized 的算子收益不能改变 `Auto`，除非官方模型矩阵也通过。
- batched GEMM：两个输入 rank、所有 batch 维、dtype、device 必须相同；transpose 只作用
  于最后两维。hipBLASLt 必须与 materialized CPU reference 对齐，并单独记录 Stream 依赖。
- Transformer：相同权重、token、mask、RoPE 和 epsilon 下比较 logits、loss 和全部参数梯度。
- KV Cache：full-prefix reference 与 cache 路径逐位置比较 logits；分叉row必须分别等于独立B1，
  `row_positions`、shared Storage地址和logical max-prefix同时检查。serial oracle不能写成并行性能。
- SFT：`-100` prompt mask 与 PyTorch ignored target 行为一致。

## “全部覆盖”的机器规则

1. `ops.h` 新增公开算子时，算子清单必须新增同名项。
2. 每项必须声明 valid shape、invalid shape 和 PyTorch oracle 测试。
3. `tests/`、`python/tests/` 新增测试文件时，必须进入 CMake/CTest 和测试清单。
4. 文档阈值必须与测试常量一致。
5. CPU 环境运行结构、shape 和 PyTorch CPU oracle；HIP 环境再运行三方对照。

缺失时，覆盖审计必须报告具体算子或测试文件，不能只给一个模糊百分比。
