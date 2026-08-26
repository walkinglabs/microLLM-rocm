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

非拥有`TensorView`的add/multiply共同域是连续、同shape、同device、同dtype的FP32/FP16/BF16。
它不会广播、转换dtype或接管pointer。PyTorch Custom Op用`at::empty_like`创建输出，只把当前
HIP Stream的非拥有handle传给microLLM；Autograd公式明确注册，Meta dispatch只验证合同并返回
同shape/dtype的meta输出。真实ROCm矩阵必须同时报告完整输出/梯度、Event/wall和allocator peak，
不能把“成功注册”写成“比原生Torch快”。
HIP Auto只在FP16/BF16、`numel >= 4,194,304`且left/right/output均16-byte aligned时使用
vector16；尾部在同一Kernel安全处理。FP32、小Tensor或任一未对齐pointer必须走scalar，不能为了
满足对齐条件复制。broad vector策略因FP32回退到0.845×–0.879×已经被反例拒绝。
PyTorch fused SwiGLU的forward共同域是连续同shape/device/dtype的FP32/FP16/BF16；输出由
`at::empty_like`拥有。FP32 Autograd调用caller-owned fused backward；低精度暂用PyTorch公式并按
FP16 `4e-3`、BF16 `6.25e-2`报告Max/RMS。16M forward性能可被引用，F+B性能失败必须同时保留。
FP32 backward的float4候选已经因0.946×–1.039×被删除；当前只保留scalar fused producer。
任何新优化先检查output gradient的stride/storage，而不是重新引入已拒绝的向量selector。
`swiglu_backward_scalar_seed_out_`只接受一个FP32元素、同device、连续且不alias，输出shape仍等于
gate/up。PyTorch bridge只有在FP32 gradient `numel>0`且全部stride为0时使用；其他布局禁止偷读
首元素。这个合同对应`sum()`广播，不代表mean或weighted backward。
PyTorch SwiGLU Autograd由C++ Function实现，FP32调用上述fused producers；FP16/BF16在
`NoGradGuard`内使用in-place ATen公式。它不承诺double backward。FakeTensor无backing pointer时只
返回Meta shape，不能把null pointer包装成Storage。
`swiglu_backward_typed_out_`要求gate/up/gradient/两个output为同shape/device/dtype且连续，dtype仅
FP16/BF16；FP32计算后每个output舍入一次。它不alias、不广播、不接受zero-stride；scalar-seed仍是
独立FP32合同。

## BF16 weight gradient

`bf16_weight_gradient(input_fp32, output_gradient_fp32)` 接受两个连续二维 FP32 Tensor：
`[rows, hidden]` 与 `[rows, width]`。它先把两个操作数舍入为 BF16，再计算
`inputᵀ @ output_gradient`，累加和输出保持 FP32，结果 shape 为 `[hidden, width]`。

CPU 是可读参考；HIP 使用 cast+transpose、cast 和 hipBLASLt。PyTorch oracle 使用相同的
BF16 舍入语义。它不承诺与 FP32 gradient bit-exact，也不会自动改变 Autograd。
20-step模型门失败后，gate/up Autograd/CLI路由已经删除；query/KV也有稳定性能反例。
因此这个API是独立算子原语，不是当前训练精度策略。

## 前向算子

| microLLM | 输入和输出 shape | PyTorch oracle | FP32 阈值 | 必测非法输入 |
|---|---|---|---|---|
| `fill_` | 任意 Tensor，shape 不变 | `torch.full` | 精确 | 非 FP32 |
| `add` | `S,S -> S`；caller-owned/Custom Op共同域支持FP32/FP16/BF16 | `torch.add` | 默认/低精度逐项舍入 | dtype/shape/device 不同、HIP 非连续 |
| `add_bias` | input `[...,D]`、bias `[D]`，输出不变 | `input+bias` | 默认 | rank/shape/device 错 |
| `add_bias_bf16` | BF16 input `[...,D]`、FP32 bias `[D]`，输出BF16 | `(input.float()+bias).bfloat16()` | 逐项按BF16舍入一致 | dtype/rank/shape/device/连续性错 |
| `multiply` | `S,S -> S`；caller-owned/Custom Op共同域支持FP32/FP16/BF16 | `torch.mul` | 默认/低精度逐项舍入 | dtype/shape/device 不同、HIP 非连续 |
| `multiply_out_` | 同`multiply`，output预分配且不与输入共享Storage | `torch.mul` + caller地址检查 | 默认/BF16舍入 | output shape/dtype/device/stride或alias错 |
| `scale` | `S,scalar -> S` | `x*scalar` | 默认 | 非 FP32、HIP 非连续 |
| `scale_in_place_` | 连续浮点Tensor原地乘有限factor，Storage地址不变 | `x*factor` | FP32默认/低精度舍入 | 非浮点、非连续、Inf/NaN factor |
| `matmul` | `[...,M,K] × [...,K,N] -> [...,M,N]`，batch 维完全相同 | `torch.matmul` | `2e-4,2e-4` | rank<2、rank/batch/inner 不同 |
| `matmul_out_` | 数学shape同`matmul`；output预先分配、连续、同dtype/device且不与输入共享Storage | `torch.matmul` + caller地址检查 | `2e-4,2e-4` | output shape/dtype/device/stride错误或alias |
| `bf16_ffn_precast_out_` | caller已填`workspace.input_bf16 [R,D]`，三个BF16权重，输出FP32 | 与`bf16_ffn_out_`同输出，但不重复cast | 完整BF16中间值和FP32输出相同 | workspace/weight/output shape、dtype、device、stride或alias错 |
| `bf16_qkv_projection_precast_out_` | caller已填BF16 input，Q/K/V caller输出与workspace | 与`bf16_qkv_projection_out_`同输出/保留语义 | 完整投影输出相同 | retain关系、workspace/weight/output shape/dtype/device/alias错 |
| `matmul_scaled_with_implementation` | 与 matmul/transpose 合同相同，输出再乘有限 factor | `(op(A)@op(B))*factor` | `2e-4,2e-4` | matmul 合同错、factor 为 Inf/NaN |
| `embedding` | weight `[V,D]`，index `S`，输出 `S+[D]` | `F.embedding` | 默认 | weight 非二维、index 非 Int32/越界、设备不同 |
| `softmax` | `[...,D] -> [...,D]`，仅最后一维；HIP支持FP32/FP16/BF16 | `torch.softmax(x,-1)` | FP32 `2e-6,2e-5`；低精度逐dtype门 | 空最后维、非最后维、非连续 |
| `softmax_typed_out_` | FP16/BF16同shape/dtype/device caller输出；FP32 reduction后只舍入output；width≤32 serial，之后block；2048–8192用≤32KiB LDS cache及1024-thread wave | `torch.softmax(x,-1)` | FP16 Max≤`5e-4`、BF16≤`4e-3`；边界1/17/32/33/64/65/128/129/1024/2047/2048/4096/8192/8193 | FP32/FP8、dim、shape/device/alias/stride错 |
| `torch.ops.microllm.softmax` | functional最后一维Softmax；FP32/FP16/BF16；CPU/ROCm/Meta/C++ Autograd；新output | `torch.softmax(x,-1)`及同seed梯度 | 同上；10格forward、三dtype梯度、current Stream/fullgraph | scalar、整数、非连续；wide性能不作普遍领先声明 |
| `torch.ops.microllm.softmax_out` | `Tensor(a!)` caller output，返回同pointer；inference-only | `torch.softmax(x,-1,out=y)` | 10格Max/RMS、pointer、native/custom peak均0 | requires-grad、alias、shape/dtype/device/stride错 |
| `rms_norm` | input `[...,D]`，weight `[D]` | `F.rms_norm` | `2e-4,2e-4` | weight shape 错、epsilon<=0 |
| `rms_norm_out_` / `rms_norm_bf16_out_` | caller输出与input同shape/device；前者FP32、后者BF16 | GPU `rms_norm` 后可选`bfloat16()` | FP32同路径；BF16逐位相同 | output dtype/shape/device/stride/alias错，weight/epsilon错 |
| `add_rms_norm` | left/right `[...,D]`，weight `[D]`，返回 sum/norm 两个 Tensor | `s=x+y; (s,F.rms_norm(s))` | `2e-4,2e-4` | shape/dtype/device/weight/epsilon 错 |
| `silu` | `S -> S` | `F.silu` | 默认 | 非 FP32 |
| `swiglu` | gate/up 都是 `S` | `F.silu(gate)*up` | `2e-6,2e-5` | shape/device 不同 |
| `swiglu_with_implementation` / `_out_` | 同shape/dtype/device；Vectorized仅对8-byte aligned HIP BF16，caller-output地址不变 | 先按BF16舍入再`F.silu(gate.float())*up.float()`并舍入BF16 | scalar/vector逐项位级相同 | Vectorized非BF16/非HIP/未对齐、非连续、output shape/dtype/device/alias错 |
| `rope` | `[...,T,...,H] -> same`，H 是偶数 | PyTorch `sin/cos` 组合 | `2e-5,2e-5` | rank<2、H 奇数、sequence_dim 错、offset/base 错 |
| `rope_split_half_bias` | input `[B,H,T,D]`、bias `[H*D]`，D 偶数 | `rope_split_half(x+bias.view(1,H,1,D))` | `3e-5,3e-5` | 非 FP32、rank/shape/device 错、offset/base 错 |
| `rope_split_half_bias_bthd` | FP32/BF16 input `[B,T,H,D]`、FP32 bias `[H*D]`，输出FP32 `[B,H,T,D]` | `rope_split_half((x.float()+bias.view(1,1,H,D)).transpose(1,2))` | FP32 `3e-5,3e-5`；BF16按输入舍入后同值 | 非连续、F16/FP8、rank/shape/device 错、offset/base 错 |
| `rope_split_half_bias_bthd_bf16` | BF16 input `[B,T,H,D]`、FP32 bias，直接输出BF16 `[B,H,T,D]` | 上项结果再`bfloat16()` | 逐项按BF16舍入一致 | 非BF16、非连续、rank/偶数D/bias/device/offset/base错 |
| `causal_softmax_with_implementation` | `Rows128`只接受HIP FP32方阵且T=256..1024；Auto保持原路由 | 与`causal_softmax`逐项比较，mask严格为0、可见行和为1 | `2e-6,1e-7` | CPU、T&lt;256、T&gt;1024、非方阵/非连续拒绝 |
| `repeat_interleave_bf16_to_float` | BF16输入，任意rank/dim，FP32输出 | `repeat_interleave(ops::cast(x, FP32), dim, repeats)` | 逐项完全相等 | 非BF16、坏dim、repeats≤0、overflow拒绝 |
| `cross_entropy` | logits `S+[C]`，targets `S`，输出 scalar | `F.cross_entropy(ignore_index=-100)` | `2e-5,2e-5` | target shape/dtype/device 错、无有效 target |
| `reduce_sum` | `S -> scalar` | `torch.sum` | `2e-5,2e-5` | 非 FP32、HIP 非连续 |
| `broadcast_scalar` | scalar + 目标 shape `S -> S` | `scalar.expand(S).clone()` | 精确 | source 不是单元素 |
| `causal_softmax` | `[...,T,T] -> same` | 上三角 mask 后 softmax | `2e-6,2e-5` | rank<2、末两维不等、T=0 |
| `attention_probability_value_bthd` | probability `[B,H,T,T]`、value `[B,T,H,D]`，输出 `[B,T,H,D]` | `(P @ V.transpose(1,2)).transpose(1,2)` | `3e-5,3e-5` | 非连续/非 FP32、B/H/T/device 不匹配 |
| `attention_probability_value_gqa_bthd` | P `[B,H,T,T]`、V `[B,T,KV,D]`、`H=KV×R`，输出 `[B,T,H,D]` | 先在 dim2 repeat V，再用上项 | `3e-5,3e-5` | B/H/KV/T/device/连续性或 R 错 |
| `causal_gqa_attention_bthd` | Q `[B,H,T,D]`、K `[B,KV,T,D]`、V `[B,T,KV,D]`，输出 `[B,T,H,D]` | causal GQA 后 `transpose(1,2)` | 前向 `3e-5`、整图 `2e-3` | 连续性、B/T/D/device、`H=KV*repeats` 或 scale 错 |
| `online_causal_gqa_attention_bthd` | BF16 Q/K/V沿用上项布局，输出FP32；gfx942上T整32且D64/128走online rocWMMA，其余走FP32 fallback | 输入先按BF16舍入，PyTorch FP32 causal GQA | native Max `2e-3`、RMS `2e-4`；fallback `3e-5` | 非BF16、连续性、B/T/D/device、head关系或scale错；tail/batch必须fallback |
| `repeat_interleave` | 维 d 从 D 变为 `D×repeats` | `torch.repeat_interleave` | 精确 | dim 越界、repeats<=0、溢出 |
| `repeat_gqa_kv_bthd` | K `[B,KV,T,D]`、V `[B,T,KV,D]` → H=KV×R 两个布局 | 分别在 dim1/dim2 `repeat_interleave` | 精确 | B/KV/T/D/device/连续性或 R 错 |

`matmul_with_implementation` 的 `Readable` 和 `HipBLASLt` 是同一数学契约的不同执行办法；二者必须通过同一个 oracle。选择器和注册表只决定实现，不能改变 shape 或数值含义。
`matmul_out_`也不增加新公式：PyTorch门仍是同一`torch.matmul`。它新增的是状态合同，因此CPU/HIP
还必须检查调用前后output Storage地址不变、timed payload transfer为零，并拒绝任何输入alias。
`SwiGLUImplementation::Auto`仍是scalar。`Vectorized`是显式研究路径：它在MI300X的两个
T1024 operator shape上更快，但未通过两模型默认性能门，所以不能把“设备可执行”写成
“模型默认受益”。

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
| `rope_split_half_bias_bthd_backward` | seed `[B,H,T,D] -> [B,T,H,D]` | 上述 PyTorch 图的 input gradient | `3e-5,3e-5` | 非连续/非 FP32、rank/偶数 D/offset/base 错 |
| `cross_entropy_backward` | 与 logits 相同 | `F.cross_entropy(...).backward(seed)` | `3e-5,3e-5` | seed 非 scalar、target 契约错 |
| `causal_softmax_backward` | 与 output 相同 | masked softmax `.backward(seed)` | `3e-5,3e-5` | output/seed shape 错、非方阵 |
| `attention_probability_gradient_bthd` | dO/V `[B,T,H,D] -> [B,H,T,T]` | `dO.transpose(1,2) @ V.transpose(1,2).T` | `3e-5,3e-5` | shape/dtype/device/连续性错 |
| `attention_probability_gradient_gqa_bthd` | dO `[B,T,H,D]`、V `[B,T,KV,D]`、`H=KV×R` → dP | repeat V 后用上项 | `3e-5,3e-5` | B/H/KV/T/D/device/连续性或 R 错 |
| `attention_value_gradient_bthd` | P `[B,H,T,T]`、dO `[B,T,H,D] -> [B,T,H,D]` | `(P.T @ dO.transpose(1,2)).transpose(1,2)` | `3e-5,3e-5` | B/H/T/dtype/device/连续性错 |
| `repeat_interleave_backward` | 原 input shape | `repeat_interleave(...).backward(seed)` | `2e-5,2e-5` | gradient 与推导 shape 不同 |
| `repeat_gqa_kv_bthd_backward` | dK `[B,H,T,D]`、dV `[B,T,H,D]` → 两个 KV 布局 | 分组 reshape 后沿 repeat 维求和 | 精确 | B/H/T/D/device/连续性或整除错 |

## Autograd 图操作

| 图操作 | 前向 shape | backward 重点 |
|---|---|---|
| `add` | 两输入同 shape | 两边收到上游梯度；分叉要累加 |
| `multiply` | 两输入同 shape | `g*right` 与 `g*left` |
| `scale` | shape 不变 | 梯度乘 scalar |
| `matmul` | 遵守 matmul 契约 | 两边分别使用转置矩阵乘 |
| `bf16_gate_up_projection` | 一个FP32输入与两个同shape BF16 mirror，返回两个FP32输出 | input累加两条FP32 master梯度；gate/up各收一条weight梯度 |
| `bf16_qkv_projection` | 一个FP32输入与三个兼容BF16 mirror，返回Q/K/V三个FP32输出 | input累加三条FP32 master梯度；三个weight边保持独立 |
| `sum` | 任意 shape 到 scalar | scalar seed 广播回原 shape |
| `mean` | 非空 shape 到 scalar | sum backward 再除元素数 |
| `reshape` | 元素总数不变 | 梯度恢复原 shape |
| `transpose` | 交换两维 | backward 再交换相同两维 |
| `embedding` | `S -> S+[D]` | 重复 index 必须 scatter-add |
| `softmax` | shape 不变 | 使用保存的前向 output |
| `rms_norm` | shape 不变 | input 与 weight 都有梯度 |
| `add_rms_norm` | left/right 同 shape，返回 sum 与 normalized 两个结果 | sum 节点先合并残差支路和归一化支路梯度，再把同一总梯度送给 left/right；weight 单独累加 |
| `silu` | shape 不变 | 极大正负输入仍有限 |
| `swiglu` | shape 不变 | gate/up 两条父边都有梯度 |
| `rope` | shape 不变 | 旋转矩阵的转置作用于 seed |
| `rope_split_half_bias` | `[B,H,T,D]` 不变 | input 收到逆旋转梯度；bias 对 B/T 求和 |
| `rope_split_half_bias_bthd` | `[B,T,H,D] -> [B,H,T,D]` | input 梯度直接写回 BTHD；bias 对 B/T 求和；图中不出现布局物化节点 |
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
- multi-tensor AdamW：workspace固定Tensor元素数和HIP设备；每步地址可以变化；缺失gradient
  必须保持该Tensor的parameter、两组moment和mirror全部不变。完整状态容差`2e-6`，metadata
  copy必须与更新Kernel在同一Stream，不能把失败的模型路由伪装成默认优化。
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
