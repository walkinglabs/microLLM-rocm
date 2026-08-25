# Experiment 227 — 先证明矩阵积木能用，再搭 online Attention

Status: `admit bounded prototype; no model route`

## 为什么不直接写 Flash Attention

上一轮推理审计已经说明：继续改softmax线程数的理论收益太小，下一步必须让QK、online max/sum
和PV在tile里连续完成。但“用了rocWMMA”不自动等于更快。矩阵片段的尺寸、一个block放多少wave、
输入布局和短长上下文都会改变结果。

因此本轮只回答一个问题：MI300X上的BF16矩阵单元，能否在Attention的`Q[T,D] × Kᵀ[D,T]`
形状上正确并值得进入下一原型？rocWMMA官方将fragment、`load_matrix_sync`、`mma_sync`和
`store_matrix_sync`作为wave级公开接口；gfx942属于其CDNA matrix-core支持范围。

- [rocWMMA programming guide](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/conceptual/programmers-guide.html)
- [rocWMMA official repository](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma)

## 合同

```text
输入：Q为row-major BF16，K按Kᵀ需要的column-major BF16读取
计算：16×16或32×32 fragment，K方向每次16，FP32累加
输出：完整T×T FP32，不抽样
reference：CPU读取同一份已舍入BF16数据
对照：同输入标量HIP + 同语义预分配FP32 hipBLASLt
计时：10 warm-up + 50 Event，三个新进程取中位数
```

首轮布局扫描覆盖16/32 tile和1/2/4/8/16 waves/block。T512表明过大的block会退化；长上下文
最终选择32×32×16、一个wave/block。T16无法被32整除，保留16×16 fallback。其他非16倍数
当前明确拒绝，不假装已有tail处理。

## 48进程结果

16个shape、每格3进程、全部完整输出为零误差。关键边界如下：

| Shape | rocWMMA/scalar | rocWMMA/hipBLASLt |
|---|---:|---:|
| T512 D64 | 6.209× | 1.784× |
| T512 D128 | 11.605× | 1.654× |
| T1024 D64 | 19.633× | 1.666× |
| T1024 D128 | 29.684× | 1.342× |
| T2048 D64 | 32.982× | 1.022× |
| T2048 D128 | 43.399× | 0.688× |

![rocWMMA QK tile boundary](../assets/rocwmma-qk-tile.svg)

T2048 D128是必须保留的反例：矩阵单元远快于标量，不代表它胜过成熟GEMM库。当前kernel从
global memory重复加载Q/K并把全部T² score写回，48.6 TFLOPS级别仍没有吃满硬件。下一原型
只有通过“不写T²、在线合并softmax状态并复用V”才可能把这个劣势翻回来。

## 决定

- 保留独立benchmark、matrix runner和HIP smoke；
- 允许进入一个边界明确的online Attention原型；
- 不在`ops`、模型或CLI增加rocWMMA路由；
- 下一原型必须覆盖causal mask、GQA head mapping、D64/D128、T32–2048与tail fallback；
- 必须比较完整Attention输出、KV语义、峰值显存与Qwen/DeepSeek完整logits；
- 若T2048 D128在删除score写回后仍输给当前路径，停止这条设计，不靠只报告T512保留。

原始证据位于
[`benchmarks/results/2026-08-25-rocwmma-qk-tile/`](../../../benchmarks/results/2026-08-25-rocwmma-qk-tile/)。

发布回归为CPU 337/337、ASan/UBSan 335/335、PyTorch-enabled CPU 311/311、完整CPU/HIP
531/531（3个条件跳过）、HIP标签182/182、RCCL标签14/14和multi-GPU 12/12；覆盖清单注册
100个测试文件。
