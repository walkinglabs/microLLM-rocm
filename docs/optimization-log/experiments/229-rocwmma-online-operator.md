# Experiment 229 — benchmark里的快kernel怎样成为可用算子

Status: `admit model gate; default model route disabled`

## 不是复制一段kernel就结束

Experiment 228只在自己的二进制里知道输入布局。公共算子必须回答更多问题：batch放在哪一维、
V为什么是`[B,T,KV,D]`、非32倍数怎么办、不是gfx942怎么办、外部SDK怎样知道功能是否编译，
以及测试怎样证明走了正确分支。

新增API：

```cpp
online_causal_gqa_attention_bthd(
    query_bf16, key_bf16, value_bf16, repeats, scale)
```

输入为`Q[B,H,T,D]`、`K[B,KV,T,D]`、`V[B,T,KV,D]`的连续BF16，输出FP32
`[B,T,H,D]`。gfx942、T为32倍数、D64/128、T≤4096时调用online rocWMMA；其他合法情况显式
cast到FP32并调用当前`causal_gqa_attention_bthd`。

Config package公开`microLLM_WITH_ROCWMMA`，但rocWMMA是header-only私有实现，不成为外部
应用必须链接的库。native/fallback计数可清零并读取，测试不靠猜测dispatch。

## 三层数值门

- CPU BF16 fallback对当前FP32 reference；
- PyTorch把Q/K/V先舍入BF16，再用FP32 causal GQA；
- gfx942原生batch2完整输出对CPU，T33再验证fallback。

原生Max/RMS门为`2e-3 / 2e-4`，fallback为`3e-4 / 3e-5`。PyTorch operator parity通过；
batch2原生和T33 fallback均为零timed payload transfer。

## 42进程公共API矩阵

| Case | Route | candidate/current |
|---|---|---:|
| Qwen B1 T32 | native | 1.649× |
| Qwen B2 T512 | native | 2.456× |
| Qwen B1 T1024 | native | 1.859× |
| DeepSeek B1 T32 | native | 1.752× |
| DeepSeek B2 T512 | native | 2.290× |
| DeepSeek B1 T1024 | native | 1.636× |
| Qwen T31/T33 | fallback | 0.634× / 0.639× |
| DeepSeek B2 T33 | fallback | 0.696× |
| synthetic D32 | fallback | 0.607× |

![Public online operator](../assets/rocwmma-online-operator.svg)

10/10 native shape过1.05门；4个fallback全都精确路由、完整输出通过，同时全部保留性能反例。
fallback慢不是隐藏bug：它明确支付三次device cast，换来所有合法shape都有结果。

## 决定

- 公共operator、CMake feature metadata、native/fallback counters和测试保留；
- 不把T31/T33偷偷padding到32，也不删除fallback反例；
- 下一节点只做显式模型A/B：当前BTHD路径对online BF16 operator；
- 必须比较完整logits、token、峰值显存和端到端prefill，不只看Attention Event；
- 如果模型误差或端到端性能不通过，operator仍保留，模型路由拒绝。

原始证据位于
[`benchmarks/results/2026-08-25-rocwmma-online-operator/`](../../../benchmarks/results/2026-08-25-rocwmma-online-operator/)。

发布回归为CPU 340/340、ASan/UBSan 338/338、PyTorch-enabled CPU 314/314、完整CPU/HIP
536/536（3个条件跳过）、HIP标签184/184；覆盖清单注册102个测试文件，CPU/HIP CMake package
三条外部消费门均为3/3，RCCL标签14/14、multi-GPU 12/12。
