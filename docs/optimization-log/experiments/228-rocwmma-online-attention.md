# Experiment 228 — 不保存整张分数表的 online Attention

Status: `admit operator integration with fallback; no model route`

## 从上一块积木继续

Experiment 227只证明rocWMMA QK tile可用。本轮要把四步接起来：

```text
32×32 BF16 QK（rocWMMA，FP32累加）
→ causal mask + online max/sum
→ FP32权重显式舍入BF16
→ 32×32 BF16 PV（rocWMMA，载入旧FP32 accumulator）
```

每个key tile到来时，旧输出先乘`exp(old_max-new_max)`，新权重再累加；最后除以online sum。
因此candidate只用每block 14.4KiB（D64）或22.4KiB（D128）shared memory，不写全局T²。

## 两次失败怎样改变设计

第一版只有一个64-thread wave，QK虽快，标量PV把T512/D128拖到标量fused的0.047×。把PV并行到
128线程时，完整输出门又发现0.029误差：循环仍按64步进，两个wave重叠写shared output。改成模板
worker步长后全量对齐，但512-thread标量PV也只有0.172×。

这两个失败阻止了“多线程就会快”的假结论。最终改为2个wave处理D64的PV、4个wave处理D128，
概率tile显式转BF16后由rocWMMA计算。单head仍缺并行度；加入真实GQA网格后，每个query head有
独立query tile，K/V按head group共享，才达到足够占用率。

## 正式合同

- Qwen式：H14、KV2、D64；DeepSeek式：H12、KV2、D128；
- T32/64/128/256/512/1024/2048；
- 每格3个fresh processes，5 warm-up + 20 Event；
- CPU、标量BF16 fused、online rocWMMA、当前框架四条路径；
- candidate Max≤2e-3、RMS≤2e-4；当前/标量Max≤3e-4、RMS≤3e-5；
- 必须验证全部`H×T×D`输出，不抽点；
- current对照使用同一批已舍入BF16数值的FP32表示。

## 42进程结果

| Shape | online/current | online/scalar | Removed score |
|---|---:|---:|---:|
| Qwen T512 | 1.526× | 0.585× | 14.0 MiB |
| Qwen T1024 | 2.487× | 1.151× | 56.0 MiB |
| Qwen T2048 | 1.673× | 2.210× | 224.0 MiB |
| DeepSeek T512 | 4.041× | 1.032× | 12.0 MiB |
| DeepSeek T1024 | 2.216× | 1.850× | 48.0 MiB |
| DeepSeek T2048 | 1.260× | 2.910× | 192.0 MiB |

![rocWMMA online Attention](../assets/rocwmma-online-attention.svg)

14/14 shape都胜当前框架，最小为DeepSeek T2048的1.260×。candidate最大Max/RMS为
`5.66e-4 / 1.16e-4`，没有随上下文变长而发散。当前路径仍精确到约1e-7，所以这是明确的
低精度取舍，不能叫bit-exact替换。

短上下文相对独立标量fused仍只有0.337×–0.678×，证明candidate不是所有参考实现中最快；它胜
当前框架也包含删除layout/分配/全局score的系统收益。

## 决定

- 保留benchmark、真实GQA矩阵、完整误差门和短标量反例；
- 下一节点允许建立公共HIP operator，但必须让不整32、batch>1、非gfx942和缺rocWMMA时显式fallback；
- operator测试必须和CPU、当前HIP、PyTorch完整输出对齐；
- 本轮不改模型、CLI、Autograd或默认dispatch；
- 只有公共operator与Qwen/DeepSeek完整logits/显存/吞吐都过门，模型路由才可讨论。

原始证据位于
[`benchmarks/results/2026-08-25-rocwmma-online-attention/`](../../../benchmarks/results/2026-08-25-rocwmma-online-attention/)。

发布回归为CPU 338/338、ASan/UBSan 336/336、PyTorch-enabled CPU 312/312、完整CPU/HIP
533/533（3个条件跳过）、HIP标签183/183、RCCL标签14/14与multi-GPU 12/12；覆盖清单注册
101个测试文件。
