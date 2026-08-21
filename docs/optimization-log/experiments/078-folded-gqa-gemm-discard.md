# Experiment 078 — 不复制K/V更快，但完整logits不再对齐

Experiment 077的last-logit profile中，长GQA仍把每个KV head物理复制到多个query head。
候选利用head连续布局，把共享同一K/V的多个query head折进GEMM行维：

```text
旧：K/V [B,KV,T,D] → repeat → [B,H,T,D] → H组T×D GEMM
新：Q [B,KV,R,T,D] → [B×KV,R×T,D]，K/V保持不复制
```

计算量、softmax、dtype和last-logit接口都不变。反驳条件是任一官方模型完整logits超过已有
`max_abs 1e-4 / RMSE 1e-5`门，即使top token不变也必须拒绝。

## 性能候选

T2048 B8、last-logit、三进程中位数：

| 模型 | reference | folded | 加速 | peak下降 | folded/PyTorch |
|---|---:|---:|---:|---:|---:|
| Qwen | 129,815 | 135,372 tok/s | 1.043× | 3.19% | 0.402× |
| DeepSeek | 66,444 | 71,369 tok/s | 1.074× | 3.52% | 0.562× |

候选确实减少临时K/V和分配，两个模型方向一致。focused HIP Attention、MHA/GQA、shape和
CPU token门都通过；如果只看吞吐或top-1，它会被错误接受。

rocprof证明机制而不是偶然wall time：Qwen的repeat Kernel从192 calls/10.90ms降到0，
DeepSeek从224 calls/22.70ms降到0，正好等于`layers × K/V两份 × 4 forwards`。Kernel总时间
下降4.0%/6.0%，softmax保持不变；GEMM虽然调用数不变，分组shape也让时间下降3.2%/6.3%。
因此性能解释成立，失败只来自不能接受的数值代价。

## 反驳实验：151,936个logit逐项比较

reference使用已推送的`ef6fe1e`独立Release build，候选使用同环境新build：

| 模型 | context/batch | max abs | RMSE | top token |
|---|---|---:|---:|---:|
| Qwen | 2048/8 | 0.07345 | 0.01567 | 9707 = 9707 |
| DeepSeek | 2048/8 | 0.05632 | 0.01187 | 30 = 30 |

两项误差都比门大约三个数量级。原因不是公式不同，而是把`R`个head折进同一大矩阵后，
hipBLASLt选择了不同shape/算法与浮点累加顺序；这些差异经过28层放大。当前证据只证明top-1
恰好未变，不能证明采样、长生成或其他prompt安全。

![Folded GQA discarded](../assets/folded-gqa-discard.svg)

## 决定

`discard`。候选源码回退，正式raw、性能收益和精度失败全部保留在
[`078-data`](078-data/)。这个结果否定了“数学等价reshape就可以自动替换官方模型路径”的
假设，也再次说明top token不是完整精度门。

下一节点不继续改变QK/PV GEMM的规约形状。应对已成为显式热点的causal softmax做局部优化，
并在第一轮就运行T2048 B8完整logits，而不是等到性能测完才检查。
