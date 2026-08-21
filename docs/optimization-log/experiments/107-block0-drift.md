# Experiment 107 — Attention全部exact，首个差异在FFN output

本节点只给block 0加12个子阶段记录。三个fresh B1/B2 pair、43个stage全部稳定，B2重复行每个stage
逐值exact。

![Block0 drift](../assets/block0-drift.svg)

## 结果

从attention norm到FFN norm的11个子阶段max/mean/RMS/relative-L2全部为0，包括Q/K/V projection。
第一个非零stage是`inference.blocks.0.ffn.output`：

```text
max-abs      0.0013504028
mean-abs     0.0000102433
RMS          0.0000502914
relative-L2  0.0000726920
```

block output的绝对误差几乎相同，因为它只是把不同的FFN output加到完全一致的attention residual。

## 结论

Attention、RoPE、Cache、norm和residual解释全部被反驳。甚至同一block的BF16 QKV flattened GEMM也
exact，因此不能泛化成“hipBLASLt换M一定不同”。剩余边界是连续BF16 FFN：cast→gate/up→SwiGLU
→down。

下一节点只在`bf16_ffn`内部加诊断，并用FP32 FFN/BF16 Attention控制反驳FFN因果关系。

数据见[`107-data`](107-data/)。
