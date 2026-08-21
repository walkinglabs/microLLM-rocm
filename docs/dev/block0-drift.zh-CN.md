# 第一层里面谁先不同：Attention 还是 FFN

逐block实验只知道差异从block 0开始。一个block里面还有很多步骤，所以继续把它拆开：

```text
attention norm
→ Q/K/V projection
→ RoPE
→ Attention context和output
→ residual
→ FFN norm
→ FFN output
→ block output
```

诊断只细分第0层，后面27层仍保留普通block记录。

## 结果

3个fresh B1/B2 pair完全稳定，43个stage的B2重复行全部逐值相同。

| block 0阶段 | max-abs | relative-L2 |
|---|---:|---:|
| attention norm | 0 | 0 |
| Q projection | 0 | 0 |
| K projection | 0 | 0 |
| V projection | 0 | 0 |
| Q/K RoPE | 0 | 0 |
| Attention context | 0 | 0 |
| Attention output | 0 | 0 |
| attention residual | 0 | 0 |
| FFN norm | 0 | 0 |
| **FFN output** | **0.0013504** | **0.00007269** |
| block output | 0.0013504 | 0.00005166 |

这说明“换了batch后所有矩阵乘都会不同”也不对。相同block里的BF16 Q/K/V projection全部exact，
第一处差异只在BF16 FFN输出。

## 现在排除了什么

- embedding、RMSNorm；
- Q/K/V projection；
- RoPE和Attention；
- KV Cache与row copy；
- residual add；
- FFN之前的输入。

下一步只需要打开`bf16_ffn`这个盒子：input cast、gate、up、SwiGLU、down。若gate/up就不同，重点是
hipBLASLt M shape；若它们exact而SwiGLU不同，才检查激活Kernel；若只在down出现，则检查down GEMM。

![Block0 drift](../optimization-log/assets/block0-drift.svg)

完整记录见[Experiment 107](../optimization-log/experiments/107-block0-drift.md)。
