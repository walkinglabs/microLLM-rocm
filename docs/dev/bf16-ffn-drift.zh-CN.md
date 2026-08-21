# BF16 FFN里面第一处差异在哪里

`bf16_ffn`可以拆成五步：

```text
FP32 input → BF16 cast → gate/up GEMM → SwiGLU → down GEMM
```

第一次运行诊断时，TraceSession把BF16 tensor写成空values，却错误标成“没有截断”。runner因此
立即失败，没有产生结论。修复后，所有浮点dtype都通过`to_vector()`导出，低精度统计和截断状态
也有单元测试。

## 三次正式结果

| FFN阶段 | max-abs | mean-abs | RMS | relative-L2 |
|---|---:|---:|---:|---:|
| input BF16 | 0 | 0 | 0 | 0 |
| **gate** | **0.015625** | 1.39e-7 | 3.27e-5 | 6.11e-5 |
| up | 0.001953 | 4.35e-8 | 7.55e-6 | 1.94e-5 |
| SwiGLU | 0.007812 | 4.28e-8 | 1.48e-5 | 1.10e-4 |
| down/output | 0.001350 | 1.02e-5 | 5.03e-5 | 7.27e-5 |

cast逐值相同。gate是第一个非零stage；up使用同一个input，是独立出现的小差异，不是被gate污染。
两个最大差值0.015625和0.001953125具有明显的BF16步长特征。

B2重复两行在全部48个stage仍完全相同，所以不是row错误。最强解释是hipBLASLt在M=32和M=64
选择了不同plan或累加路径。

下一步记录gate/up的exact shape与algorithm ID，再尝试让M32/M64使用同一可用algorithm。只有同
algorithm确实缩小误差且不明显拖慢端到端，才考虑改变默认。

![BF16 FFN drift](../optimization-log/assets/bf16-ffn-drift.svg)

完整记录见[Experiment 108](../optimization-log/experiments/108-bf16-ffn-drift.md)。
