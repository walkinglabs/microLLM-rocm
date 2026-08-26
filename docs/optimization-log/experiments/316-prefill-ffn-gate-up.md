# Experiment 316 — FFN norm没错，两个投影先分叉

## 结论

FFN norm在B1/2/4/8跨batch和同batch都位级一致。gate是按代码执行顺序的第一处差异：B2 Max
`9.54e-6`，B4/B8 `7.63e-6`。up也读取同一个exact输入，并独立出现差异。因此SwiGLU不是首因；
下一步同时筛gate/up共享的真实FP32 GEMM descriptor。

## 用简单话解释

FFN像两条并行支路。相同输入分别经过gate和up，之后才在SwiGLU处相乘。现在输入完全一样，但两条
支路各自算完矩阵乘法后都出现最后几位差异，所以不能怪后面的乘法。我们应该先检查这两个矩阵乘法
能否用一种对不同batch都保持相同行结果的算法。

## 证据规模

- DeepSeek T2048，B1/2/4/8；
- 两个fresh Release进程，共8个；
- 7个完整边界，前两个batch行；
- B1临时270,532,608 bytes，B≥2临时541,065,216 bytes；
- 两轮误差统计逐项相同；
- 临时二进制最终保留0个。

| Batch | Gate跨B1 Max | Gate同batch Max | Up跨B1 Max | 第一阶段 |
|---:|---:|---:|---:|---|
| 2 | 9.54e-6 | 8.34e-6 | 8.11e-6 | gate |
| 4 | 7.63e-6 | 7.39e-6 | 7.63e-6 | gate |
| 8 | 7.63e-6 | 7.39e-6 | 7.63e-6 | gate |

原始小型统计与图见
[`benchmarks/results/2026-08-26-prefill-ffn-stage-trace`](../../../benchmarks/results/2026-08-26-prefill-ffn-stage-trace/README.md)。

Trace包含同步、D2H和文件写入，不提供性能结论。
