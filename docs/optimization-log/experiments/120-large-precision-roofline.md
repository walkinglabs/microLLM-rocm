# Experiment 120：4096³ FP8 达到 477 TFLOPS，但只占峰值 18.25%

## 为什么更换 reference

1024³以前保留独立CPU矩阵乘。到2048/4096，CPU reference会让实验主要测CPU。新模式显式写入
`reference=fp32`：用已经被小尺寸独立参考验证过的hipBLASLt FP32输出比较FP16/BF16/FP8。

这不是独立FP32正确性证据，因此FP32 reference自身误差为0，边界必须留在JSON和文档中。

## 正式结果

| size | FP32 TF | FP16 TF | BF16 TF | FP8 TF | FP8/FP32 | FP8/FP16 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 57.25 | 90.00 | 92.30 | 99.06 | 1.73× | 1.10× |
| 4096 | 110.83 | 336.27 | 351.05 | **477.19** | **4.31×** | **1.42×** |

![Large precision roofline](../assets/large-precision-roofline.svg)

4096³相对官方峰值：

| dtype | achieved | peak utilization | max error vs FP32 |
|---|---:|---:|---:|
| FP32 | 110.83 TF | 67.83% | reference |
| FP16 | 336.27 TF | 25.72% | 0.00043 |
| BF16 | 351.05 TF | 26.85% | 0.00240 |
| FP8 | 477.19 TF | 18.25% | 0.04652 |

## 结论

- FP8在大矩阵上有真实、稳定加速，不再是规格推测；
- 2048只比FP16快10%，4096扩大到42%；
- FP8误差约0.04–0.047，需要模型级scale/logit门；
- FP32路径已用到约68%峰值，FP8只有18%，低精度algorithm/shape仍有很大空间；
- “FP8理论16×FP32”没有在当前dense GEMM实现中出现，实际4096为4.31×。

## 环境与边界

- 10/10 accuracy pass，pre/post GPU门通过；
- Event kernel time，不含量化准备和端到端模型；
- FP8输入E4M3-FNUZ、输出BF16；
- 无结构化稀疏；
- 不执行INT8/INT4；
- 不能把477 TFLOPS直接换算成Qwen tokens/s。

下一步单独实现INT8 executed probe；FP8模型级工作则需要缓存量化权重、动态/块scale和官方模型
完整logits，不应与INT8能力探针混成一个结论。
