# Experiment 119：FP8 到 1024³ 才快 10.7%，峰值利用率仍只有 0.52%

## 为什么不能只引用 2614.9 TFLOPS

MI300X 官方 dense peak 是 FP32 163.4、FP16/BF16 1307.4、FP8 2614.9 TFLOPS，HBM3
带宽 5.3 TB/s。但芯片数字不是程序速度。我们需要同时保存：

```text
achieved TFLOPS = 2MNK / Event time
arithmetic intensity = 2MNK / estimated input+output bytes
roofline bound = min(dtype peak, 5.3 TB/s × intensity)
```

本轮使用真实 hipBLASLt FP32/FP16/BF16/FP8 Kernel，FP8为MI300 E4M3-FNUZ、BF16输出。

## 正式结果

| size | FP32 TF | FP16 TF | BF16 TF | FP8 TF | FP8/FP32 | FP8 max error |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.184 | 0.186 | 0.171 | 0.176 | 0.955× | 0.0583 |
| 256 | 1.353 | 1.413 | 1.400 | 1.244 | 0.919× | 0.0452 |
| 512 | 9.704 | 9.160 | 9.172 | 9.378 | 0.966× | 0.0555 |
| 1024 | 12.307 | **18.634** | 12.813 | 13.625 | **1.107×** | 0.0541 |

![MI300 precision roofline](../assets/mi300-precision-roofline.svg)

1024³的官方峰值利用率：

| dtype | achieved | official peak | utilization |
|---|---:|---:|---:|
| FP32 | 12.31 TF | 163.4 TF | 7.53% |
| FP16 | 18.63 TF | 1307.4 TF | 1.43% |
| BF16 | 12.81 TF | 1307.4 TF | 0.98% |
| FP8 | 13.62 TF | 2614.9 TF | 0.52% |

## 结果解释

- 128–512 中，FP8慢3%–8%，低精度并不会自动抵消launch、descriptor、scale和输出开销；
- 1024³ 才出现FP8对FP32的10.7%收益；
- 当前最快是FP16，不是理论峰值最高的FP8；
- FP8误差约0.045–0.058，远高于FP16/BF16；
- 所有路径离官方峰值很远，这些尺寸仍不足以证明MI300硬件饱和。

这也解释了为什么microLLM某些decode比Torch快：主要收益来自更短的软件路径和更少同步，不是
我们的GEMM已经接近2.6 PFLOPS。

## 正确性和环境

- 4尺寸×5路径=20/20 accuracy pass；
- warmup 5、Event measured 20；
- pre VRAM/use最大0%/1%，post最大0%/4%；
- FP32 max error `2.86e-6`，FP16 `6.90e-4`，BF16 `6.54e-3`，FP8 `5.83e-2`。

## 仍未证明什么

- 本轮没有执行INT8或INT4；硬件能力表不是executed-kernel证据；
- 没有结构化稀疏，不能引用翻倍峰值；
- 只到1024³，不能作为大GEMM最大吞吐；
- square GEMM不代表M=1 decode或Transformer完整图；
- FP8包含显式scale且输出BF16，不是所有FP8配方。

下一节点应避免CPU reference拖住4096³：先用已经被小尺寸独立参考验证的FP32 hipBLASLt作为
large-shape数值基线，运行2048/4096；同时建立独立INT8 executed probe。两者必须分开报告。
