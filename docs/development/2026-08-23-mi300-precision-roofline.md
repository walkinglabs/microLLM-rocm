# 2026-08-23：MI300 executed precision roofline

## 合同

在physical GPU2执行128/256/512/1024 square GEMM。每点warmup5、Event measured20，同时记录
误差、P95、achieved TFLOPS、官方dtype峰值利用率和5.3TB/s roofline利用率。

## 结果

- 20/20 accuracy pass；pre/post设备门通过；
- 1024³：FP32 12.31、FP16 18.63、BF16 12.81、FP8 13.62 TFLOPS；
- FP8相对FP32：128/256/512/1024为0.955/0.919/0.966/1.107×；
- FP8最大误差约0.058；
- 最佳FP8只占官方峰值0.52%，最佳FP32占7.53%。

## 决定

保留runner和raw。拒绝“FP8自动更快”和“模型已经接近MI300峰值”两种解释。下一节点用GPU参考
解除CPU 4096³ reference瓶颈，并把INT8执行能力与float roofline分开。

详见[Experiment 119](../optimization-log/experiments/119-mi300-precision-roofline.md)。
