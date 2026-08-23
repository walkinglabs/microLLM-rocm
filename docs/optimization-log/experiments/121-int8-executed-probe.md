# Experiment 121：INT8 真实跑到416 TOPS，但模型INT8仍未开始

## 旧证据缺什么

`gfx942`能力表写着原生INT8 Matrix，但“硬件支持”不等于仓库提交过一个INT8 Kernel。本轮直接
调用hipBLASLt：

```text
INT8 A × INT8 B → INT32 accumulator/output
```

每个shape从GPU取回完整输出，固定5个位置由CPU整数点积抽样，误差必须精确为0。

## 正式结果

| size | median / P95 ms | TOPS | peak util | roofline util | sample error |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.0104 / 0.0115 | 0.40 | 0.015% | 0.178% | 0 |
| 256 | 0.0132 / 0.0142 | 2.55 | 0.097% | 0.563% | 0 |
| 512 | 0.0159 / 0.0165 | 16.91 | 0.647% | 1.869% | 0 |
| 1024 | 0.0321 / 0.0334 | 66.87 | 2.557% | 3.696% | 0 |
| 2048 | 0.0574 / 0.0627 | 299.14 | 11.440% | 11.440% | 0 |
| 4096 | 0.3304 / 0.3526 | **416.03** | **15.910%** | **15.910%** | 0 |

![MI300 INT8 executed probe](../assets/mi300-int8-probe.svg)

2048以后算术强度使roofline进入compute-bound，所以peak utilization与roofline utilization相同。

## 这次能说什么

> MI300X上的raw hipBLASLt INT8 GEMM已实际执行，4096³达到416 TOPS和官方峰值15.9%。

不能说：

- microLLM Tensor 已支持 INT8；
- 已有权重量化器或activation量化器；
- scale、zero-point、per-channel/group policy已实现；
- Transformer Linear、加载、训练或生成支持INT8；
- 416 TOPS等于模型tokens/s。

## 环境和正确性

- 6/6 shape，5 samples/shape exact；
- pre VRAM/use最大0%/1%，post最大0%/4%；
- HIP smoke和Python合同通过；
- int32输出在当前小值输入和K≤4096下没有溢出。

## 下一步

是否加入公共INT8 Tensor必须由真实模型目标驱动。最小有用路线是weight-only INT8：保留FP16/BF16
activation、定义per-channel scale、在GEMM前或epilogue反量化，并先比较权重流量与额外Kernel。
完整INT8 activation需要另一套calibration与数值门，不能一次加入。
