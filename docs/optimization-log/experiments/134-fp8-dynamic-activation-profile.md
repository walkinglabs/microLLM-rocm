# Experiment 134：量化三段已经接近GEMM时间

T512单步profile使用host weight-amax，避免weight准备混入dynamic-scale Kernel。whole-process仍含
模块加载、权重cast/quantize和销毁，因此只对名字明确的Kernel做前向归因。

| 模型 | scale | finalize | device quantize | 三段合计 | Tensile GEMM | 可归因合计 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 0.675ms | 0.685ms | 0.763ms | **2.122ms** | 3.119ms | 5.241ms |
| DeepSeek | 0.875ms | 0.810ms | 1.424ms | **3.109ms** | 5.519ms | 8.628ms |

![Dynamic activation profile](../assets/fp8-dynamic-activation-profile.svg)

Qwen168次、Deep197次dynamic调用，正好等于每个Linear一次。三段占dynamic+GEMM的40.5%/36.0%。
但Q/K/V共享attention norm输入，gate/up共享FFN norm输入；当前分别量化3次/2次。共享后理论调用
数为Qwen96、Deep113（每层QKV一次、O一次、gate/up一次、down一次，加Deep/output边界）。

whole-process最大热点cast-transpose是权重加载：Qwen4.88ms、Deep21.49ms，不属于measured
prefill。profile下TPS受工具影响，不与Exp133 Release TPS比较。

下一改动只共享量化Tensor，不改变scale算法、权重、GEMM或输出顺序；完整logits应逐值相同。
