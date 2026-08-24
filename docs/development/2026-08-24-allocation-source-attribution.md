# Allocation source diagnostics

## 实现

- 固定`AllocationSource`枚举，避免诊断tag自己分配字符串；
- `ScopedAllocationSource`支持嵌套恢复，关闭时一次分支后no-op；
- thread-local按source/device/exact bytes聚合calls和total bytes；
- runtime allocation在backend新分配和cache reuse前都记录逻辑请求；
- model为embedding、Attention norm/projection/layout/core/output、residual、FFN、head标tag；
- CLI只允许zero-warmup single-prefill诊断并输出机器可读records；
- 两模型×三进程runner要求分布逐项确定。

第一次编译遗漏`model.cpp`对diagnostics头的include，编译门立即失败；补头后CPU/HIP targeted
tests通过。该失败保留在开发记录中，不写成“一次成功”。

## 结果

Qwen/DeepSeek T512三进程分布完全一致。共同最大来源为`attention.core`：572.5/792.7MB，
占全部逻辑分配53.0%/43.6%。下一节点进入Attention core，不再猜projection Tensor。

完整报告：[Experiment 186](../optimization-log/experiments/186-allocation-source-attribution.md)。
