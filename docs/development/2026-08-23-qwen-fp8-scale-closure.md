# Qwen FP8 global-scale closure

Qwen-only 1.6/3.2矩阵9/9执行、0/8过门、8/8 top相同。最佳RMS 0.2167，仍为门的
4.33倍且没有字面反弹。结合DeepSeek在0.2附近已经转弯，工程上停止跨模型全局scale枚举；文档
明确没有把有限搜索写成数学证明。

下一实现只改变weight尺度：每个Linear按自己的amax选择E4M3-FNUZ scale，activation先固定。

详见[Experiment 126](../optimization-log/experiments/126-qwen-fp8-scale-closure.md)。
