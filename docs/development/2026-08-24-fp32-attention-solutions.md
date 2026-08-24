# FP32 Attention solution inventory

## 实现

- raw hipBLASLt复刻框架row-major、transpose和strided-batch layout；
- Qwen/DeepSeek T512的QK/PV四个exact shape；
- heuristic最多64个solution，去重index并记录workspace；
- default output作为同revision reference；每个candidate完整D2H检查finite/Max/RMS；
- correctness通过后才做HIP Event与wall P50/P95；
- 三fresh进程runner按共同index的median选推荐，不用单次最快值；
- HIP smoke与Python runner contract。

## 结果

四个case各有64个共同passing index。推荐加速1.114×–1.324×，最大Max/RMS
4.47e-7/6.64e-8，workspace均为0。下一节点注册exact key并做完整模型A/B。

完整报告：[Experiment 188](../optimization-log/experiments/188-fp32-attention-solutions.md)。
