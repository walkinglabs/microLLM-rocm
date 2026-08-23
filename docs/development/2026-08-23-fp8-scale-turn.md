# FP8 global-scale turn search

activation 0.4/0.8正式矩阵18/18执行、0/16过门。DeepSeek保留top token的最佳RMS从0.2的
1.170回升到1.235，曲线转折已出现；Qwen则继续降到0.303，最佳点仍在0.8边界。

停止DeepSeek全局scale搜索；Qwen只再扩一次1.6/3.2。完整logits与top-token门继续优先于RMS
和速度。

详见[Experiment 125](../optimization-log/experiments/125-fp8-scale-turn.md)。
