# FP8 scale boundary expansion

沿Experiment 123相同runner和完整logits门，只把activation scale从0.05上方扩到0.1/0.2。
MI300X正式18/18 worker执行成功、0/16候选过门。Qwen和DeepSeek的最好RMS分别降到0.669
和1.170，但两个最佳点再次落在0.2上边界。

本轮推翻“0.05附近已经到误差谷底”，没有推翻全部全局scale。下一节点只继续0.4/0.8，找到
转折后停止数字搜索。

详见[Experiment 124](../optimization-log/experiments/124-fp8-scale-boundary.md)。
