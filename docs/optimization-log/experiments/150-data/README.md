# Experiment 150 invalid data

这套20-worker pilot执行成功，但workload与Exp148 retained O-only不一致，因此数值选择无效。

- runner内部weight minimum为0.0001；retained策略为0.005；
- fraction=1四组Max/RMS均不匹配Exp148；
- 20 worker/16 comparison的shape、有限值和计数合同仍保留；
- 原runner selection只作为作废副本，不进入结论；
- Exp149的污染数据没有被读取或合并；
- retry必须使用修正runner、新实验号、新目录和新GPU预检。
