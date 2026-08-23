# Experiment 150：程序跑完了，但改了两个变量，所以答案作废

新runner把worker从48降到20，并在干净GPU上全部执行成功。但审查fraction=1时发现，它与
Exp148 retained O-only不一致：

```text
retained weight minimum = 0.005
pilot hardcoded minimum = 0.0001
```

fraction搜索本应只改变activation clipping，现在同时改变了权重量化下限。四个fraction=1
Max/RMS都与Exp148不同，因此不能判断任何clipping fraction。

![Fraction pilot workload mismatch](../assets/fp8-fraction-pilot-workload-invalid.svg)

## 为什么不接受“反正1.0最好”

runner确实报告1.0优于0.75/0.5/0.25，但那只是0.0001权重策略内部的结果。目标是优化已保留的
0.005 O-only路径。换了起点，结论就不能迁移；即使方向看起来合理，也必须作废。

## 修复

runner新增显式`--fp8-weight-scale`，默认0.005；合同测试直接检查生成命令中的值。retry使用
新实验编号和目录，不能合并这20个worker。Exp149是GPU争用invalid，Exp150是workload
mismatch invalid，两者分别保留。
