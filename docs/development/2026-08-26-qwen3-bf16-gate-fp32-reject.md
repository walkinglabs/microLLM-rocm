# Qwen3 gate-FP32简单校准拒绝

日期：2026-08-26
状态：oracle预筛拒绝

新增可复现五case preflight runner。候选4/5匹配FP32，在T512/B1与full-BF16一起选错1096；
oracle为2955。20行raw与summary进入结果目录。

该节点不运行完整shape或性能，避免在已知答案失败后优化速度。它只拒绝gate-FP32；默认与公开
precision limits保持不变。
