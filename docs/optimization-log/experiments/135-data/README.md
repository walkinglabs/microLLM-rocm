# Experiment 135 data

Qwen/DeepSeek T512共享dynamic activation量化矩阵。保存18条正式记录、verification、三次GPU预检
和fresh Release Ninja 50-step构建日志。每个worker含1次warmup+3次measured forward。

Qwen tensor calls=384=96×4，Deep=452=113×4，row calls均0。max/RMS与Exp133逐值相同，
所以性能变化没有混入数值变化；精度门仍失败。
