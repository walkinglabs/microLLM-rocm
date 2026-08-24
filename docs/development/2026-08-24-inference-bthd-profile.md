# copy归零以后，下一块时间是什么

日期：2026-08-24

BTHD以后，strided copy在profile中完全消失，总Kernel快1.169倍和1.118倍。

新的非GEMM热点是cast和softmax。QKV grouped先产生BF16，旧路径立刻把Q/K/V转FP32；Q/K随后又进入
bias+RoPE融合。下一次最小实验是让这个融合直接读BF16 Q/K，少两次cast。V暂时不动。
