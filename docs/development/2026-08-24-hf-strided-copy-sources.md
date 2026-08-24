# 剩下的copy到底是谁产生的

日期：2026-08-24

profiler说每步还有96/112次strided copy，但只有shape，没有“谁调用”。现在记录会带模型已有的
source标签。

结果非常整齐：

    Q、K、V：每层各转一次布局
    context：每层转回来一次

Qwen一共100.7MB，DeepSeek一共205.5MB，全部属于Attention。FFN、embedding和output都是0。

因此下一步不是把copy Kernel写快一点，而是让Attention从开始到结束都理解BTHD布局，把四次copy
直接删掉。

诊断默认关闭。只有明确打开allocation或strided诊断时，source scope才记录状态；两者都关闭时
仍是快速空操作。
