# Post-bias training profile

## 初中生版本

一次完整程序会先搬书、再做练习。如果只看整段录像，搬书可能很显眼，但把搬书加速不会让每道
练习题更快。

这里跑两次同一个程序：一次做1步训练，一次做3步。两份Kernel次数和时间相减，再除以多出来
的2步，就能近似看到“每一步训练真正增加了什么”。

结果发现cast-transpose的168次没有随训练步数增加，它只属于权重加载。训练每步真正最大的
部分是hipBLASLt GEMM：18.98 ms，占53.47%。AdamW排第二，但已有完整失败证据，不能因为排名
又高就重复旧方案。

因此下一节点是枚举真实训练GEMM的solution index，并继续使用完整数值→Event→整机三道门。
原始数据见
[`benchmarks/results/2026-08-23-post-bias-training-profile/`](../../benchmarks/results/2026-08-23-post-bias-training-profile/)。
