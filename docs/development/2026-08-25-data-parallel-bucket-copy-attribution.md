# 2026-08-25 — Model-S reducer每步126次backend allocation

![Data parallel bucket copy attribution](../optimization-log/assets/data-parallel-bucket-copy-attribution.svg)

3bucket current path每步创建6 bucket、6 average、114 unpacked Tensor，做228次copy，临时
374,068,224 bytes。allocation/backend均126，cache reuse为0。

这不是普通对象开销：非默认通信stream让pool停用，所有临时Tensor都触发backend申请。
先做in-place average删除6个Tensor并稳住bucket地址，再进入persistent plan。

