# 2026-08-26 — FP32 FFN down solution runner

down与gate/up使用相同的通用C++ row-invariance工具，但真实维度交换为K8960、N1536。wrapper只修改
descriptor常量、record type和SVG标题；inventory、CPU sentinel、完整重复block、同进程default Event
和每M 0.95门完全复用。

这个runner不包含模型scope，也不依赖已删除的gate/up CLI。operator没有通过前，不会重新增加用户路径。
