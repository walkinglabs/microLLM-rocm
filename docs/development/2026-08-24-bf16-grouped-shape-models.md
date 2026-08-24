# 矩阵大小一样，完整模型也可能不一样

日期：2026-08-24

rows=1024可以来自：

    batch=1，文本长1024
    batch=2，每条文本长512

投影矩阵大小相同，但Attention看到的batch和序列不同。因此我们分别跑完整模型。

六个case全部变快：Qwen快1.028–1.107倍，DeepSeek快1.021–1.076倍。36个进程的数值、
每个batch行的top-1、显存和setup都通过。

第一次正式运行还发现CLI文件bug：B2 Tensor内部有两行，写文件时却只写第一行。现在：

    last模式：写全部batch行
    full模式：每个batch各取自己的最后token再写

专门测试会生成一个真实tiny Qwen权重，让B1、B2 last、B2 full都经过真实二进制。B2文件
必须正好是两行，不能再靠模型内部测试代替CLI边界。

结论：rows相同只能说明投影shape相同，不能省略batch和sequence的完整模型测试。
