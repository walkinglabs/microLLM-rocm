# 2026-08-26 — FFN完整值矩阵runner

runner固定DeepSeek T2048、B1/2/4/8和两个fresh process，使用exact诊断Attention stack后比较Block-0
FFN的norm、gate、up、SwiGLU activation、down和两个输出边界。当前official route的FFN权重为FP32，
所以合同不会虚构一个没有执行的BF16 input cast；若以后切换低精度权重，需要另建dtype track。

大Tensor不写进JSON。TraceSession把选中的逻辑值转换为little-endian FP32临时二进制，JSON只留一个
样例和元数据。runner校验shape、dtype、元素数、文件大小、route、solution cache和dispatch，再逐块
计算bitwise、Max、RMS和relative-L2。每个batch比较完成后删除临时文件，结果目录只保留小型统计和SVG。

这个runner是显微镜，不是benchmark：导出过程包含同步、转换和文件写入，任何wall时间都不能用来声称
模型更快或更慢。
