# 2026-08-25 — 一块积木快，不等于整辆车快

online Attention单独测试比旧算子快。但模型每一层先得到FP32的Q、K、V；新算子只收BF16，所以
每层还要做三次转换。Qwen有24层，DeepSeek有28层。局部省下的分数表读写，要和这些转换以及
模型里其他矩阵乘一起计算。

正式实验让两条模型路径使用相同权重、token、Arena和其他优化，只切换online开关。程序不只看
最后选中的token，还保存词表里151936个完整分数。每个进程也数清到底有多少层走了新kernel。

结果中，六组top token都没变，显存少3.5到57MiB；但速度只剩原来的76%到88%。Qwen长上下文
完整logits最大差0.511，RMS差0.112，也超过事先写下的门。

![Full-model online Attention discard](../optimization-log/assets/rocwmma-online-model-discard.svg)

所以框架保留这个公共算子供研究和独立调用，却不让模型默认使用。未来如果RoPE可以直接输出
BF16，省掉三次转换，可以重新做实验；不能拿今天的算子跑分替代明天的整模证据。

## 发布回归

CPU 341/341、ASan/UBSan 339/339、PyTorch-enabled CPU 315/315、完整CPU/HIP 537/537
（3个条件跳过）、HIP标签184/184、RCCL标签14/14、multi-GPU 12/12，覆盖清单注册103个
测试文件。实验开关默认false；完整HIP
重跑证明原有模型、训练和低精度路径没有改变。
