# 2026-08-25 — Attention怎样边算边忘

## 为什么整张表很贵

长度2048的句子，两两比较会得到2048×2048张表。一个head就有四百多万个数；Qwen式14个
head用FP32保存约224MiB。表写进显存后，softmax和V加权又要读回来。

online办法像分批统计考试成绩：每次只看32列，记住当前最大值、指数和以及已经加权的答案。
下一批出现更大值时，把旧统计按比例缩小，再加入新批。最后只留下输出，不留下整张分数表。

## GPU里的分工

- 一个wave用rocWMMA算32×32个QK分数；
- 32个线程分别维护32行的最大值与和；
- 权重转成BF16小块；
- D64用两个wave、D128用四个wave做PV矩阵乘；
- query head各自并行，多个query head按GQA规则共享较少的K/V head。

最早版本让一个wave用普通循环做PV，T512慢了约21倍。增加第二个wave时又发生shared写重叠，
完整输出测试立即报0.029误差。修好步长后，普通PV仍慢；换成矩阵PV并加入真实head并行，才超过
当前框架。

## 现在证明了什么

Qwen式和DeepSeek式共14个shape、42个fresh processes全部通过。相对当前框架快1.260–4.041倍，
并且不写全局score。候选误差最大约5.66e-4，不是bit-exact；短上下文还有标量kernel更快的
反例。

![rocWMMA online Attention](../optimization-log/assets/rocwmma-online-attention.svg)

所以它现在只是“值得接成operator”，不是“已经能给模型默认使用”。下一步还要补batch、非32
倍数、硬件fallback、公共测试和完整模型logits。

## 发布回归

CPU 338/338、ASan/UBSan 336/336、PyTorch-enabled CPU 312/312、完整CPU/HIP 533/533
（3个条件跳过）、HIP标签183/183、RCCL标签14/14、multi-GPU 12/12，覆盖清单注册101个
测试文件。CPU覆盖率仍为78.4% lines、86.6% functions、59.1% branches；设备prototype由
gfx942 smoke和42进程完整矩阵负责，不能用CPU覆盖率代替。
