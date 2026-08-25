# 少读六次数据，为什么程序仍然更慢

## 想法没有错，但账没有算全

DeepSeek里6个query heads共享一个value head。旧程序会为每个query head重新读同一份value。新
Kernel只读一次，再分别更新6个答案；每个答案的加法顺序不变，所以结果能逐位相同。

但为了让另一个Kernel看到softmax概率，程序必须先：

1. 把全部概率写到显存；
2. 结束第一个Kernel；
3. 启动第二个Kernel；
4. 再从显存读取概率。

节省的是2-byte BF16 value读取，新增的是4-byte FP32 probability读写和一次Kernel边界。最终目标
case只有旧速度约一半。

## 还抓到一个编译器问题

第一版让循环在运行时决定有几个head accumulator，编译器把数组放到较慢的私有内存，速度只有
约十分之一。把6/7做成编译期常量后，accumulator留在寄存器，速度恢复到约二分之一。这是有效
修复，但不能把整个算法变成赢家。

## 为什么停止这一小条路线

我们已经分别测试了少线程、序列split和GQA复用。能保持模型精度的方案不快，能快的方案改变加法
树并让完整logits失败。继续微调tile很难改变这个结构性矛盾。

下一步转到服务层：batch增大时，每个token会有更多独立head blocks，也许能自然填满GPU，而且
不改变任何数学顺序。仍要同时报告吞吐、单请求延迟、KV和peak，不能只看总tokens/s。
