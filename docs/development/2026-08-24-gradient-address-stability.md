# 2026-08-24 — 为什么 shape 一样，gradient 地址仍可能不同

## 给初学者的比喻

Tensor的shape像“箱子尺寸”，Storage地址像“仓库货架编号”。两只箱子尺寸相同，不代表第二次
一定放回同一个货架。

HIP Graph录下的是货架编号，不是“帮我找一只同尺寸箱子”。如果下一次backward把gradient放到
另一格货架，旧Graph会读到过期内存。

## exact-size pool能保证什么

框架的default-Stream pool会按精确字节大小保存退役块。同尺寸申请可以复用，但一个尺寸可能有
很多块，而且临时Tensor与参数gradient会共同竞争。池采用后进先出；context变化会改变中间Tensor
的创建和销毁顺序，所以最终拿到哪一块也可能改变。

“allocator会复用”不等于“每个参数一定拿回自己的地址”。

## benchmark怎么做

新benchmark建立finite synthetic model，不需要下载权重：

1. 使用真实Qwen/DeepSeek配置创建完整参数shape；
2. norm填1、bias填0、其他参数填0.01，确保loss有限；
3. 做一次warmup backward；
4. 打开训练使用的exact-size pool；
5. 连续做两次measured backward；
6. 对每个命名参数比较gradient Storage身份、字节和所有者数；
7. 输出`address_stable`和变化次数，但不输出真实指针数值。

每次观察的Storage use count都是2：一份由参数节点拥有，一份是诊断时的短暂Storage句柄。这也
证明结果没有被额外长期引用偷偷稳定下来。

## 为什么要测两个context

Qwen在T8和T512都完全稳定。DeepSeek T8也稳定，但T512的198个大gradient全部换地址，只有
norm/bias等141项稳定。模型相同，context改变就让结论翻转。

因此正确的Graph缓存键不能只写：

```text
model + dtype
```

它至少还要包含shape/context，并验证当前parameter、gradient、moment、mirror地址与准备时快照
一致。验证失败必须recapture或fallback，不能继续launch旧Graph。

## 对下一版的影响

- Qwen T512可以直接进入optimizer Graph性能实验；
- DeepSeek T8可以进入短context实验；
- DeepSeek T512当前必须拒绝；
- stable gradient buffer若要实现，必须覆盖几乎全部7.108GB payload，不能只修四个K/V Tensor；
- 地址稳定只证明“可以安全尝试”，不证明端到端会更快。

## 发布验证

CPU 333/333、ASan/UBSan 331/331、PyTorch-enabled CPU 307/307、完整CPU+HIP
526/526（3个条件跳过）、HIP标签181/181、RCCL 14/14、multi-GPU 12/12。覆盖清单注册
96个测试文件；本节点只增加benchmark/runner，没有改变`src/`或`include/`覆盖分母，当前覆盖率
保持78.5% lines、86.8% functions、59.2% branches。
