# 新默认之后，GPU时间去了哪里

## 先用一个简单比喻

一次生成像工厂生产64件产品。每件产品都经过28层，所以同一种Attention工作一共执行
`64 × 28 = 1,792`次。

旧路线让一个小组既算“每个历史词有多重要”，又把所有历史信息加起来。这个小组人数太少，
MI300X的大量计算单元没有被充分使用。新路线先把每个历史位置的分数并行算出，再按原来的顺序
完成归一化和加权求和，因此答案不变而速度提高。

## 怎么测才公平

加载18亿参数会花很长时间，但它不是每次generation都重复发生。我们启动两个全新进程：

```text
进程 A = 加载 + 热身 + 1次generation
进程 B = 加载 + 热身 + 3次generation

一次generation = (B - A) / 2
```

两个进程必须报告相同模型、shape、缓存和`auto-enabled`策略。工具同时保存Kernel、HIP API、
copy和allocation原始CSV，不允许只保存一张图片。

## 结果怎么读

总Kernel时间是831.31ms。其中：

- finalize为349.17ms，是最大的单独阶段；
- GEMM为272.79ms；
- 并行score为64.81ms；
- score加finalize仍占总时间49.80%。

这里的“Kernel总时间”是所有GPU Kernel时长相加，不一定等于墙钟时间，因为不同工作可能有
排队或重叠。应用层一次generation为776.14ms，两种口径必须分开保存。

## 为什么下一步不是内存池

程序表面上提出38,755次Tensor申请，但每次都复用了已有块，真正向HIP backend申请的新块是0。
因此“看见很多对象”不能推出“显存分配很慢”。下一实验只改变finalize如何把工作分给线程。

## 能推翻当前解释的实验

如果一种新的线程映射让finalize单算子变快，却让完整模型不变或变慢，那么“finalize是下一项
端到端主因”就被推翻。届时应检查launch、同步、score流量或GEMM，而不是继续堆更多复杂Kernel。

原始数据、派生分析、验证清单和图在
[`benchmarks/results/2026-08-25-post-materialized-deepseek-t2048-profile`](../../benchmarks/results/2026-08-25-post-materialized-deepseek-t2048-profile/)。
