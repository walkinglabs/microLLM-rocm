# 2026-08-24 — 训练图里的残差加法与RMSNorm

## 一句话结果

我们补齐了一个能正确求导的`autograd::add_rms_norm`，但没有把它接进Transformer训练默认
路径。原因很简单：它数学正确，也真的少启动了72个Kernel，可真实模型训练没有变快。

## 为什么这个算子返回两个结果

设：

```text
sum = left + right
normalized = rms_norm(sum, weight)
```

后面的代码可能同时使用`sum`和`normalized`。所以API返回一对`Value`，而不是只返回第二个。
可以把它想成一条岔路：一份水直接向前流，另一份经过过滤器后再向前流。

## 图为什么仍有两个节点

GPU可以在一次Kernel里算出两个结果，但反向传播仍要知道两条路在哪里汇合：

```text
left ─┐
      ├─ sum node ─┬─ direct residual use
right ┘            └─ RMSNorm node ─ use normalized
                               └─ weight
```

反向时先从两条使用路径收集`sum`的梯度。等它们相加完成后，sum节点把同一个总梯度送给
left和right。RMSNorm节点同时计算对sum和weight的梯度。这就是为什么“少一个GPU启动”不等于
“少一个数学节点”。

## 验收门

- 手写小Tensor：两个前向结果和组合参考一致；
- 分叉图：left、right、weight三组梯度与组合图一致；
- PyTorch：使用`torch.nn.functional.rms_norm`作为独立oracle；
- HIP：前向和反向全在设备上，测量区间H2D/D2H均为0；
- 覆盖审计：多返回值API也必须出现在图算子清单；
- 真实模型：Qwen2.5-0.5B与DeepSeek Distill 1.5B，B1/T512，交替顺序三进程。

## 为什么没有接进模型

Qwen吞吐比为`0.9785×`，DeepSeek为`0.9980×`，峰值显存都不变。DeepSeek的固定观察参数还
出现末位差异。Profile显示总Kernel调用6,903→6,831，但总Kernel时间只下降0.045%。

因此仓库只保留通用、可测试的Autograd能力，删除临时模型和CLI路由。这样以后图优化器需要
多输出融合时可以复用正确基础，同时用户不会误以为当前训练默认路径已经加速。

完整数据见[实验210](../optimization-log/experiments/210-training-add-rms-norm-fusion-discard.md)。
