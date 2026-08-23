# Block reduction确定性修复

Exact matmul registry的完整回归暴露了一个旧问题：同一shape test偶发token不同。detached旧revision
20次也失败1次，排除了registry改动。完整logits把失败放大为18/20，trace定位到第一个Attention
context，固定Q/K/V直接复现20/20非确定。

根因、修复、20进程门和三进程性能见
[Experiment 156](../optimization-log/experiments/156-block-reduction-determinism.md)。核心规则是：共享
reduction scratch不仅要在写入之间同步，还要保证所有线程已经把结果读进寄存器，才能被下一次
reduction复用。

Shape matrix现在检查uncached prefill、cached prefill与两步decode的完整finite logits、Max/RMS；
严格token gate仍由独立generation/scheduler测试承担。这样既不放宽数值，也不让低margin argmax
偶然相同掩盖内部漂移。
