# Experiment 278：weighted overlap为什么省了等待，整步却更慢

Status: discard for performance; correctness primitive retained

## 假设

在Model-S T128中，backward时间足够长。把ready bucket通信塞进backward，应该像等长batch一样
让整步至少加速1.01x。

## 唯一改动

数据、模型、bucket和optimizer都不变。只把`[B1,B2]`的local gradient scale从backward结束后
统一执行，改成每个leaf ready时先scale、再记录Event。

## 固定实验

- 两张AMD Instinct MI300X VF，gfx942；
- Model-S，57个参数Tensor、15,586,176个值；
- rank rows `[1,2]`，T128，即128/256 token，average 192；
- rank scale `0.666666687 / 1.333333373`；
- 25 MiB bucket，每步3个bucket；
- 同步`bucket-views`与`overlap-views`交替3轮；
- 每轮3步，丢弃第1步，保留每策略6个steady sample；
- CPU参数门Max `1e-2`、RMS `1e-5`；三步loss门显式为`1e-3`；
- 每轮额外逐项比较两策略最终57个Tensor。

## 结果

![Ranked weighted overlap discard](../assets/ranked-weighted-overlap-discard.svg)

同步finish为2.664ms，overlap为1.381ms，等待缩短1.930x。但ready hook中的57次scale让
forward/backward从6.169ms增加到7.689ms（+1.520ms）。最终steady step从8.954ms变成
9.332ms，速度比只有0.9594x，低于1.01准入门。

逐轮速度比为0.9575x、0.9354x、1.0274x；最后一轮单看会误判为成功。任意删掉一整对运行后，
综合速度仍只有0.9519x–0.9735x，因此拒绝结论不依赖一个异常慢样本。

正确性全部通过：rank Max/RMS 0/0；三轮同步与overlap的15,586,176个最终参数逐项Max/RMS
均为0/0；相对CPU的Max/RMS为0.004938/3.218e-6；加权loss最大差1.72e-5。两条路径current/
peak都为249,378,820 / 417,369,612 bytes，增量0。临时参数文件全部删除，peer failure仍有界
终止。

原始证据：[`ranked weighted overlap`](../../../benchmarks/results/2026-08-25-ranked-weighted-overlap/)

## 决定

拒绝“每个leaf单独scale”的性能路由，但保留它作为显式、已验证的正确性原语。它不能成为
uneven batch默认快路径。

## 推翻当前解释的下一实验

当前证据更支持“57次scale Kernel启动成本抵消通信隐藏”，而不是“RCCL没有重叠”：finish已经
缩短1.930x。下一实验只把权重移动到ready bucket：leaf只标记ready，通信Stream打包后对3个
bucket各scale一次，再all-reduce。如果scale调用57→3而参数仍精确一致、整步过1.01门，就支持
这个解释；如果仍不过门，则weighted overlap track停止。
