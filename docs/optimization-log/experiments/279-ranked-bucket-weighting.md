# Experiment 279：把57次scale合成3次，能拿回被吃掉的收益吗

Status: keep as an explicit measured T128 route; not a general default

## 假设

Experiment 278的finish已经快1.930x，整步却只有0.9594x。主要损失来自每个leaf一次scale，而
不是RCCL没有重叠。把57次leaf scale改成3次bucket scale，应让Model-S T128整步过1.01门。

## 唯一改动

同步对照保持`bucket-views`：backward结束后57个leaf分别scale。候选
`bucket-weighted-overlap`让leaf只报告ready；通信Stream等待Event、pack完整bucket、对bucket
scale一次，再all-reduce average。模型、数据、3个bucket和optimizer都不变。

## 固定实验

- 两张AMD Instinct MI300X VF，gfx942；
- Model-S，T128，rank rows `[1,2]`，128/256 token；
- local scale `0.666666687 / 1.333333373`；
- 25 MiB上限，每步3个bucket；
- 两策略交替3轮，每轮3步，丢弃第1步；
- 每策略6个steady sample；
- CPU参数Max/RMS门`1e-2 / 1e-5`，三步loss门显式`1e-3`；
- 每轮逐项比较策略最终57个Tensor、15,586,176个值。

## 结果

![Ranked bucket weighting](../assets/ranked-bucket-weighting.svg)

scale调用按合同从57个leaf变为0个leaf+3个bucket。同步finish为2.771ms，候选为1.361ms，
快2.035x；forward/backward只增加0.641ms。steady step由9.262ms降到8.687ms，达到
1.0661x，超过1.01准入门。

三次策略间完整参数比较Max/RMS均0/0；rank Max/RMS 0/0；CPU Max/RMS为
0.004938/3.218e-6；加权loss最大差1.72e-5。current/peak都是249,378,820 /
417,369,612 bytes，增量0；later backend allocation为0；临时权重全部删除；peer failure有界。

原始证据：[`ranked bucket weighting`](../../../benchmarks/results/2026-08-25-ranked-bucket-weighting/)

## 不把一次keep写成普遍结论

逐轮速度比是0.9510x、1.0437x、1.1131x：第一轮候选更慢。leave-one-pair-out为
1.0825x、1.0197x、1.0027x，删掉第三轮时会落到1.01门以下；候选CV也有7.20%。因此保留的是
“当前两张MI300X、Model-S T128、25 MiB bucket上的显式路由”，不是默认打开或跨shape结论。

## 下一实验

候选仍有每步57次device-to-device pack copy和3次bucket scale。下一最小实验使用持久化的
pointer/offset描述，把每个bucket的多段gather和local scale融合成一次Kernel，目标是57 copy +
3 scale变成3次gather-scale launch。若完整数值门通过但整步或敏感性不改善，就拒绝融合并停止
这条T128 reducer局部优化线。
