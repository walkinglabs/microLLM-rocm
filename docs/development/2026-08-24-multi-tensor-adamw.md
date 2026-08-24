# 2026-08-24 — multi-tensor AdamW原语

## 结果

仓库新增一个显式HIP multi-tensor AdamW原语，但没有改变`training::AdamW`或训练CLI。临时
接入在Qwen达到`1.0573×`，在DeepSeek只有`1.0094×`，未通过两模型`1.01×`门。

## 组件边界

- `AdamWMultiTensorWorkspace`只保存shape相关的block map和device descriptor buffer；
- 每步`AdamWMultiTensorEntry`提供当前parameter/gradient/moment/mirror地址；
- pinned host staging保证地址卡在异步H2D结束前不会被CPU改写；
- native-stream async copy与更新Kernel保持同一队列顺序；
- missing gradient让对应Tensor完全不变；
- workspace只接受一个HIP设备和规划时固定的元素数。

## 为什么不进入训练器

减少launch不等于减少显存带宽。DeepSeek AdamW Kernel只快`1.0828×`，最终端到端差一点过门。
默认路径继续使用已经验证的per-Tensor Scalar/Auto策略。未来实现可以复用原语与测试，但必须先
改善descriptor/block调度，重新通过相同模型门。

完整过程见[实验211](../optimization-log/experiments/211-multi-tensor-adamw-discard.md)。
