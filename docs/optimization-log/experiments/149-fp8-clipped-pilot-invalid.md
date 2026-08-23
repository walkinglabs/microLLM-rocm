# Experiment 149：GPU被别人占用，严格门拒绝给答案

计划搜索activation amax fraction `1.0/0.75/0.5/0.25`。fresh build和合同都通过，fraction 1.0
的三次外部预检也都是0%。但Qwen T512 FP8 worker结束后，post gate看见：

```text
GPU use = 22%
VRAM    = 9%
runner exit = 1
```

随后fraction 0.75第一次预检已是17%/10%，后续18个监控样本多数为98%–100% use、57% VRAM。

![Invalid clipped pilot](../assets/fp8-clipped-pilot-invalid.svg)

## 为什么3行也不能用

前3行是在外部任务出现前写出，但一套fraction需要两个模型、两个context和统一执行边界。只保留
部分shape会改变选择规则，也无法证明争用从哪一时刻开始影响时钟或缓存。因此有效suite=0/4、
有效FP8行=0，而不是“先用3行看看趋势”。

## 决定

- 不选择fraction；
- 不报告top token或TPS；
- 不放宽空闲门，也不切换到同样被占用的其他卡；
- invalid数据永久保留，但不得与retry拼接；
- GPU重新独立空闲后，从fraction 1.0完整重跑。

这是失败证据，不是优化失败：基础设施正确阻止了一次受污染结论。
