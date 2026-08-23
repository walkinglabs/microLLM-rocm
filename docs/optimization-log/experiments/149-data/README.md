# Experiment 149 invalid data

这是外部GPU争用导致的invalid实验包，不含任何有效fraction结论。

- fresh build 50步与两个合同有效；
- fraction 1.0预检通过，但运行中post gate发现22% use/9% VRAM并退出1；
- 已写出的3行全部排除；未保存不完整logits payload；
- fraction 0.75首个预检样本即17% use/10% VRAM，runner未启动；
- `verification.json`保存后续三组高占用序列；
- `gates.json`禁止拼接数据、选择fraction或解释TPS。

重试必须使用新目录，从fraction 1.0完整开始。
