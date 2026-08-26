# Step 132 — Block-0 prefill FFN stage trace

Status: trace instrumentation complete; measurement planned

Experiment 315关闭了exact Linear solution组合路线。Experiment 313只记录聚合FFN output，还不能区分
差异来自gate projection、up projection、SwiGLU乘法还是down projection。

下一节点先扩展显式filtered trace，不改变默认路径：

```text
FFN norm → gate projection → up projection → SwiGLU activation
→ down projection → FFN output → block output
```

固定DeepSeek T2048、B1/2/4/8、两个fresh Release进程和exact诊断Attention stack，完整比较前两个相同
输入行。普通trace、未开启detail的推理和训练记录数必须不变。第一处非零阶段才允许进入下一个
solution或可读Kernel反驳；trace本身不计时，也不产生默认优化声明。
