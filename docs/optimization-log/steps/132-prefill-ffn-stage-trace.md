# Step 132 — Block-0 prefill FFN stage trace

Status: completed by Experiment 316

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

结果：8个Release进程中，FFN norm跨/内batch位级一致。gate是按执行顺序的第一处差异，up也从相同
exact输入独立漂移；B2/B4/B8 gate Max为9.54e-6/7.63e-6/7.63e-6。同batch重复行也在两个GEMM
内分叉。下一步筛真实FP32 gate/up descriptor。详见
[`Experiment 316`](../experiments/316-prefill-ffn-gate-up.md)。
