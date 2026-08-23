# FP8 FFN-only outer-row policy

`Fp8ActivationScaleMode::FfnOuterRow`只路由FFN gate/up/down三个Linear。Attention Q/K/V/O和
untied output head继续使用固定activation scale，严格对应Experiment 130的范围证据。

准备阶段不为FFN保存无用scalar activation scale。一层untied tiny模型有8个Linear：8份weight
scale加5份非FFN activation scale，共13个FP32值；静态策略为16份。

HIP一层门证明：

- 恰好3次outer-row fallback；
- `outer_row_native_status=0`，没有冒充native执行；
- 0 payload H2D、0 payload D2H；
- Attention/output没有进入outer-row计数；
- 输出全部有限。

完整Release/MI300回归344/344通过，2个条件跳过；sanitizer定向5/5。当前仅为opt-in候选，
官方Qwen/DeepSeek完整logits、吞吐和fallback次数尚未测量，因此不是默认模型策略。
