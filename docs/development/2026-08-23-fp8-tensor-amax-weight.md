# FP8 per-Tensor weight amax policy

## 初中生版本

旧策略让所有Linear共用同一把尺子。有的层数字小、有的层数字大：尺子太短会截断大数，尺子
刻度太粗又会损失小数。新`tensor-amax`策略在模型准备阶段逐个查看Linear权重，让每个权重
Tensor拥有自己的尺子：

```text
weight scale = max(abs(weight)) / 240
```

`240`是MI300 E4M3-FNUZ在不乘scale前能表达的最大有限幅值。全零权重使用配置中的固定scale
作为fallback；NaN/Inf会让整个准备过程失败，而且任何权重都不会被半途替换。

## API

```cpp
config.linear_precision = LinearPrecision::Float8E4M3FNUZ;
config.fp8_weight_scale_mode = Fp8WeightScaleMode::TensorAmax;
config.fp8_activation_scale = 0.2F;  // 本节点仍是固定值
auto report = model.prepare_fp8_inference_weights();
```

CLI：

```text
--fp8-linear true --fp8-weight-scale-mode tensor-amax
```

report/JSON会给出扫描字节数、实际最小/最大weight scale。这个策略是inference-only；训练的
FP32-master straight-through路径仍使用显式固定scale。

## 生命周期和性能边界

模型加载到GPU后，第一版会为每个Linear做一次权重D2H扫描以取得amax。它简单、容易检查，但
会增加启动时间。准备完成后只保留FP8权重与同设备scale，forward热路径不得再扫描。

新增HIP测试最初确实发现prepared forward仍每步发生8次D2H：原因是分支判断前无条件计算
lazy amax。修复后同一测试证明准备期发生一次性扫描，而热路径H2D/D2H均为0。

## 当前证据

- 完整Release/MI300回归：331/331通过，2个条件跳过；
- CPU：不同Tensor得到不同scale；
- CPU：非有限权重事务式拒绝，模型仍保持FP32；
- HIP：lazy与prepared输出一致；准备后热路径0 payload transfer；
- official Qwen/DeepSeek完整logits：36/36执行，四个RMS相对最初静态点下降39%–78%，
  但0/4通过；因此只保留opt-in基础设施，不接受当前模型策略。完整证据见
  [Experiment 127](../optimization-log/experiments/127-fp8-tensor-amax-weight.md)。
