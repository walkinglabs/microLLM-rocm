# FP8 device-side activation Tensor amax

## 初中生版本

以前每个Linear都拿CPU给的一把固定尺子量activation。新策略先让GPU自己看这一块输入里最大的
绝对值，再在GPU上算尺子长度：

```text
scale = max(max(abs(input)) / 240, minimum_scale)
```

接着量化Kernel直接读取GPU上的scale。整个过程中CPU不知道这个具体小数，也不需要知道。

## 为什么要改ScaledTensor合同

旧`ScaledTensor`同时保存GPU scale Tensor和host float，并默认两者都存在。动态scale若把float
取回host，会让每个Linear发生D2H和同步。现在新增`host_scale_available`：

- 固定scale：GPU Tensor与host float都可用；
- device amax：GPU Tensor是真值，host只保留minimum配置但标记为不可用；
- hipBLASLt直接读取GPU scale指针；
- unsupported shape的FP8→BF16 fallback也用GPU scale反量化，不偷偷D2H；
- CPU reference始终有host scale。

## API

```cpp
auto scaled = ops::quantize_fp8_dynamic(
    input, DType::Float8E4M3FNUZ, minimum_scale);

config.fp8_activation_scale_mode = Fp8ActivationScaleMode::TensorAmax;
config.fp8_activation_scale = 1.0e-4F;  // minimum, no longer a fixed scale
```

CLI：

```text
--fp8-linear true
--fp8-weight-scale-mode tensor-amax
--fp8-activation-scale-mode tensor-amax
--fp8-activation-scale 0.0001
```

这个scale覆盖一个Linear的完整输入Tensor，不是per-row/per-token。

## 当前测试

- CPU amax、minimum fallback、非有限值拒绝和解量化参考；
- HIP FP32/FP8量化与device-scale反量化；
- HIP transfer counter：动态量化+反量化0 H2D、0 D2H；
- prepared Transformer同时使用weight/activation tensor-amax，热路径0 payload transfer；
- 动态activation不保留无用的persistent scale，报告字节数减少一半；
- official完整logits与吞吐：36/36执行；RMS相对weight-only下降63%–81%，但0/4过门；
  single-block amax让T512相对BF16只剩4.4%–5.3%吞吐。保留基础设施，拒绝当前模型策略，
  详见[Experiment 129](../optimization-log/experiments/129-fp8-device-activation-amax.md)。

第一版amax用一个256-thread block扫描完整Tensor，是正确性候选。正式模型若精度过门，再根据
trace决定是否做多block reduction；不能先用更复杂Kernel掩盖数值策略错误。
