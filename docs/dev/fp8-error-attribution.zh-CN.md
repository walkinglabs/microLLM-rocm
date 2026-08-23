# FP8误差从哪里来：把权重和激活分开检查

## 1. 先用一个简单比喻

一次Linear计算可以写成：

```text
输出 = 激活 × 权重
```

FP8像一把刻度比较粗的尺子。激活要用它量一次，权重也要用它量一次。如果最后答案不准，
只看最终结果无法知道是哪一把尺子带来的误差。

microLLM提供三种明确模式：

| 模式 | 激活 | 权重 | 乘法 | 用途 |
|---|---|---|---|---|
| `full` | FP8 | FP8 | 真实FP8 GEMM | 测最终FP8实现 |
| `weight-only` | 保持FP32 | FP8往返 | FP32 GEMM | 只看权重舍入误差 |
| `activation-only` | FP8往返 | 保持FP32 | FP32 GEMM | 只看激活舍入误差 |
| `both-roundtrip` | FP8往返 | FP8往返 | FP32 GEMM | 区分舍入与原生GEMM |

“FP8往返”表示先压成FP8，再还原成FP32。还原不能找回已经丢掉的信息，因此留下的差异就是
这一侧的量化误差。后两种模式故意使用FP32 GEMM，它们是诊断工具，不是加速路径。

## 2. 为什么不能把两种诊断速度当成FP8速度

`weight-only`每次Linear都要把已保存的FP8权重临时还原为FP32；`activation-only`也要把激活
还原后再计算；`both-roundtrip`两边都要还原。这多做了Kernel和临时内存工作。它们回答
“误差从哪里来”，不回答“FP8有多快”。

只有`full`会进入`ops::fp8_matmul`和hipBLASLt FP8路径。结果JSON中的以下字段能防止口径混淆：

```text
fp8_diagnostic_mode
compute_dtype
inference_weight_policy
fp8_linears_covered
fp8_converted_tensors
fp8_dynamic_tensor_calls
fp8_native_shapes
```

## 3. C++配置

```cpp
microllm::model::ModelConfig config = /* 模型配置 */;
config.linear_precision =
    microllm::model::LinearPrecision::Float8E4M3FNUZ;
config.fp8_weight_scale_mode =
    microllm::model::Fp8WeightScaleMode::DeviceTensorAmax;
config.fp8_activation_scale_mode =
    microllm::model::Fp8ActivationScaleMode::TensorAmax;
config.fp8_diagnostic_mode =
    microllm::model::Fp8DiagnosticMode::WeightOnly;

microllm::model::TransformerModel model(config);
auto report = model.prepare_fp8_inference_weights();
auto logits = model.forward_inference(tokens);
```

`Fp8DiagnosticMode`在非FP8模型中会被拒绝。诊断模式只支持graph-free inference；调用训练图
`forward()`会报错，避免把诊断语义悄悄带进反向传播。

## 4. 命令行

下面示例只隔离激活误差：

```bash
build/apps/microllm_hf_infer \
  --config /path/config.json \
  --weights /path/model.safetensors \
  --tokens 1,2,3,4,5,6,7,8 \
  --device hip \
  --fp8-linear true \
  --fp8-weight-scale-mode device-tensor-amax \
  --fp8-activation-scale-mode tensor-amax \
  --fp8-activation-minimum-scale 0.0001 \
  --fp8-diagnostic-mode activation-only \
  --workload prefill \
  --new-tokens 0
```

将最后一个值改成`weight-only`即可只测权重；改成`both-roundtrip`可让两边同时舍入但继续使用
FP32 GEMM；改成`full`才是完整FP8计算。

## 5. 官方模型矩阵

`benchmarks/single_gpu/hf_fp8_matrix.py`接受同名参数，并在每个shape中轮换FP32、BF16和目标
模式的进程顺序。它比较完整151,936维logits，而不是只看top token。

```bash
python3 benchmarks/single_gpu/hf_fp8_matrix.py \
  --manifest /path/model-manifest.json \
  --binary build/apps/microllm_hf_infer \
  --output-directory /tmp/fp8-weight-only \
  --models qwen2.5-0.5b \
  --contexts 8,512 \
  --fp8-weight-scale-mode device-tensor-amax \
  --fp8-activation-scale-mode tensor-amax \
  --fp8-diagnostic-mode weight-only
```

FP32同引擎路径是本实验的直接差分参考；PyTorch仍用于支持域的外部FP32/BF16总体验证，但它
不能替代这个单变量反事实，因为我们需要确保输入、权重和除目标舍入外的全部执行条件相同。

## 6. 自动检查证明了什么

- CPU：准备前后逐值一致；weight-only没有激活动态量化，activation-only保留全部FP32权重；
- HIP：两种诊断热路径均为0 H2D/0 D2H；只有activation-only增加动态激活调用；
- CLI：fresh binary必须包含参数名和`fp8_linears_covered`输出字段；
- runner：默认`full`保持兼容，并对显式诊断模式做命令合同检查；
- 全仓：Release/MI300配置356/356通过，其中CPU标签248、HIP标签108，2项按环境条件跳过。

这些测试证明“隔离开关按设计工作”，不能提前证明哪个误差源占主导。主导来源必须由两个官方
模型、短/长上下文的完整logits实测决定。

## 7. Exp141实际看到了什么

正式24个worker显示：Qwen的weight-only在T8/T512的Max和RMS都比activation-only大；DeepSeek
则是activation-only的RMS更大，但T512最坏坐标由weight-only主导。八个诊断精度门都失败。

所以不能写成“FP8只需要修一边”。`both-roundtrip`现已实现：两边同时经历FP8舍入、再用
FP32 GEMM；它用于区分双侧舍入共同传播和真实FP8 GEMM本身。完整Exp141数据见
[Experiment 141](../optimization-log/experiments/141-fp8-error-source-isolation.md)。

不能用两个独立summary的RMS差值代替直接比较。专用runner会在同一模型/上下文中轮换FP32、
`full`和`both-roundtrip`，并保存三组完整向量差：

```bash
python3 benchmarks/single_gpu/hf_fp8_native_roundtrip.py \
  --manifest /path/model-manifest.json \
  --binary build/apps/microllm_hf_infer \
  --output-directory /tmp/native-roundtrip \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --contexts 8,512 \
  --physical-gpu-index 2
```

`pairs.jsonl`中的`full_vs_both_roundtrip`才是判断原生GEMM额外差异的直接证据。
