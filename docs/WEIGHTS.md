# 权重 API：把外部模型的数字安全地装进框架

这份文档只讲框架权重，不等于某个模型架构已经兼容。

## 完整临时参数比较

`microllm_compare_safetensors BASELINE CANDIDATE` 要求两份文件的 Tensor 名字、shape
和 FP32 dtype 完全相同，然后比较每一个值，输出整体 Max/RMS、有限性、元素数量和最差
Tensor 名字。它用于不能把多 GB 参数快照提交进仓库的训练实验。这个诊断工具不替代下文
的原子模型加载和保存合同。

## 1. 权重是什么

模型由许多有名字的 Tensor 组成，例如：

```text
token_embedding.weight
blocks.0.attention.q_proj.weight
blocks.0.ffn_norm.weight
final_norm.weight
```

可以把模型想成一台有很多抽屉的机器：名字是抽屉标签，Tensor 是抽屉里的数字。只有数字数量、排列 shape 和抽屉用途都一致，模型才能正确运行。

`StateDict` 就是一张“标签 → Tensor”的清单：

```cpp
using StateDict = std::map<std::string, Tensor>;
```

## 2. 导出 state_dict

```cpp
model::TransformerModel model(config);
auto weights = model.state_dict();
```

导出的 Tensor 是独立副本。修改 `weights` 不会偷偷修改模型。也可以指定目标设备：

```cpp
auto gpu_snapshot = model.state_dict(Device::hip(0));
```

## 3. strict 加载为什么重要

最危险的情况不是程序报错，而是只加载一部分权重后继续生成。

strict 模式先检查全部权重，再修改模型：

- missing：模型需要，但文件中没有；
- unexpected：文件中存在，但模型不认识；
- incompatible：名字找到了，但 dtype、shape 或变换不合法。

只要任何一类不为空，strict 模式抛出错误，而且一个参数也不会修改。

```cpp
auto report = model.load_state_dict(weights);  // strict=true
```

non-strict 用于研究和局部初始化。它只加载兼容项，并返回完整报告：

```cpp
model::LoadWeightsOptions options;
options.strict = false;
auto report = model.load_state_dict(weights, options);
```

non-strict 不是“忽略错误后模型一定可用”。调用者必须阅读 `missing`、`unexpected` 和 `incompatible`。

## 4. 为什么外部线性权重要转置

microLLM 的线性权重使用：

```text
[input_dimension, output_dimension]
output = input × weight
```

Hugging Face/PyTorch 的 `nn.Linear.weight` 通常保存为：

```text
[output_dimension, input_dimension]
```

它们表示同一个数学变换，但文件排列方向相反。`WeightMapping` 同时描述外部名字和变换：

```cpp
mapping["blocks.0.attention.q_proj.weight"] = {
    "model.layers.0.self_attn.q_proj.weight",
    model::WeightTransform::Transpose2D,
};
```

加载器先验证源 Tensor 是二维，再转置、检查最终 shape，最后才写入模型。

## 5. Qwen 风格名字映射

```cpp
auto mapping = model::qwen_style_weight_mapping(config);
model::LoadWeightsOptions options;
options.mapping = mapping;
auto report = model.load_safetensors("model.safetensors", options);
```

这个 API 解决的是名字和线性权重方向。它不会凭空补出当前模型没有的层。

例如：

- Qwen3 有 QK-Norm 和显式 head dimension；
- Qwen2.5 的一些模型含 Q/K/V bias；
- DeepSeek-V3 使用 MLA、MoE、FP8 scale 和专家并行。

如果外部文件包含这些当前模型没有的参数，strict 模式会把它们报告为 unexpected，而不是假装加载成功。

## 6. safetensors 文件长什么样

safetensors 可以理解成：

```text
8 字节 header 长度
JSON header：每个 Tensor 的名字、dtype、shape、数据起止位置
连续二进制数据
```

本项目当前读取：

- `F32`；
- `BF16`，读取时转换为 FP32；
- `F16`，读取时转换为 FP32；
- `I8`，保持一字节有符号整数，供显式量化权重API使用。

写出 API：

```cpp
io::save_safetensors("weights.safetensors", model.state_dict());

io::SafetensorsSaveOptions options;
options.dtype = io::WeightFileDType::BFloat16;
model.save_safetensors("weights-bf16.safetensors", options);
```

混合INT8权重和FP32 scale时使用保真模式：

```cpp
io::StateDict state{
    {"linear.weight", packed.values},
    {"linear.weight.scale", packed.scale},
};
io::save_safetensors(
    "linear-int8.safetensors", state,
    {.dtype = io::WeightFileDType::Preserve});
```

当前`Preserve`只接受I8或F32 Tensor，防止不明确的隐式转换。完整量化、还原、HIP和
文件命名合同见[INT8权重格式](dev/int8-weight-format.zh-CN.md)。能保存这对Tensor不等于
Transformer已经使用INT8 GEMM；默认模型计算路由仍需单独实现和验收。

`TransformerModel::prepare_int8_inference_weights()`提供显式、单向、事务式研究路线：全部
Linear候选成功后才释放FP32 Linear权重；Embedding与Norm保持FP32。准备后只能调用graph-free
inference，Autograd、重新加载和`state_dict()`会拒绝。M=1 FP32输入使用显式融合decode，M>1
回到完整反量化基线，因此它尚不是默认或通用INT8模型路径。
HIP准备使用device-only amax，不回传权重payload。固定Qwen虽然常驻显著下降且短decode更快，
完整logits与token仍失败；这证明文件和准备API可用，不证明量化模型正确。

默认计算仍是 FP32。推理可在加载后调用 `prepare_bf16_ffn_inference()` 和
`prepare_bf16_attention_inference()`，事务式地把 FFN 与 Q/K/V/O 权重替换成单份 BF16；
Norm、Embedding、KV cache 和 tied 输出头仍保持 FP32。能读取 BF16/F16 文件本身仍不
等于启用了这条计算路径。

## 7. 单文件、多分片和 index

单文件：

```cpp
auto state = io::load_safetensors("model.safetensors");
```

明确给出多个分片：

```cpp
auto state = io::load_safetensors_files({
    "model-00001.safetensors",
    "model-00002.safetensors",
});
```

Hugging Face index：

```cpp
auto state = io::load_safetensors_index("model.safetensors.index.json");
```

index 中的路径必须是相对路径，不能使用 `..` 逃出 index 目录。分片中出现重复权重、index 指向不存在的权重或数据范围越过文件都会失败。

## 8. 直接加载到 GPU

```cpp
auto gpu_state = io::load_safetensors("model.safetensors", Device::hip(0));

model.to(Device::hip(0));
model.load_safetensors("model.safetensors");
```

第二种写法会按照模型参数所在设备复制权重。加载后旧梯度会清空，避免用旧模型梯度更新新模型权重。

如果模型以 `ParameterInitialization::Uninitialized` 创建并已经移动到 HIP，单文件
`model.load_safetensors()` 会走流式快路径：先只检查完整 header，再按文件顺序读取 Tensor。
BF16/F16 不在 CPU 展开成 FP32；它们以原始 2-byte payload 进入一块可复用 staging，随后
cast 或 cast+transpose 直接写模型已有参数 Storage。

只检查或自己消费原始 payload 可以使用：

```cpp
auto metadata = io::inspect_safetensors("model.safetensors");
io::visit_safetensors("model.safetensors", [](const auto& info, auto bytes) {
    // bytes 只在本次回调中有效；Tensor 按文件 offset 顺序到达。
});
```

## 9. 当前内存边界

已初始化模型仍先完整读取 StateDict，再验证并一次提交，保持原子替换语义。未初始化 HIP
模型的单文件快路径把临时峰值限制为“FP32 参数 + 最大低精度 staging”。固定 MI300X 实测：

- Qwen2.5-0.5B：17.659 秒降到 0.580 秒；
- DeepSeek Distill 1.5B：65.100 秒降到 1.356 秒；
- H2D 字节恰好等于 BF16 文件 payload，没有 D2H。

继续扩展到任意大模型还需要：

- memory mapping；
- 对所有分片先做全局 header 预检，再直接送到目标 GPU/rank；
- FP8、INT4、逐通道INT8和对应scale/zero-point流式模型装载；
- tied weight 不重复分配；
- 加载进度、取消和峰值内存报告。

Qwen2.5-0.5B 已经通过一个固定官方 checkpoint 的严格加载和 logits 对齐；这仍不
代表任意 Hugging Face 大模型都兼容。其他 Qwen 规模、Qwen3、DeepSeek、量化格式
仍必须分别通过 config、权重、tokenizer 和完整 logits 门。

官方fixture不把权重放进仓库。`tools/prepare_hf_fixture.py`读取
`data/model_fixtures.toml`中的固定revision，支持下载或验证已有目录，并通过完整header/index
重新计算参数量与Tensor数。生成的本地manifest可直接传给benchmark；可提交evidence不会记录
本机payload路径。

## 10. 测试门

权重 API 的独立测试覆盖：

- state_dict 独立副本；
- strict 原子失败；
- non-strict 完整报告；
- 加载后 forward 一致；
- Qwen 名字和二维转置；
- tied embedding 映射；
- F32/BF16/F16 round-trip，以及混合I8权重/F32 scale保真round-trip；
- 单文件、多个分片和 index；
- 损坏 header、不支持 dtype、重复权重和不安全路径；
- safetensors 直接加载到 HIP；
- I8/F32文件直接加载到HIP，以及反量化热路径零H2D/D2H；
- header inspection、payload-order visitor 和回调生命周期；
- 未初始化 HIP 单文件的 BF16 原始字节数、staging 峰值与失败前零传输；
- GPU 模型保持 GPU 参数。

### 用官方 safetensors 包做双向检查

“自己写、自己读”可能让 writer 和 reader 共享同一个错误。因此仓库还提供一个
可选的外部互操作测试：

```bash
python -m pip install torch safetensors packaging numpy

cmake -S . -B build/safetensors-interop \
  -DMICROLLM_ENABLE_HIP=OFF \
  -DMICROLLM_BUILD_TORCH_OPS=OFF \
  -DMICROLLM_SAFETENSORS_PYTHON=/path/to/python
cmake --build build/safetensors-interop \
  --target microllm_safetensors_interop --parallel
ctest --test-dir build/safetensors-interop \
  -R '^Safetensors.OfficialInterop$' --output-on-failure
```

测试做两个方向：

1. C++ 写文件，官方 Python 包读取并检查名字、dtype、shape 和每个值；
2. 官方 Python 包写文件，C++ 读取并检查相同内容。

F32、BF16、F16和混合I8/F32文件都会执行。这个测试是可选门，因为默认 CPU
构建不应偷偷下载 Python 包；配置了指定解释器后，它会成为正式 CTest。
