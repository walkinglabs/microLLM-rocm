# 权重 API：把外部模型的数字安全地装进框架

这份文档只讲框架权重，不等于某个模型架构已经兼容。

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
- `F16`，读取时转换为 FP32。

写出 API：

```cpp
io::save_safetensors("weights.safetensors", model.state_dict());

io::SafetensorsSaveOptions options;
options.dtype = io::WeightFileDType::BFloat16;
model.save_safetensors("weights-bf16.safetensors", options);
```

当前计算 Kernel 仍以 FP32 为主。能读取 BF16/F16 文件，不等于已经用 BF16/F16 Kernel 计算。

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

## 9. 当前内存边界

第一版先完整读取 StateDict，再验证并复制到模型。这很容易解释和测试，但加载大模型时会同时占用文件解码 Tensor 和模型 Tensor 两份内存。

真正加载 Qwen 大模型和 DeepSeek-V3 还需要：

- streaming：每次只读取一个 Tensor；
- memory mapping；
- 分片直接送到目标 GPU/rank；
- FP8、INT8、INT4 和对应 scale/zero-point；
- tied weight 不重复分配；
- 加载进度、取消和峰值内存报告。

在这些能力完成前，不能把“支持 safetensors API”写成“支持任意 Hugging Face 大模型”。

## 10. 测试门

权重 API 的独立测试覆盖：

- state_dict 独立副本；
- strict 原子失败；
- non-strict 完整报告；
- 加载后 forward 一致；
- Qwen 名字和二维转置；
- tied embedding 映射；
- F32/BF16/F16 round-trip；
- 单文件、多个分片和 index；
- 损坏 header、不支持 dtype、重复权重和不安全路径；
- safetensors 直接加载到 HIP；
- GPU 模型保持 GPU 参数。
