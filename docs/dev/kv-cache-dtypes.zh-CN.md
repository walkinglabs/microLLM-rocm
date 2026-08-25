# KV Cache 数据类型：把“草稿本”变薄，但不能看错答案

这篇文档只讲一个问题：推理时，为什么要选择 FP32 或 BF16 KV Cache？

## 1. 先把 KV Cache 想成一本草稿本

模型读到一句话时，每一层都会记下一些中间结果，后面生成新 token 时再回来查。
这些中间结果叫 Key 和 Value，保存它们的地方叫 KV Cache。

如果不保存，模型每生成一个 token，都要从头重算整段文字。保存以后，它只计算新来的
token，再查以前的草稿，所以长文本生成会快很多。

一层 Cache 的形状是：

```text
[batch, kv_heads, capacity, head_dimension]
```

- `batch`：同时处理多少条请求；
- `kv_heads`：这一层有多少组 Key/Value；
- `capacity`：这次请求最多能放多少 token；
- `head_dimension`：每个 head 写多少个数字。

每一层都有一份 Key 和一份 Value，所以理论字节数是：

```text
2 × layers × batch × kv_heads × capacity × head_dimension × 每个数字的字节数
```

## 2. FP32 和 BF16 有什么不同

FP32 每个数字占 4 字节，BF16 每个数字占 2 字节。可以把它们想成同一段笔记的
“大字版”和“缩写版”：

```text
FP32 Cache：信息更细，4 字节
BF16 Cache：信息更粗，2 字节
```

因此，同一个 shape 的 BF16 Cache 必须恰好只有 FP32 的一半大。若不是一半，说明
shape、stride、dtype 或统计方法至少有一个出错。

BF16 不是把整个 Attention 都改成低精度。本仓库采用：

```text
FP32 Query
+ BF16 Key/Value Storage
→ 读取时转成 FP32
→ 点积、softmax、求和都用 FP32
→ FP32 输出
```

这样只减少“草稿本”的宽度，不同时改变 softmax 和累加规则，实验更容易解释。

## 3. 为什么 BF16 不是默认值

数字变短会丢掉一部分细节。短文本可能完全不影响最终 token，长文本中的许多小误差却
可能累积，并在两个候选 token 很接近时改变选择。

MI300X 实测中：

- Qwen 在 context 32、512、2048 的 B1/B8 测试里通过既定 logit 和 token 门；
- Release矩阵中DeepSeek有5/6 shape通过，T512 B1的RMSE为0.0586，超过0.05门；
- Release的12个shape中16个greedy token都一致；普通构建的诊断实验曾在DeepSeek
  T2048发生token分叉，说明结果会受编译配置影响，不能只测一种构建后推广。

所以当前规则是：

```text
默认：FP32 Cache
显式实验：BF16 Cache
```

不能因为显存减半，就删掉RMSE失败、build-sensitive反例，或把BF16偷偷设成所有模型的
默认值。

现在还有第三种“混合草稿本”：只有敏感层用FP32，其余层用BF16。单独layer 1只对原prompt
有效，换constant/ramp后会失败。当前固定DeepSeek的robust-strict配方使用layers 0–3 FP32，
14/14挑战通过，Cache仍缩小1.75×。它是checkpoint特定选项，不是新的默认值。

Qwen没有找到同样的robust低精度配方。constant T2048中，前4/8/12层FP32仍发生巨大logit
误差和token分叉，只有全FP32通过。因此需要严格门时必须允许按输入/策略回退FP32。

## 4. C++ API 怎样选择

直接构造 Cache：

```cpp
microllm::inference::KVCache cache(
    layers, capacity, batch, microllm::DType::BFloat16);
```

普通生成接口：

```cpp
microllm::inference::GenerationConfig config;
config.kv_cache_dtype = microllm::DType::BFloat16;
auto tokens = microllm::inference::generate(model, prompt, config);
```

不传 dtype 时仍使用 FP32。

逐层策略直接传一个与模型层数相同的列表：

```cpp
microllm::inference::KVCache cache(
    {microllm::DType::BFloat16,
     microllm::DType::Float32,
     microllm::DType::BFloat16},
    capacity);
```

普通生成可以设置`GenerationConfig::kv_cache_layer_dtypes`。列表长度不等于模型层数、
出现FP16/FP8或空的逐层Cache构造都会直接报错。

## 5. 命令行怎样选择

```bash
build/apps/microllm_hf_infer \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokens 1,2,3,4 \
  --device hip \
  --workload decode \
  --new-tokens 4 \
  --use-cache true \
  --kv-cache-dtype bf16
```

可选值只有 `fp32` 和 `bf16`。程序会输出类型、实际 Storage 字节、活跃 view 字节、
元素字节数和利用率。

在BF16基础上覆盖个别FP32层：

```text
--kv-cache-dtype bf16 --kv-cache-fp32-layers 0,1,2,3
```

层编号从0开始，必须唯一且在模型层数内。输出会分开报告FP32/BF16层数和字节；混合
Cache没有一个统一的“每元素字节数”，因此该字段为0，理论字节按每层真实dtype求和。

`--cache-logits-output PATH` 可以保存真正经过 cached decode 后的完整 logits。默认保存最后一步；
`--cache-logits-step N`可以选择从0开始的具体步骤，并保存完整`[B,V]` FP32值。它只用于精度诊断，
要求至少生成一个 token；不要拿开启诊断输出的运行做正式性能排名。

## 6. 怎样判断做对了

证据分四层：

1. 算子：HIP 输出与“先把 K/V 圆整到 BF16”的 CPU reference 对齐；
2. 模型：FP32/BF16 Cache 的完整 logits 比较最大误差和 RMSE；
3. 生成：固定 prompt 的 greedy token 逐项比较；
4. 系统：Cache 字节必须精确减半，并单独报告整机峰值和吞吐。

本节点的模型门是：

```text
maximum absolute error <= 0.25
RMSE <= 0.05
top-1 token 相同
生成 token 全部相同
```

“程序能跑”不代表精度通过。“Cache 减半”也不代表整机峰值减半，因为模型权重和临时
activation 仍然占显存。

## 7. 测试在哪里

```text
tests/ops/ops_test.cpp
  CPU BF16 Storage、圆整和 Attention

tests/ops/hip_ops_test.cpp
  BF16 store、B2、fused 路径、T4097 fallback、零 host payload transfer

tests/model/model_test.cpp
  prefill、继续 decode、Cache 字节和 logits

tests/inference/generator_test.cpp
  公共生成 API 的 dtype 选择和 token

python/tests/test_kv_cache_precision.py
  真实模型矩阵的字节、logit、token 和状态规则
```

## 8. 为什么本节点不加入 FP8

FP8 还需要 scale、动态范围和 E4M3/E5M2 选择。把它与 BF16 一起改，会同时改变太多
条件，无法判断速度或误差来自哪里。FP8 KV Cache 应是后续独立实验，并保留自己的失败门。
