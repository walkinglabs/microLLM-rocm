# N9 — 怎样证明一个官方 Hugging Face 模型真的跑对了

本章不从“下载权重”开始。先理解为什么一个模型目录里有好几类文件，以及每一类
都可能让最终文字发生变化。

## 运行前先预测

模型目录通常包含：

```text
config.json          层数、hidden size、head、RoPE、Norm 等规则
model.safetensors    参数数字
vocab.json           token 字符串到 ID 的表
merges.txt           小 token 合并规则
tokenizer_config     特殊 token 和 chat template
```

请先回答：如果 290 个权重 Tensor 全部成功读取，但 RoPE 排列方式错了，程序会
崩溃、报错，还是可能正常输出一串错误 logits？

答案是第三种。文件能读出来只证明“箱子打开了”，不证明这些数字放到了正确位置。

## 五道证据门

支持一个外部模型至少要依次通过：

```text
config 门
  ↓ 参数量和每层 shape 正确
tokenizer 门
  ↓ 文本得到完全相同的 token IDs
weight 门
  ↓ 名字、shape、转置和数量严格匹配
logits 门
  ↓ 同输入下完整词表数值接近 PyTorch
generation 门
  ↓ 连续多个 greedy token 完全一致
```

训练还要增加第六道门：相同 token、权重、loss 和学习率下，梯度与参数更新对齐。

## 第一步：只检查 config，不申请五亿参数

```bash
"$MICROLLM_ENGINE_DIR/build/cpu-debug/apps/microllm_hf_inspect" \
  --config "$MICROLLM_ENGINE_DIR/tests/fixtures/qwen25-0.5b-config.json"
```

检查输出中的：

- `model_type` 是否是已经实现的 Qwen2；
- 参数总数是否为 `494032768`；
- hidden size、层数、query heads、KV heads 是否互相整除；
- Q/K/V 是否需要 bias；
- RoPE 是否使用 Qwen 的 split-half 排列；
- RMSNorm epsilon 是否来自 config，而不是写死。

当前解析器会拒绝尚未实现的 sliding-window Attention、MRoPE 和其他模型家族。
明确报错比“勉强运行一个错误结构”更安全。

## 第二步：检查 tokenizer

```bash
"$MICROLLM_ENGINE_DIR/build/cpu-debug/apps/microllm_hf_tokenize" \
  --vocab /path/to/qwen/vocab.json \
  --merges /path/to/qwen/merges.txt \
  --text "Hello world"
```

Qwen2.5 参考 IDs 是 `[9707,1879]`。只测英文不够，还要覆盖：

- 中文；
- emoji；
- 一个和多个空格；
- 换行；
- 缩写；
- 普通文本中出现特殊 token；
- system/user/assistant chat 边界。

Tokenizer 的验收要求 ID 完全一致，不能使用浮点误差容忍。

## 第三步：严格加载权重并运行 Qwen

先记录官方模型的 revision、许可证和本地文件来源。权重不提交到课程分支。

```bash
"$MICROLLM_ENGINE_DIR/build/hip-release/apps/microllm_hf_infer" \
  --config /path/to/qwen/config.json \
  --weights /path/to/qwen/model.safetensors \
  --vocab /path/to/qwen/vocab.json \
  --merges /path/to/qwen/merges.txt \
  --text "Hello world" \
  --device hip \
  --new-tokens 4 \
  --logits-output /tmp/microllm-qwen-logits.f32
```

`strict` 加载必须证明 290 个 Tensor 全部匹配，同时不存在多余权重和缺失权重。
固定实验应生成：

```text
IDs   [0,358,2776,264]
文字  ! I'm a
```

2026-08-19 的 MI300X FP32 记录中，完整 logits 与 Transformers 的最大绝对差为
`6.03199e-5`，4 个 greedy token 完全相同。这是一个固定输入的 smoke，不代表
所有 prompt 或所有 Qwen 尺寸。

## 第四步：不要只比较 top-1

两个实现可能恰好选择同一个最大 token，但其余 151935 个 logits 已经偏离。完整
对照流程在 `main` 的 Hugging Face 文档中：

- [Qwen 运行与 logits 对照](https://github.com/walkinglabs/microLLM-rocm/blob/main/docs/HUGGINGFACE.md)
- [PyTorch 参考工具](https://github.com/walkinglabs/microLLM-rocm/blob/main/tools/huggingface/pytorch_qwen_reference.py)
- [完整向量比较工具](https://github.com/walkinglabs/microLLM-rocm/blob/main/tools/huggingface/compare_logits.py)

报告至少保存：最大绝对误差、MSE、cosine、top-k token 是否相同以及非有限值数量。

## 第五步：运行一次真实训练更新

```bash
"$MICROLLM_ENGINE_DIR/build/hip-release/apps/microllm_hf_train_step" \
  --config /path/to/qwen/config.json \
  --weights /path/to/qwen/model.safetensors \
  --tokens 1,2,3,4 \
  --device hip \
  --learning-rate 0.00001
```

这个命令执行 forward、cross entropy、完整 backward 和 AdamW。验收不是“参数
发生变化”这么简单，还要比较 PyTorch 的 loss 和同一个参数更新，并检查优化器
期间 Tensor payload 的 H2D/D2H 次数都是 0。

历史 MI300X 记录：

```text
microLLM loss       6.83602953
PyTorch FP32 loss   6.83601570
绝对差              1.383e-5
optimizer H2D/D2H   0 / 0
```

## DeepSeek 为什么要写完整名字

当前跑通的是 `DeepSeek-R1-Distill-Qwen-1.5B`。它把推理数据蒸馏到密集 Qwen2.5
架构，可以复用相同的 GQA、Q/K/V bias、RoPE、RMSNorm 和 SwiGLU。

2026-08-19 的 MI300X FP32 smoke：

```text
参数数              1777088000
严格加载 Tensor     339
logits 最大绝对差   6.409e-5
greedy token         8/8 完全相同
```

它不证明旗舰 DeepSeek-R1/V3 已被支持。旗舰模型还需要 MLA、MoE、专家路由、
专家并行和不同的低精度规则。把 Distill-Qwen 简写成“支持 DeepSeek”会误导读者。

## 必做反例

从下面选择一个，先写预测，再运行 `main` 中对应的测试或实验：

1. 把 split-half RoPE 当成 interleaved，观察 logits 差异；
2. 忽略 Q/K/V bias，观察 strict shape 仍可能通过但数值失败；
3. 只比较 top-1，再用完整 logits 揭示被隐藏的偏差；
4. tokenizer 少加一个特殊 token，比较 chat prompt IDs。

## 本章提交物

- config 摘要和手算参数量；
- 至少 6 类文本的 tokenizer IDs 对照；
- strict 权重加载报告；
- 完整 logits 指标；
- 连续 greedy token 对照；
- 一个稳定失败；
- 明确写出“这个实验没有证明什么”。

下一章只改变 dtype，模型、权重、输入和生成规则保持不变，从而判断低精度究竟
节省了什么，又损失了什么。
