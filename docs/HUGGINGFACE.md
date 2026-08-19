# Hugging Face 模型：从 config 到真实 Qwen 输出

这份文档用简单方式说明：一个 Hugging Face 模型为什么不能“下载权重就运行”，
以及怎样在 microLLM 中逐层验收。

## 1. 一个模型目录里有什么

```text
config.json          模型层数、Attention、RoPE 和 Norm 规则
model.safetensors    每一层真正的参数数字
vocab.json           token 字符串对应的整数 ID
merges.txt           小字符怎样合并成较长 token
tokenizer_config     特殊 token 和聊天模板
```

只读懂 safetensors 不等于支持模型。shape、bias、RoPE 排列、epsilon、tokenizer
中任何一处不同，程序都可能正常运行却输出错误文字。

## 2. 当前验证目标

当前通过真机验证的是 `Qwen/Qwen2.5-0.5B` revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`。它有 494,032,768 个参数和 290 个
权重 Tensor。BF16 文件准确转换成 FP32，再在 MI300X 上执行 FP32 reference。

## 3. 先检查 config

```bash
build/cpu-debug/apps/microllm_hf_inspect \
  --config tests/fixtures/qwen25-0.5b-config.json
```

程序会拒绝当前未实现的模型家族、sliding-window Attention 和 MRoPE，避免静默
使用错误结构。

## 4. 检查 tokenizer

```bash
build/cpu-debug/apps/microllm_hf_tokenize \
  --vocab /path/to/vocab.json \
  --merges /path/to/merges.txt \
  --text "Hello world"
```

Qwen2.5 的正确 IDs 是 `[9707,1879]`。C++ 测试还覆盖中文、连续空格、换行、
emoji、缩写和数字。基础 Instruct system/user/assistant 模板的 30 个 token IDs 也
与 Transformers 完全一致。

## 5. 在 MI300X 上运行真实权重

```bash
build/hip-release/apps/microllm_hf_infer \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --vocab /path/to/vocab.json \
  --merges /path/to/merges.txt \
  --text "Hello world" \
  --device hip \
  --new-tokens 4
```

固定实验生成 `[0,358,2776,264]`，文字是 `! I'm a`，与 Transformers FP32
完全一致。

## 6. 比较完整 logits

用 `--logits-output` 保存完整 FP32 logits，然后运行：

```bash
python tools/huggingface/pytorch_qwen_reference.py \
  --model-directory /path/to/local-model \
  --tokens 9707,1879 \
  --output /tmp/qwen-reference

python tools/huggingface/compare_logits.py \
  --microllm /tmp/microllm-logits.f32 \
  --pytorch /tmp/qwen-reference/pytorch_logits.f32 \
  --output /tmp/qwen-comparison.json
```

记录结果位于 `benchmarks/results/2026-08-19-qwen25-0.5b/`。

## 7. 当前边界

- 通过 0.5B 不等于所有 Qwen2.5 尺寸都已实测；
- 基础 Instruct chat template 和三个特殊 token 已通过；工具调用模板尚未实现；
- 当前真实模型对照是 FP32 compute，BF16/FP8 端到端仍要分别报告；
- Qwen3 的 QK-Norm、DeepSeek MLA/MoE 是不同结构。
