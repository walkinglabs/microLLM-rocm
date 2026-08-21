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
权重 Tensor。默认路径把 BF16 文件准确转换成 FP32 reference；可选的单份 BF16 FFN
准备路径已经在同一 revision 上完成 logits、exact token、显存和吞吐实测。

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
  --new-tokens 4 \
  --warmup 2 \
  --steps 5
```

固定实验生成 `[0,358,2776,264]`，文字是 `! I'm a`，与 Transformers FP32
完全一致。

`--warmup` 会运行完整生成但不计入吞吐；完成后框架重置峰值分配计数。cached模式把
prompt写入Cache的时间单独记为`mean_cache_prepare_ms`，`decode_tokens_per_second`只计算
steady decode；两者之和写入`mean_end_to_end_generation_ms`。默认值仍是`warmup=0`、
`steps=1`，只适合一次性正确性检查，不能与排除初始化的正式性能行比较。

`--batch`支持full-sequence prefill和uncached reference decode。当前cached decode只支持
batch1；B2/B4/B8会明确失败，不能静默退回B1。`--use-cache true|false`用于检查两条路径
是否生成相同token。

cached模式默认`--cache-prefill-mode full`，一次完整prompt直接填入每层预分配Storage。
显式`token`会逐token重放，只用于复现旧性能失败和reference；发布结果必须记录所选模式。

多context、batch和KV Cache显存矩阵见：

```bash
python3 benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --micro-binary build/apps/microllm_hf_infer \
  --pytorch-python /usr/local/bin/python3 \
  --output-directory /tmp/inference-matrix \
  --contexts 8,128,512 --batches 1,2,4,8 \
  --decode-tokens 4 --warmup 1 --steps 2 --runs 3
```

runner分别记录microLLM混合驻留政策和PyTorch整网BF16，不能只写一个“BF16”掩盖
常驻权重差异。详细解释见[推理矩阵设计](dev/inference-matrix.zh-CN.md)。

多步训练使用相同规则：

```bash
build/hip-release/apps/microllm_hf_train_step \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokens 1,2,3,4 \
  --device hip \
  --learning-rate 0.00001 \
  --warmup 2 \
  --steps 5
```

输出分别保留 `warmup_ms`、`measured_ms`、`mean_step_ms`、15 个 measured token 的
吞吐和 measured 区间峰值显存。不得把 warm-up 时间混入 measured throughput，也
不得删除 setup/warm-up 字段后再发布结果。

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
- 当前 BF16 覆盖单份 Linear 混合推理和 FP32-master Linear 训练；continuous BF16
  training island 与 FP8 整网仍要分别报告；
- Qwen3 的 QK-Norm、DeepSeek MLA/MoE 是不同结构。
- 未初始化 HIP 模型的单文件权重已使用 header 预检和低精度 streaming；多 shard/index
  仍保留完整 StateDict 原子路径，不能把单文件加载速度推广到所有 checkpoint 布局。
