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

当前通过真机验证的固定目标包括 `Qwen/Qwen2.5-0.5B` 和
`Qwen/Qwen3-0.6B`。Qwen2.5 revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` 有 494,032,768 个运行时参数和
290 个权重 Tensor。Qwen3 revision
`c916fa4defd319b7d4e4da17604ca7338f4d99f5` 有 596,049,920 个运行时参数；
文件保存 311 个 Tensor、751,632,384 个值，因为 token embedding 和 lm_head 的 tied
payload 各保存了一份。Qwen3 strict loader 会先逐字节验证两份来源一致，再只建立
310 个运行时目标。

两种模型均已完成固定 prompt 的官方 logits/token 真机门。显式 BF16 路径有独立误差、
常驻显存和吞吐证据；这些结果不自动推广到其他 Qwen 尺寸。

扩展Qwen3 BF16矩阵进一步说明“固定短prompt通过”不是全shape结论：T1/32/128/512、B1/B2
的64个framework进程全部运行，但24个decode row中只有16个完整token-exact，另外8个必须标记
`precision_mismatch`。两边policy不同，当前只记录边界；第一处分叉的共同FP32 full-logit
实验现已完成：T32/B1的FP32 margin只有0.03260，Transformers整网BF16把前两名舍入成
14.1875平局，microLLM mixed BF16保留FP32 argmax。完整5-case sweep中mixed/full BF16
匹配FP32为4/1，映射到8行是7/1；T128/B2是反例，不能写成通用mixed-BF16胜利。
该反例进一步隔离到FFN-only BF16：它选择25，Attention-only仍选择FP32的320；BF16 Cache只
缩小误差，没有恢复argmax。下一步只定位FFN层。

### 一条命令准备固定 fixture

仓库不会提交数GB权重，但会固定来源、revision、许可和结构预期：

```bash
python3 tools/prepare_hf_fixture.py prepare \
  --download-root /absolute/path/to/models \
  --manifest /absolute/path/to/hf-models.local.json \
  --evidence /tmp/hf-fixture-evidence.json
```

注册表在`data/model_fixtures.toml`。工具下载指定revision，检查完整safetensors header/安全分片
index、参数量、Tensor数、config、vocab与merges，再生成下文所有runner共用的manifest。
`stored_parameter_count`回答“文件里写了多少个值”，`runtime_parameter_count`回答“模型真正拥有
多少个独立参数”；旧manifest的`parameter_count`保持为runner所需的运行时口径。Qwen来源标记
Apache-2.0；DeepSeek Distill来源标记MIT。使用者仍应阅读注册表中的官方license链接。

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

`--batch`支持full-sequence prefill、cached decode和uncached reference decode。
`--use-cache true|false`用于检查两条路径是否生成相同token；当前batch内所有请求必须共享
长度和position，continuous batching/request scheduling仍是后续能力。

`--kv-cache-dtype fp32|bf16`选择Cache Storage类型。默认FP32；BF16严格把Cache字节
减半，但当前是显式实验策略，因为DeepSeek有一条Release RMSE门失败。完整设计、API和
误差门见[KV Cache数据类型](dev/kv-cache-dtypes.zh-CN.md)。

`--kv-cache-fp32-layers 1,5`可以在BF16基础上让指定层使用FP32。固定DeepSeek实验中，
layer 1只对原prompt有效；robust-strict使用layers 0–3 FP32、其余BF16，14/14多prompt
挑战通过、Cache仍缩小1.75×。层选择来自固定checkpoint实验，因此不会自动启用。

Qwen uniform BF16在repeat/rotated/ramp通过，但constant T32/512/2048完整logits失败；
constant T2048只有全FP32 Cache通过。`fp32`默认值因此仍是必要fallback。

`--bf16-ffn`与`--bf16-attention`是独立的权重准备开关。普通运行通常同时开启；精度诊断可以只
开启一个，用JSON中的两个`*_converted_tensors`字段验收实际路径。Attention-only不隐式准备FFN。

`--bf16-ffn-fp32-layers 0,3`是在`--bf16-ffn true`基础上的显式研究开关：指定Block的三个FFN
权重和激活保持FP32，其余Block仍单向准备为BF16。索引必须唯一且在模型层数内；JSON同时输出
`bf16_ffn_fp32_layers`和实际转换数。空列表保持原来的全层BF16行为。这个接口用于可反驳的逐层
精度实验，不会按模型名自动选择层，也不改变默认策略。

`--bf16-decode-algorithm-index N`为当前cached decode的`M=batch,K=hidden,N=ffn` gate/up BF16
GEMM注册一个version-local hipBLASLt solution。它要求HIP、cached decode和BF16 FFN；不会影响
prefill或down projection。JSON报告index与registry count。solution编号不是跨ROCm稳定API，只有
完成exact-shape support、CPU数值和完整模型门后才能用于实验，不能写成硬编码默认。

`--fp32-prefill-q-solution-index N`和`--fp32-prefill-kv-solution-index N`只为full cached prefill的
FP32 Q投影或共享K/V投影注册version-local solution。registry key除shape/environment外还包含明确的
projection scope，因此不会误命中同形Attention output或FFN Linear。它们要求HIP、full cached
decode和FP32 Attention权重；JSON报告index以及registry hit/miss/cache/dispatch计数。默认均为`-1`。

`--cache-logits-output PATH`保存真正经过cached decode后的完整logits，只用于精度诊断。
默认保存最后一步；`--cache-logits-step N`可选择`0 <= N < new_tokens`的具体decode步，包含
batch的完整`[B,V]` FP32值。至少生成一个token；开启诊断输出的运行不作为正式性能排名。

`--prefill-cache-output PATH --prefill-cache-layer N`导出full prefill结束、decode开始前的指定层K/V
活动前缀。文件第一行是JSON元数据，后面依次是packed key和value原始dtype字节，因此BF16可以按位
比较而不先转FP32。它只允许cached decode、full prefill、warmup 0、steps 1；导出搬运属于诊断，
不能作为性能行。JSON同时报告layer、dtype、shape和K/V字节数。

cached模式默认`--cache-prefill-mode full`，一次完整prompt直接填入每层预分配Storage。
显式`token`会逐token重放，只用于复现旧性能失败和reference；发布结果必须记录所选模式。

uncached batch默认`--batch-argmax-mode device`，只把每行选中的Int32 token带回host。
显式`host`会搬回完整`B×V` logits，仅用于性能反例；它不是生产默认。

HIP greedy且没有stop token时，cached generation会把全部选中ID写进device token history，结束后
一次带回host；随机sampling和提前停止仍逐步读取。接口、边界和测试见
[GPU token history](dev/device-token-history.zh-CN.md)。

多context、batch和KV Cache显存矩阵见：

```bash
python3 benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /path/to/hf-models.local.json \
  --micro-binary build/apps/microllm_hf_infer \
  --pytorch-python /usr/local/bin/python3 \
  --output-directory /tmp/inference-matrix \
  --contexts 8,128,512 --batches 1,2,4,8 \
  --decode-lengths 1,8,32 --warmup 1 --steps 3 --runs 3
```

runner分别记录microLLM混合驻留政策和PyTorch整网BF16，不能只写一个“BF16”掩盖
常驻权重差异。它还使用steady decode语义，保证每个计入吞吐的token真的执行一次模型
forward，并分别记录KV Cache的预留、活跃字节、每请求成本和峰值显存。详细解释见
[推理矩阵设计](dev/inference-matrix.zh-CN.md)。

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
- 固定 Qwen3-0.6B 的显式 head dimension 与 QK-Norm 已通过；其他 Qwen3 尺寸、训练、
  长上下文和量化策略仍要分别验收。DeepSeek MLA/MoE 是另一类结构。
- 未初始化 HIP 模型的单文件权重已使用 header 预检和低精度 streaming；多 shard/index
  仍保留完整 StateDict 原子路径，不能把单文件加载速度推广到所有 checkpoint 布局。
