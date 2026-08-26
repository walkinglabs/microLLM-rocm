# Qwen3 fixture → 通用推理矩阵

这组结果验证两件事：Qwen3 tied checkpoint 的双参数量 manifest 能被通用 runner 消费；
runner 不能把“两个进程都运行成功”误写成“跨框架答案一致”。

有效命令使用 MI300X、BF16 KV cache、exact cache capacity：

```bash
/tmp/microllm-torch-rocm-venv/bin/python \
  benchmarks/single_gpu/hf_inference_shape_matrix.py \
  --manifest /tmp/microllm-qwen3-runtime-manifest-v2.json \
  --micro-binary build/hip-release/apps/microllm_hf_infer \
  --pytorch-python /tmp/microllm-torch-rocm-venv/bin/python \
  --output-directory /tmp/qwen3-fixture-shape-matrix-v3 \
  --models qwen3-0.6b \
  --contexts 1,32,128,512 --batches 1,2 \
  --decode-lengths 1,4,32 --cases prefill,cached \
  --micro-kv-cache-dtype bf16 --micro-cache-capacity exact \
  --warmup 1 --steps 1 --runs 1 --allow-amdsmi-fallback
```

64/64 framework processes execute successfully. Corrected cross-framework aggregation is
24 pass + 8 `precision_mismatch`，所以整体状态是 `complete_with_recorded_limits`，不是
`pass`。所有 8 个 prefill top token 相同，24 个 decode row 中 16 个完整 token-exact；
8 个分叉 row 及其共同前缀记录在 [`summary.json`](summary.json)。24/24 decode row 的
active KV bytes 相同，exact-capacity allocation efficiency 均为 1。

可复核文件：[`raw.jsonl`](raw.jsonl)保存64个独立framework worker输出；
[`matrix-summary.json`](matrix-summary.json)是修正状态逻辑后从同一raw离线重建的完整32-row
汇总；`summary.json`只提取适合版本审查的关键事实。三者都由回归测试交叉检查。

microLLM 使用 BF16 Linear weights、FP32 activations/QK-Norm 和 BF16 Cache；Transformers
使用整网 BF16。二者不是相同 reduction policy，因此分叉首先是需要定位的数值边界，不能在没有
共同 FP32 全-logit 轨迹时归因给任一实现。

第一次调度同时设置两个 visible-device 变量，导致两边都看不到 GPU。该轮是 environment
invalid，不是 32 个模型失败。runner 现在识别双方 device-unavailable 并提前停止，避免继续制造
整张假失败矩阵。

本次每格只有 1 warm-up + 1 measured process。吞吐范围用于确认 measurement path 非零，不能
替代 2+5 或多进程正式性能门。
