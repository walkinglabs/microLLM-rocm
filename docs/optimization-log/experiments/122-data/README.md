# Experiment 122 data

官方Qwen/DeepSeek FP32、BF16、FP8静态scale last-logit prefill矩阵。

```text
contexts      8,512
policies      FP32, BF16 FFN+Attention, FP8 Linear
runs          3 fresh processes
policy order  Latin rotation
warmup/steps  1/3
FP8 scales    activation 0.025, weight 0.005
oracle        internal FP32 complete vocabulary logits
```

- `raw.jsonl`：正式v2 36条；
- `summary.json`：12 aggregates和4个保留的FP8精度失败；
- `rejected-worker-raw.jsonl`：v1完成的Qwen18条；
- `rejected-worker-preflight.jsonl`：v1环境干净，Deep失败来自status6；
- `gpu2-preflight.jsonl`：v2正式运行前3次0/0。

v1在Deep第一个FP8 worker停止，无summary。v2增加exact-shape native registry和
FP8→BF16 dequant/GEMM软件回退后，从新目录完整重跑。
