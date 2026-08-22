# Experiment 114 data

这组数据只保留 Release 构建的正式运行。Debug/未优化二进制的先导结果没有进入仓库，也没有
参与结论。

固定条件：

```text
GPU                 AMD Instinct MI300X VF / gfx942
visible physical GPU 3
models              Qwen2.5-0.5B, DeepSeek-R1-Distill-Qwen-1.5B
requests            8
prompt lengths      256,256,512,512,1024,1024,2048,2048
output lengths      8,8,8,8,16,16,16,16
cache dtype         BF16
warmup              1 complete workload
measured steps      3 complete workloads per process
fresh processes     3 per model/policy
```

运行命令：

```bash
HIP_VISIBLE_DEVICES=3 python3 benchmarks/single_gpu/hf_continuous_matrix.py \
  --manifest /tmp/microllm-bf16-model-manifest.json \
  --binary build-hipblaslt/apps/microllm_hf_infer \
  --output-directory /tmp/microllm-exp114-length-buckets-release \
  --suite length-buckets --warmup 1 --steps 3 --runs 3 \
  --timeout-seconds 1200
```

- `raw.jsonl`：12 个 fresh-process 原始 JSON；
- `summary.json`：四组中位数和两组 policy 对照；
- `gpu3-telemetry.log`：运行前、运行中、进程间和结束后的设备利用率采样；
- `gates.json`：代码、测试和实验验收状态。

Telemetry 首尾均为 GPU 3 `0% busy / 0% VRAM`。采样中的非零值来自本实验自身；各 fresh
process 之间设备会回到空闲状态。
