# Experiment 001 — block-parallel CrossEntropy

Status: `keep`

## Observed bottleneck

Qwen FP32 training baseline rocprof:

```text
cross_entropy_backward_kernel  49.52% kernel time
cross_entropy_kernel           26.21% kernel time
combined                       75.73%
```

Both baseline kernels launch one GPU thread for the complete rows × 151,936 vocabulary
workload.

## Hypothesis

One-block-per-row parallel max/exp-sum plus element-parallel gradient generation will
remove CE as the dominant training hotspot and improve both official train rows.

The hypothesis is weakened if CE Kernel time falls by an order of magnitude while
end-to-end measured train throughput changes by less than normal run variation.

## Scope

- allowed: HIP CE forward/backward implementation, launcher scratch interface, CE tests;
- unchanged: public mathematical API, loss reduction, ignore index, dtype, models,
  optimizer, matmul, allocator and benchmark protocol;
- no new global synchronization;
- no tolerance relaxation.

## Fixed comparison

```text
GPU        MI300X gfx942
dtype      FP32
models     pinned Qwen2.5-0.5B and DeepSeek Distill Qwen 1.5B
warm-up    2
measured   5
PyTorch    committed fixed raw baseline from experiment 000
```

## Correctness gates

- [ ] small hand values and mask
- [ ] rows 1/3/32 and classes 2/32/8192/151936
- [ ] extreme finite logits
- [ ] forward CPU/HIP
- [ ] backward CPU/HIP
- [ ] PyTorch oracle
- [ ] official multi-step loss/parameter trajectory
- [ ] CPU/sanitizer/HIP full regression

## Implementation

- one 256-thread block per logits row;
- parallel maximum and exponential-sum reductions;
- separate parallel final mean reduction;
- backward row statistics plus one thread per logit element;
- FP32 accumulation and unchanged ignore-index semantics;
- device scratch Tensors stay on the same Stream/device.

## Correctness result

- CPU `148/148` pass;
- ASan/UBSan `146/146` pass;
- HIP `40/40` pass;
- Python/PyTorch operator parity pass;
- classes `2/32/8192/151936`, rows `1/3/32` pass;
- forward/backward graph execution performs zero Tensor-payload host transfer;
- official Qwen/DeepSeek generated tokens unchanged;
- multi-step parameter trajectory remains aligned.

## Performance result

| Workload | Before | After | Relative change | Peak memory ratio |
|---|---:|---:|---:|---:|
| Qwen train | 7.300 token/s | 24.027 token/s | 3.29× | 0.950951 |
| Qwen generate | 18.771 token/s | 18.847 token/s | 1.00× | 1.225171 |
| DeepSeek train | 5.794 token/s | 13.295 token/s | 2.29× | 0.797368 |
| DeepSeek generate | 10.018 token/s | 10.053 token/s | 1.00× | 0.988725 |

```text
geometric score before  0.191660
geometric score after   0.318328
score improvement       66.1%
```

## Profiler result

Before, CE forward/backward consumed about 75.73% of Kernel time. After:

```text
cross_entropy_rows                    0.381%
cross_entropy_backward_stats          0.228%
backward elements/factor/finalize     about 0.014%
combined                              about 0.62%
```

rocprof-instrumented measured Qwen step changed from 420.7 ms to 149.1 ms. The next
hotspots are now:

```text
strided copies            33.55%
RMSNorm backward          30.54%
RMSNorm forward           11.14%
AdamW                       6.66%
```

Full evidence is in [001-data](001-data/README.md).

## Commands

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest build/hip-release/benchmarks/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output build/hip-release/benchmarks/experiment-001-microllm.jsonl
```

Before trace: `/tmp/microllm-qwen-train-profile-aa486c8/`.

After trace: `/tmp/microllm-qwen-train-profile-exp001/`.

## Decision

`keep`.

The hypothesis was supported. The experiment also falsified any remaining idea that
AdamW or GEMM should be optimized before transpose copies and RMSNorm for this workload.
