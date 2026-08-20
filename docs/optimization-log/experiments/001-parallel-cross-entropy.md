# Experiment 001 — block-parallel CrossEntropy

Status: `running`

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

## Results

Pending implementation and measurement.

## Decision

Pending.
