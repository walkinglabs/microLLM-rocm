# N6 — 时间花在哪里，以及怎样只优化一个热点

## 运行前预测

For a HIP operation, distinguish:

```text
kernel Event time
synchronized operator wall time
complete train/generate wall time
setup/library initialization
host-device copies
allocator activity
```

Predict why timing only the host function call around an asynchronous kernel is
incorrect.

## Micro benchmark

```bash
cmake --preset hip-release -S "$MICROLLM_ENGINE_DIR"
cmake --build "$MICROLLM_ENGINE_DIR/build/hip-release" --parallel
"$MICROLLM_ENGINE_DIR/build/hip-release/benchmarks/microllm_bench_ops" \
  --op matmul --m 1 --k 384 --n 384 --size 384 \
  --warmup 10 --repetitions 50 --device hip --implementation readable
```

Every JSON record includes Event and wall min/mean/max, correctness error, GPU/ROCm
metadata, warm-up and repetitions. The validator in
`$MICROLLM_ENGINE_DIR/benchmarks/validate_json.py` parses emitted JSON;
successful calculation alone is not sufficient evidence.

## End-to-end benchmark

```bash
"$MICROLLM_ENGINE_DIR/build/hip-release/benchmarks/microllm_bench_model" \
  --mode train --model tiny --device hip \
  --steps 10 --warmup 3 --batch 1 --context 8 --new-tokens 8
```

The tiny workload is a stable failure:

| mode | CPU | readable HIP |
|---|---:|---:|
| train | 11,991 tokens/s | 1,017 tokens/s |
| generate | 15,655 tokens/s | 841 tokens/s |

The GPU is slower because this path launches many tiny kernels and repeatedly crosses
the host boundary. “GPU available” does not imply “GPU faster.”

## rocprofv3

```bash
"$MICROLLM_ENGINE_DIR/scripts/profile_hip.sh" /tmp/microllm-trace -- \
  "$MICROLLM_ENGINE_DIR/build/hip-release/benchmarks/microllm_bench_model" \
  --mode train --model tiny --device hip \
  --steps 2 --warmup 1 --batch 1 --context 8 --new-tokens 8
```

Observed trace:

- 756 hipMemcpy calls: 67.25% of HIP API duration;
- 792 malloc/free pairs;
- 480 kernel launches;
- copy dispatches: 60.79% of kernel-domain duration;
- matmul: 16.20%.

This falsifies “matmul is the only bottleneck.”

## hipBLASLt experiment

The readable path stays as oracle. hipBLASLt is an optional candidate:

| shape | readable | hipBLASLt |
|---|---:|---:|
| 64×64×64 | 0.05794 ms | 0.06142 ms |
| M=1,K=384,N=384 | 0.24491 ms | 0.05496 ms |

The first Auto rule looked at the smallest dimension and rejected Model-S projection
because M=1. Measuring the real shape corrected the rule.

Model-S measured generation improves from 55.86 readable-HIP to 187.10 Auto tokens/s,
but setup-inclusive throughput drops from 9.13 to 2.39 tokens/s because library
initialization dominates five tokens. Both facts are required in the report.

## 可反驳的下一步

The current main hypothesis is host traffic/launch count. Preallocate device KV cache
without changing matmul; if end-to-end speed does not improve, the hypothesis loses
support. N7 applies the same evidence discipline to RCCL.
