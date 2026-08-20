# N8 — 统一评价与失败图集

N8 adds no framework feature. It asks whether every claim can be traced to a command,
test, raw record, and scope.

## Evidence table

| Claim | State | Evidence |
|---|---|---|
| CPU Storage/Tensor lifetime and views | pass | core tests + ASan/UBSan |
| CPU/HIP readable forward operators | pass on gfx942 | HIP conformance tests |
| Autograd formulas | pass on CPU | hand gradients + finite differences |
| Tiny HIP training | pass on MI300X | five-step finite trajectory |
| Checkpoint resume | pass | three subsequent exact AdamW steps |
| Tiny Transformer overfit | pass | loss 1.81171 → 0.00673309 |
| Beyond-context generation | stable failure | expected/observed cycle record |
| Cached/full logits | pass CPU MHA/GQA | every prefix within 2e-5 |
| Model-S CPU training | smoke pass | 11.2473 → 1.98712 over 3 steps |
| Model-S HIP logits | pass on gfx942 | max error 4.05312e-06 |
| hipBLASLt selected shape | pass | micro + Model-S e2e JSONL |
| Tiny HIP speed | failure vs CPU | train/generate e2e JSONL |
| Two-rank parameter equivalence | pass | rank diff 0, reference diff 1.49e-08 |
| Bucket/overlap | pass synthetic | raw RCCL JSONL |
| Four-rank execution | failed | `/dev/shm` ENOSPC debug record |
| Python ctypes | pass CPU/HIP | unittest through C ABI |
| PyTorch Custom Ops CPU | pass | Torch 2.13 dispatcher comparison |
| PyTorch Custom Ops ROCm | environment failure | matching wheel Bus error on import |
| Official Qwen2.5-0.5B FP32 | pass on gfx942 | strict 290 tensors, full logits, 4 greedy tokens, one AdamW step |
| DeepSeek-R1-Distill-Qwen-1.5B FP32 | pass on gfx942 | strict 339 tensors, full logits, 8 greedy tokens |
| FP8 small Transformer | pass on gfx942 | scaled GEMM, FP32 master/backward, KV decode |
| Official HF BF16/FP8 end to end | unverified | only FP32 whole-model alignment retained |
| Reference-length pretraining/SFT | unverified | only short immutable-corpus smoke retained |
| Radeon compatibility | unverified | no Radeon run record |

## Re-run evidence gates

CPU:

```bash
"$MICROLLM_ENGINE_DIR/scripts/check_cpu.sh"
```

HIP:

```bash
ctest --test-dir "$MICROLLM_ENGINE_DIR/build/hip-release" -L hip --output-on-failure
```

RCCL:

```bash
ctest --test-dir "$MICROLLM_ENGINE_DIR/build/rccl-release" -L rccl --output-on-failure
```

Benchmark JSON validity:

```bash
python3 "$MICROLLM_ENGINE_DIR/benchmarks/validate_json.py" \
  "$MICROLLM_ENGINE_DIR/build/hip-release/benchmarks/microllm_bench_ops" \
  "$MICROLLM_ENGINE_DIR/build/hip-release/benchmarks/microllm_bench_model"
```

## Failure atlas

### F1: low loss, bad longer generation

Possible explanations: positional memorization or cache divergence. Cached/full tests
weaken the second within trained length. Context-eight and randomized-offset training
can falsify the first.

### F2: tiny GPU slower than CPU

Trace supports copies/launches rather than matmul alone. Device-native AdamW and
preallocated cache are separate rebuttal experiments.

### F3: optimized kernel, startup regression

hipBLASLt improves Model-S measured tokens/s but worsens five-token setup-inclusive
throughput. Longer generation can determine the break-even point.

### F4: four GPUs visible, four-rank RCCL fails

The current explanation is shared-memory capacity. A larger `/dev/shm` environment is
the direct falsification test.

## Publication rule

Only rows marked pass with hardware/config scope may enter release claims. A planned
recipe, compiled source, skipped test, or visible GPU is not a measurement.

This table is a teaching snapshot. Before publishing a new claim, re-run the current
`main` gates and read its
[live evidence status](https://github.com/walkinglabs/microLLM-rocm/blob/main/docs/development/STATUS.md);
do not assume the snapshot is newer than the engine.
