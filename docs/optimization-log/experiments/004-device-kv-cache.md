# Experiment 004 — preallocated device KV Cache and direct GQA

Status: `keep`

## Observed bottleneck

Cached decoding currently copies K/V to the CPU, concatenates a larger cache, copies it
back, then physically repeats KV heads for GQA. The cache allocation changes every token.

## Hypothesis

Preallocating `[1, kv_heads, max_sequence, head_dim]`, storing each token in place and
mapping `query_head → kv_head` inside cached Attention will improve both generation rows
and eliminate cache payload host roundtrips.

## Scope

- allowed: cached inference-only K/V storage, device store and GQA attention path;
- unchanged: training graph, model parameters, dtype, sampling, output head, allocator
  implementation and benchmark protocol;
- no full K/V head expansion, per-token whole-cache allocation or global synchronization.

## Required gates

- [x] every prefix versus full forward for MHA and GQA
- [x] position zero, last position, overflow and reset/reuse
- [x] cache storage address remains stable
- [x] CPU reference and HIP result agree
- [x] cached HIP graph performs zero payload host transfer
- [x] exact Qwen/DeepSeek greedy tokens
- [x] context-length measurement curve
- [x] full CPU/sanitizer/HIP regressions

## Implementation

- each layer owns K/V backing storage shaped `[1, kv_heads, capacity, head_dim]`;
- the public layer Tensor is a prefix view, so its logical sequence length still grows;
- a device Kernel stores the current K/V row at `position` in place;
- cached score/context Kernels map `query_head / repeats` directly to a KV head;
- no CPU concatenation, full-cache copy, transposed key copy or physical GQA expansion;
- the generator reserves `prompt length + max_new_tokens`, not the model's theoretical
  maximum context;
- ordinary training Attention and sampling remain unchanged.

## Rejected first candidate

The first implementation allocated every model's configured maximum context. It was
correct and faster, but not acceptable:

```text
Qwen inference peak       1.98 GB → 2.78 GB
DeepSeek inference peak   7.11 GB → 14.63 GB
```

The cause was not the in-place algorithm; `Attention` mistakenly passed the model
maximum instead of the request capacity. The retained correction threads the KVCache
capacity through every layer. This failed attempt is kept here because it changed the
design: “preallocated” now means “preallocated to a known request bound.”

## Correctness result

- CPU debug: `151/151` pass;
- ASan/UBSan: `149/149` pass;
- HIP release: `42/42` pass;
- Python/PyTorch operator parity: `4/4` pass;
- CPU tests cover direct cached GQA hand values and invalid contracts;
- model tests cover MHA/GQA every prefix, last position, overflow, reset and reuse;
- MI300X four-step GQA test proves stable K/V addresses and zero Tensor-payload H2D/D2H;
- official Qwen and DeepSeek generated token lists remain exact;
- training final losses/parameter changes are unchanged.

## Performance result

| Workload | Experiment 003 | Experiment 004 | Step speedup | PyTorch ratio | Peak memory ratio |
|---|---:|---:|---:|---:|---:|
| Qwen train | 71.057 token/s | 72.328 token/s | 1.02× | 1.409241 | 0.896769 |
| Qwen generate | 57.322 token/s | 85.645 token/s | 1.49× | 1.220315 | 0.960651 |
| DeepSeek train | 47.913 token/s | 49.474 token/s | 1.03× | 1.886431 | 0.797368 |
| DeepSeek generate | 18.597 token/s | 35.788 token/s | 1.92× | 0.573549 | 0.988733 |

```text
geometric score before  0.885816
geometric score after   1.167931
score improvement       31.8%
```

The selected-matrix score crosses `1.0`, but DeepSeek generation is only `0.574×`
PyTorch. This is why the project reports four bars as well as one score.

## Context curve

Qwen, FP32, prompt length 2, one warm-up and three measured generations:

| New tokens | Decode token/s | Peak engine bytes |
|---:|---:|---:|
| 1 | 53.23 | 1,977,423,884 |
| 32 | 97.51 | 1,978,185,740 |
| 128 | 90.71 | 1,980,545,036 |
| 512 | 68.37 | 1,989,982,220 |

The cache remains bounded and throughput degrades gradually as Attention reads a longer
prefix. The curve is evidence for these four points, not for the full model context.

## Profiler result

Matched Qwen generation (`1` warm-up + `1` measured, four new tokens):

```text
profiled decode             33.60 → 44.04 token/s
hipMemcpy calls              2712 → 600
copyBuffer Kernel calls      2269 → 253
copyBuffer Kernel time      7.591 → 1.414 ms
hipMalloc calls              7393 → 6193
H2D copy records              466 → 370
D2H copy records                9 → 9
```

The remaining D2H records include full-logit CPU sampling, which is deliberately Step
005 rather than hidden inside this experiment.

Raw JSONL, the context curve and before/after compact rocprof tables are in
[004-data](004-data/README.md). Large PFTrace files remain under
`/tmp/microllm-qwen-infer-profile-exp003-before-kv/` and
`/tmp/microllm-qwen-infer-profile-exp004/` on the measurement host.

## Replay commands

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest build/hip-release/benchmarks/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train \
  --output /tmp/experiment-004.jsonl

./scripts/profile_hip.sh /tmp/experiment-004-profile -- \
  build/hip-release/apps/microllm_hf_infer \
  --config /tmp/qwen25-local/config.json \
  --weights /tmp/qwen25-0.5b-model.safetensors \
  --tokens 9707,1879 --device hip --new-tokens 4 \
  --warmup 1 --steps 1 --top-k 10
```

## Results

The hypothesis is supported after rejecting the overallocated first candidate.

## Decision

`keep`. The next single variable is device-side greedy/top-k sampling so complete logits
no longer cross to the CPU every token.
