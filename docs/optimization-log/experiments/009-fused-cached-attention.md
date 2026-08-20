# Experiment 009 — fused cached decode Attention

Status: `keep`

## Observed path

Cached decode currently allocates scores, launches a score Kernel, launches a row
softmax, allocates probabilities, then launches context. Qwen trace attributes roughly
7% Kernel time to these three stages, plus allocator/Event traffic.

## Hypothesis

For a bounded cached prefix, one block per query head can compute score, stable softmax
and context using shared scores. Removing two intermediate Tensors and two launches
should improve decode, especially the still-below-parity DeepSeek generation row.

## Scope

- FP32 cached inference only;
- fused path for sequence `<=4096`, readable three-stage fallback above it;
- direct query-head to KV-head mapping remains unchanged;
- training Attention, sampling, allocator, GEMM and model parameters unchanged;
- no claim for long-context prefill or backward.

## Required gates

- [x] MHA and GQA
- [x] sequence 1/32/128/512 and 4097 fallback boundary
- [x] CPU/HIP output parity
- [x] official prefix/token equality
- [x] no host payload transfer
- [x] three-process fixed inference matrix and interleaved context curve

## Implementation

- one block owns one query head;
- threads compute cached QK scores across positions;
- block reductions compute stable maximum and exponential denominator;
- probabilities stay in dynamic shared memory;
- threads compute output columns directly from shared probabilities and V cache;
- sequence `>4096` uses the old score → softmax → context path;
- two intermediate Tensors and two cached-decode launches disappear per layer;
- CPU reference and training graph remain unchanged.

## Correctness result

- CPU debug: `152/152` pass;
- ASan/UBSan: `150/150` pass;
- HIP release: `54/54` pass;
- Python/PyTorch operator parity: `4/4` pass;
- direct MHA/GQA test covers 1/32/128/512 fused and 4097 fallback;
- focused execution performs zero payload H2D/D2H;
- full cached-prefix model gate and official Qwen/DeepSeek token sequences pass;
- training loss/update is unchanged because training Attention did not change.

## Three-process generation medians

| Model | Baseline samples | Baseline median | Candidate samples | Candidate median | Change |
|---|---|---:|---|---:|---:|
| Qwen | 134.87 / 118.24 / 134.96 | 134.87 | 142.65 / 120.61 / 142.25 | 142.25 | +5.5% |
| DeepSeek | 48.93 / 49.05 / 51.37 | 49.05 | 50.55 / 55.06 / 53.04 | 53.04 | +8.1% |

Training source is unchanged, so score calculation reuses baseline training medians
instead of the later candidate-run timing drift:

```text
Qwen train ratio       2.086361
Qwen generate ratio    2.026893
DeepSeek train ratio   2.622345
DeepSeek generate ratio 0.849978
robust score           1.752183
robust baseline score  1.695566
```

The post-candidate unmodified inference run was 121.80/51.05 token/s, supporting that
DeepSeek's gain is modest and Qwen has large process variance. All raw values remain.

## Interleaved Qwen context curve

Each point runs baseline then candidate, prompt length 2, one warm-up and three measured
generations:

| New tokens | Baseline token/s | Fused token/s | Change | Peak bytes change |
|---:|---:|---:|---:|---:|
| 1 | 79.75 | 73.52 | -7.8% | none |
| 32 | 136.43 | 161.65 | +18.5% | none |
| 128 | 130.64 | 154.75 | +18.5% | none |
| 512 | 88.81 | 140.23 | +57.9% | none |

The one-token regression is a required stable failure: fusion launch/shared setup costs
more than the work saved at that point. The model keeps one implementation rather than
adding an unmeasured one-token branch; future autotuning may revisit it.

## Profiler result

Matched Qwen four-token decode:

```text
profiled decode                 46.89 → 74.16 token/s
old score + softmax + context   3.747 ms
fused cached Attention          2.231 ms
cached Attention Kernel time      -40.5%
all Kernel time                50.290 → 47.941 ms
measured logical allocations     2469 → 2229
```

The remaining 24 softmax calls are from the separate full-forward report, not cached
decode.

Raw repeats, median summary, interleaved curves and compact profiler tables are in
[009-data](009-data/README.md). Large PFTrace remains at
`/tmp/microllm-qwen-infer-profile-exp009/`.

## Replay command

```bash
python3 benchmarks/single_gpu/hf_model_matrix.py \
  --manifest build/hip-release/benchmarks/hf-models.local.json \
  --infer-binary build/hip-release/apps/microllm_hf_infer \
  --train-binary build/hip-release/apps/microllm_hf_train_step \
  --device hip --modes infer,train --output /tmp/experiment-009.jsonl
```

## Results

Supported for cached decode. The one-token result falsifies a universal fusion benefit.

## Decision

`keep`. Prefill/backward remain future stages rather than being claimed complete.
