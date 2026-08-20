# Experiment 040 — persistent BF16 training weight mirrors

Status: `keep` as an optional speed/memory trade-off

## The problem in plain language

The trainable weight is FP32 because tiny updates must not disappear. BF16 matrix
multiplication needs a rounded BF16 copy. Experiment 037 rebuilt that copy during every
forward pass. It was like photocopying the same textbook before every question.

This candidate keeps one BF16 photocopy beside each FP32 Linear weight. AdamW changes the
FP32 master and refreshes its BF16 copy in the same HIP Kernel launch. The next forward can
use the copy immediately.

```text
forward:   activation × persistent BF16 weight
backward:  FP32 gradients for the FP32 master
AdamW:     update FP32 master + write BF16 mirror in one launch
checkpoint: save FP32 master/moments only; rebuild the derived mirror after restore
```

## Contract and tests

- Mirrors exist only for Linear weights; Norm, bias, gradient and AdamW moments stay FP32.
- A mirror has the same shape/device as its master and is contiguous BF16.
- The autograd edge still points to the FP32 master, not to the derived mirror.
- AdamW performs no host payload transfer and updates master/mirror together.
- `load_state` deliberately refreshes every mirror, so a checkpoint cannot leave stale
  forward weights.
- Model loading after mirror preparation is rejected; weights must be loaded first.

The complete HIP/CPU build passed `241/241`, the CPU sanitizer build passed `168/168`, and
the PyTorch-enabled CPU build passed `173/173`, including Torch operator/graph parity.
An additional MI300X smoke ran the old no-mirror BF16 CLI path and confirmed zero mirror
tensors, a finite loss and a changed FP32 parameter.

## Official MI300X result

Baseline is the retained Experiment 037 BF16 FP32-master training path. Each candidate
number is the median of three independent processes, with two warm-up and five measured
steps.

| Model | Candidate | vs old BF16 | vs micro FP32 | vs PyTorch BF16 autocast | Peak vs old BF16 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 151.69 tok/s | 1.094× | 1.005× | 3.415× | 1.079× |
| DeepSeek Distill 1.5B | 78.41 tok/s | 1.059× | 0.960× | 2.734× | 1.108× |

![Persistent BF16 training mirrors](../assets/bf16-training-mirrors.svg)

Qwen stores 168 mirrors using 715,653,120 bytes. DeepSeek stores 197 mirrors using
3,087,138,816 bytes. Every measured process changed the observed FP32 parameter and had a
lower final loss. Measured optimizer host-to-device and device-to-host calls are both zero.

## Decision

Keep the API and fused HIP Kernel because both models improve by more than 5%, Qwen reaches
its microLLM FP32 throughput, and correctness/restore gates pass. Keep the CLI switch
because the extra persistent memory is real: peak engine memory rises 7.9% and 10.8%.

This is not a memory optimization and DeepSeek still trails microLLM FP32 by about 4.0%.
The next BF16 training experiment should keep these mirrors fixed and remove activation
casts across a continuous FFN island. If that candidate cannot improve both models without
another large persistent allocation, it must be discarded.
