# FP8 activation fraction pilot runner

The first Exp149 attempt used the general FP32/BF16/FP8 matrix once per fraction. That repeats
references and extends the time window in which an external GPU task can contaminate the pilot.

The archived `hf_fp8_fraction_pilot.py` ran one FP32 oracle for each model/context and four FP8 fractions by
default. For Qwen/DeepSeek at T8/T512 this is 20 workers instead of 48. It always uses the retained
Attention O-projection weight scope and warm-up/steps `0/1`; throughput is explicitly non-evidence.
The retained weight minimum is an explicit runner argument and defaults to 0.005, matching Exp148;
the activation minimum defaults to 0.0001.

Every fraction still compares all 151,936 logits. The selector minimizes worst-case RMS normalized
by the 0.05 gate, requires stable top tokens, and accepts a clipped fraction only when it strictly
beats fraction 1.0. Max error is a deterministic tie-breaker, not a replacement for RMS selection.

The runner verifies that fraction 1 records zero clipped calls and every lower fraction records one
clipped call for every dynamic Tensor call. Strict pre/post GPU gates remain unchanged. A failed run
still exits rather than converting partial rows into a result.

```bash
python3 benchmarks/single_gpu/hf_fp8_fraction_pilot.py \
  --manifest /path/model-manifest.json \
  --binary build/apps/microllm_hf_infer \
  --output-directory /tmp/fraction-pilot \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --contexts 8,512 \
  --fractions 1,0.75,0.5,0.25 \
  --fp8-weight-scale 0.005
```

The CPU contract tests fraction parsing, retained scope construction, fixed numerical workload and
the no-improvement fallback to 1.0 without importing PyTorch.

Exp150 exposed an initial hardcoded 0.0001 weight minimum, while the retained Exp148 policy uses
0.005. The completed pilot was invalidated, the scale became an explicit/defaulted argument, and
the contract now asserts 0.005. See
[Experiment 150](../optimization-log/experiments/150-fp8-fraction-pilot-workload-invalid.md).

After Exp151/152 closed every tested fraction below 1, the specialized runner and its registered test
were removed. The experiment copies remain reproducible from their archived revision and commands.
