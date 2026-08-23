# FP8 weight reconstruction audit

Exp143 shows that all-column scaling helps DeepSeek but harms Qwen, so the next scope cannot be
chosen from model-family averages alone. `hf_fp8_weight_audit.py` measures every official Linear
weight before another model policy is implemented.

Hugging Face stores Linear weights as `[output,input]`; microLLM transposes them to
`[input,output]`. The audit therefore computes per-row scales in the source file, which are exactly
the per-output-column scales used by the engine. It selects Q/K/V/O, gate/up/down and an optional
LM head while rejecting embeddings and Norms.

For each Tensor it reconstructs E4M3-FNUZ with one scalar scale and with one output-channel scale,
then records squared error, relative L2, RMS, max error and scale spread. Group summaries combine
squared errors before taking a square root; averaging per-Tensor relative errors would overweight
small matrices.

This is an external PyTorch ROCm diagnostic and is labeled as such. It chooses an Attention/FFN/head
scope but cannot prove microLLM model precision. Every selected scope must return to the native
microLLM complete-logit matrix.

```bash
python3 benchmarks/single_gpu/hf_fp8_weight_audit.py \
  --manifest /path/model-manifest.json \
  --output-directory /tmp/fp8-weight-audit \
  --models qwen2.5-0.5b,deepseek-r1-distill-qwen-1.5b \
  --device cuda:0
```

The CPU contract tests family classification and aggregate math without importing PyTorch, so the
ordinary repository test suite does not gain a mandatory framework dependency.

Exp145 audited 365 official weights. Aggregated column/scalar relative-L2 ratios are 0.99276 for
Qwen and 0.99597 for DeepSeek; the best groups are Qwen Attention (0.98955) and DeepSeek output head
(0.99033). The next minimal native-model counterfactual is output-head-only, not global FFN.
