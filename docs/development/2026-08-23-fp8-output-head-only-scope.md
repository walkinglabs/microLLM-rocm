# FP8 output-head-only column scale scope

Exp145 selects the untied output head as DeepSeek's best and cheapest output-channel group. The new
`Fp8OutputChannelScope::OutputHeadOnly` keeps every block Linear on device Tensor-amax and applies
output-channel scaling only to an independent LM head.

The CLI contract is:

```text
--fp8-weight-scale-mode output-channel-amax
--fp8-output-channel-scope output-head-only
```

Using the scope with another weight mode is rejected. The policy name and JSON include the scope so
an output-head-only run cannot be grouped with the all-Linear experiment.

For an untied tiny model, preparation retains seven scalar scales plus 16 head-column scales and
records exactly one column quantization. For a tied model, no output head exists: all seven block
weights remain scalar and column quantization is zero. Preparation and graph-free inference are
value-identical before/after conversion.

The HIP gate proves one device column vector, seven device scalar scales, no preparation D2H, no
hot-path payload transfer and no repeated weight quantization. This predicts Qwen's tied model will
remain on the scalar baseline while DeepSeek adds one post-scale per forward; official complete
logits must prove the prediction.
