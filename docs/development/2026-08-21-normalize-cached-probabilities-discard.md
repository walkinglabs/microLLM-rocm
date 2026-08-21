# Cached probability normalization rejection

Experiment 091 normalizes each shared cached-Attention score once before the Value phase. DeepSeek
T2048 B1/B8 complete logits are bit exact, but alternating Release performance is 0.994x/0.997x.
The extra shared pass and barrier provide no measurable benefit, so the candidate is fully reverted.

See [Experiment 091](../optimization-log/experiments/091-normalize-cached-probabilities-discard.md).
