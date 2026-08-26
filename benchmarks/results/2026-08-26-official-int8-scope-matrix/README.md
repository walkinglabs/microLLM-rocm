# Official Qwen INT8 mixed-precision scope matrix

Output-column INT8 is isolated to all FFN or all Attention Linear weights. Complete 151,936
logits and two generated tokens are compared with the same FP32 process contract. The fixed gate
requires Max <= 0.1 and RMS <= 0.02 in addition to exact tokens.

FFN fails broadly. Attention recovers tokens and speed but misses both logit limits. Neither scope
is accepted. Because Attention is close, one final decomposition into QKV versus O projection is
allowed; FFN is closed.
