# Step 33 — training residual-add plus RMSNorm

Status: complete, candidate rejected

## Question

Can the existing inference `add_rms_norm` Kernel safely remove one training launch per block
without changing how branched gradients meet?

## Contract

- expose one Autograd operation that returns both the residual sum and normalized Tensor;
- preserve the residual sum as its own graph node;
- compare both outputs and left/right/weight gradients with PyTorch;
- prove the HIP graph performs no payload transfer;
- run Qwen2.5-0.5B and DeepSeek Distill 1.5B at B1/T512 in fresh processes;
- keep the model route only if both throughput medians are at least `1.01×`, peak does not
  increase, loss stays within 0.5%, and the observed parameter is equal.

## Decision

Reject the model route. It reaches `0.9785×/0.9980×`, saves no peak memory, and DeepSeek's
observed parameter differs in the last represented bits. Retain only the independently useful,
fully tested Autograd primitive.
