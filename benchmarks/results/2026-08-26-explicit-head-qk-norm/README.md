# Explicit head dimension and QK-Norm evidence

One synthetic decoder block deliberately uses hidden size 8, two query heads and head width 6,
so query width 12 cannot accidentally equal the residual width. Q/K-Norm weights are shared over
the last head dimension. The existing alignment harness compares the same parameters with an
independent PyTorch formula, including logits, loss, every parameter gradient and timing records.

This proves the framework core and mapping seam. It is not an official Qwen3 checkpoint claim;
Hugging Face parser/config compatibility is the next separate node.
