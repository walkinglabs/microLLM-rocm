# Caller-owned PyTorch Custom Op Softmax

The schema declares `Tensor(a!)` mutation and returns the exact caller output. Inputs
requiring gradients are rejected because this out variant is inference-only; the
functional Custom Op remains the differentiable API.

Six fresh MI300X processes compare against `torch.softmax(..., out=...)` for FP16/BF16
widths 1/17/128/1024/4096. All 10 precision and pointer-identity rows pass, and both
implementations report zero peak-extra bytes.

- width1024 reaches 1.116× native out for FP16 and 1.087× for BF16;
- width4096 reaches 0.813× for FP16 and 0.467× for BF16;
- the returned Tensor always has the caller's pointer.

The API is retained as a zero-allocation integration surface. The wide rows remain
counterexamples; no universal speed claim is made.
