from __future__ import annotations

import os


_FORMULAS_REGISTERED = False


def _register_formulas() -> None:
    global _FORMULAS_REGISTERED
    if _FORMULAS_REGISTERED:
        return
    import torch

    def add_backward(context, gradient):
        del context
        return gradient, gradient

    def multiply_setup(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def multiply_backward(context, gradient):
        left, right = context.saved_tensors
        return gradient * right, gradient * left

    torch.library.register_autograd("microllm::add", add_backward)
    torch.library.register_autograd(
        "microllm::multiply", multiply_backward,
        setup_context=multiply_setup,
    )
    _FORMULAS_REGISTERED = True


def load_library(path: str | None = None) -> None:
    import torch

    resolved = path or os.environ.get("MICROLLM_TORCH_OP_LIBRARY")
    if not resolved:
        raise RuntimeError("set MICROLLM_TORCH_OP_LIBRARY or pass the Custom Op library path")
    torch.ops.load_library(resolved)
    _register_formulas()


def add(left, right):
    import torch

    return torch.ops.microllm.add(left, right)


def multiply(left, right):
    import torch

    return torch.ops.microllm.multiply(left, right)


def swiglu(gate, up):
    import torch

    return torch.ops.microllm.swiglu(gate, up)
