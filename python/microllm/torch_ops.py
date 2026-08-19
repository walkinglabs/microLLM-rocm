from __future__ import annotations

import os


def load_library(path: str | None = None) -> None:
    import torch

    resolved = path or os.environ.get("MICROLLM_TORCH_OP_LIBRARY")
    if not resolved:
        raise RuntimeError("set MICROLLM_TORCH_OP_LIBRARY or pass the Custom Op library path")
    torch.ops.load_library(resolved)


def add(left, right):
    import torch

    return torch.ops.microllm.add(left, right)


def multiply(left, right):
    import torch

    return torch.ops.microllm.multiply(left, right)
