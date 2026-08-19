from __future__ import annotations

from importlib import import_module


_CAPI_NAMES = {
    "DType",
    "Device",
    "MicroLLMError",
    "Tensor",
    "add",
    "hip_device_count",
    "matmul",
    "multiply",
    "softmax",
}

__all__ = sorted(_CAPI_NAMES | {"torch_ops"})


def __getattr__(name: str):
    if name in _CAPI_NAMES:
        module = import_module("._capi", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "torch_ops":
        module = import_module(".torch_ops", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
