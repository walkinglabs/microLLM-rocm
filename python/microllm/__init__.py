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

_PROFILING_NAMES = {"ProfileScope", "export_perfetto", "merge_rocprof_perfetto",
                    "profile", "profile_scope"}
__all__ = sorted(_CAPI_NAMES | _PROFILING_NAMES | {"torch_ops"})


def __getattr__(name: str):
    if name in _CAPI_NAMES:
        module = import_module("._capi", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _PROFILING_NAMES:
        module = import_module(".profiling", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "torch_ops":
        module = import_module(".torch_ops", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
