from __future__ import annotations

from importlib import import_module


_CAPI_NAMES = {
    "DType",
    "Device",
    "Event",
    "Stream",
    "MicroLLMError",
    "Tensor",
    "add",
    "hip_device_count",
    "matmul",
    "matmul_out",
    "multiply",
    "multiply_out",
    "softmax",
}

_PROFILING_NAMES = {"HipEventProfileScope", "ProfileScope",
                    "calibrate_python_rocprof_clock", "export_perfetto",
                    "hip_event_profile_scope", "merge_rocprof_perfetto",
                    "profile", "profile_scope", "roctx_available"}
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
