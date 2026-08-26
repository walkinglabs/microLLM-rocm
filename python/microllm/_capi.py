from __future__ import annotations

import ctypes
import ctypes.util
import enum
import os
from pathlib import Path
from typing import Iterable, Sequence


class MicroLLMError(RuntimeError):
    pass


class DType(enum.IntEnum):
    FLOAT32 = 0
    INT32 = 1
    FLOAT16 = 2
    BFLOAT16 = 3


class Device(enum.IntEnum):
    CPU = 0
    HIP = 1


class _TensorHandle(ctypes.Structure):
    pass


_TensorPointer = ctypes.POINTER(_TensorHandle)


class _EventHandle(ctypes.Structure):
    pass


_EventPointer = ctypes.POINTER(_EventHandle)


class _StreamHandle(ctypes.Structure):
    pass


_StreamPointer = ctypes.POINTER(_StreamHandle)
_CUintptr = ctypes.c_size_t


def _library_path() -> str:
    configured = os.environ.get("MICROLLM_LIBRARY")
    if configured:
        return configured
    discovered = ctypes.util.find_library("microllm")
    if discovered:
        return discovered
    candidates = [
        Path(__file__).resolve().parent / "libmicrollm.so",
        Path(__file__).resolve().parent.parent / "lib" / "libmicrollm.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise MicroLLMError(
        "libmicrollm was not found; set MICROLLM_LIBRARY to the built shared library"
    )


_lib = ctypes.CDLL(_library_path())
_lib.ml_capi_version.restype = ctypes.c_uint32
_lib.ml_engine_version.restype = ctypes.c_char_p
_lib.ml_last_error.restype = ctypes.c_char_p
_lib.ml_hip_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
_lib.ml_hip_device_count.restype = ctypes.c_int
_lib.ml_tensor_from_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_TensorPointer),
]
_lib.ml_tensor_from_f32.restype = ctypes.c_int
_lib.ml_tensor_from_i32.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_TensorPointer),
]
_lib.ml_tensor_from_i32.restype = ctypes.c_int
_lib.ml_tensor_from_external.argtypes = [
    _CUintptr,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
    ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(_TensorPointer),
]
_lib.ml_tensor_from_external.restype = ctypes.c_int
_lib.ml_tensor_destroy.argtypes = [_TensorPointer]
_lib.ml_tensor_destroy.restype = None
_lib.ml_tensor_rank.argtypes = [_TensorPointer, ctypes.POINTER(ctypes.c_size_t)]
_lib.ml_tensor_rank.restype = ctypes.c_int
_lib.ml_tensor_shape.argtypes = [_TensorPointer, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64)]
_lib.ml_tensor_shape.restype = ctypes.c_int
_lib.ml_tensor_numel.argtypes = [_TensorPointer, ctypes.POINTER(ctypes.c_int64)]
_lib.ml_tensor_numel.restype = ctypes.c_int
_lib.ml_tensor_dtype.argtypes = [_TensorPointer, ctypes.POINTER(ctypes.c_int)]
_lib.ml_tensor_dtype.restype = ctypes.c_int
_lib.ml_tensor_device.argtypes = [
    _TensorPointer,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.ml_tensor_device.restype = ctypes.c_int
_lib.ml_tensor_is_owning.argtypes = [_TensorPointer, ctypes.POINTER(ctypes.c_int)]
_lib.ml_tensor_is_owning.restype = ctypes.c_int
_lib.ml_tensor_data_ptr.argtypes = [_TensorPointer, ctypes.POINTER(ctypes.c_size_t)]
_lib.ml_tensor_data_ptr.restype = ctypes.c_int
_lib.ml_tensor_storage_bytes.argtypes = [
    _TensorPointer, ctypes.POINTER(ctypes.c_size_t)]
_lib.ml_tensor_storage_bytes.restype = ctypes.c_int
_lib.ml_tensor_copy_f32.argtypes = [_TensorPointer, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
_lib.ml_tensor_copy_f32.restype = ctypes.c_int
_lib.ml_tensor_copy_i32.argtypes = [_TensorPointer, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
_lib.ml_tensor_copy_i32.restype = ctypes.c_int
_lib.ml_tensor_to.argtypes = [
    _TensorPointer,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_TensorPointer),
]
_lib.ml_tensor_to.restype = ctypes.c_int
_lib.ml_event_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.POINTER(_EventPointer)]
_lib.ml_event_create.restype = ctypes.c_int
_lib.ml_event_destroy.argtypes = [_EventPointer]
_lib.ml_event_destroy.restype = None
_lib.ml_event_record_default_stream.argtypes = [_EventPointer]
_lib.ml_event_record_default_stream.restype = ctypes.c_int
_lib.ml_event_ready.argtypes = [_EventPointer, ctypes.POINTER(ctypes.c_int)]
_lib.ml_event_ready.restype = ctypes.c_int
_lib.ml_event_synchronize.argtypes = [_EventPointer]
_lib.ml_event_synchronize.restype = ctypes.c_int
_lib.ml_event_elapsed_ms.argtypes = [_EventPointer, _EventPointer,
                                     ctypes.POINTER(ctypes.c_float)]
_lib.ml_event_elapsed_ms.restype = ctypes.c_int
_lib.ml_stream_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(_StreamPointer)]
_lib.ml_stream_create.restype = ctypes.c_int
_lib.ml_stream_from_external.argtypes = [ctypes.c_int, ctypes.c_int, _CUintptr,
                                         ctypes.POINTER(_StreamPointer)]
_lib.ml_stream_from_external.restype = ctypes.c_int
_lib.ml_stream_destroy.argtypes = [_StreamPointer]
_lib.ml_stream_destroy.restype = None
_lib.ml_stream_synchronize.argtypes = [_StreamPointer]
_lib.ml_stream_synchronize.restype = ctypes.c_int
_lib.ml_stream_native_handle.argtypes = [_StreamPointer,
                                         ctypes.POINTER(ctypes.c_size_t)]
_lib.ml_stream_native_handle.restype = ctypes.c_int
_lib.ml_stream_is_owning.argtypes = [_StreamPointer, ctypes.POINTER(ctypes.c_int)]
_lib.ml_stream_is_owning.restype = ctypes.c_int
_lib.ml_event_record.argtypes = [_EventPointer, _StreamPointer]
_lib.ml_event_record.restype = ctypes.c_int
_lib.ml_event_wait.argtypes = [_EventPointer, _StreamPointer]
_lib.ml_event_wait.restype = ctypes.c_int
for _name in ("ml_add", "ml_multiply", "ml_matmul"):
    _function = getattr(_lib, _name)
    _function.argtypes = [_TensorPointer, _TensorPointer, ctypes.POINTER(_TensorPointer)]
    _function.restype = ctypes.c_int
_lib.ml_softmax.argtypes = [_TensorPointer, ctypes.POINTER(_TensorPointer)]
_lib.ml_softmax.restype = ctypes.c_int
for _name in ("ml_add_on_stream", "ml_multiply_on_stream",
              "ml_matmul_on_stream"):
    _function = getattr(_lib, _name)
    _function.argtypes = [_TensorPointer, _TensorPointer, _StreamPointer,
                          ctypes.POINTER(_TensorPointer)]
    _function.restype = ctypes.c_int
_lib.ml_softmax_on_stream.argtypes = [
    _TensorPointer, _StreamPointer, ctypes.POINTER(_TensorPointer)]
_lib.ml_softmax_on_stream.restype = ctypes.c_int
for _name in ("ml_multiply_out_on_stream", "ml_matmul_out_on_stream"):
    _function = getattr(_lib, _name)
    _function.argtypes = [_TensorPointer, _TensorPointer, _TensorPointer,
                          _StreamPointer]
    _function.restype = ctypes.c_int
_lib.ml_add_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _StreamPointer]
_lib.ml_add_out_on_stream.restype = ctypes.c_int
_lib.ml_softmax_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _StreamPointer]
_lib.ml_softmax_out_on_stream.restype = ctypes.c_int
for _name in ("ml_rms_norm_out_on_stream", "ml_rms_norm_bf16_out_on_stream"):
    _function = getattr(_lib, _name)
    _function.argtypes = [_TensorPointer, _TensorPointer, _TensorPointer,
                          ctypes.c_float, _StreamPointer]
    _function.restype = ctypes.c_int
_lib.ml_swiglu_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _StreamPointer]
_lib.ml_swiglu_out_on_stream.restype = ctypes.c_int
_lib.ml_causal_gqa_attention_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _TensorPointer,
    _TensorPointer, _TensorPointer, _TensorPointer,
    ctypes.c_int64, ctypes.c_float, _StreamPointer]
_lib.ml_causal_gqa_attention_out_on_stream.restype = ctypes.c_int
_lib.ml_embedding_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _StreamPointer]
_lib.ml_embedding_out_on_stream.restype = ctypes.c_int
_lib.ml_rope_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float, _StreamPointer]
_lib.ml_rope_out_on_stream.restype = ctypes.c_int
_lib.ml_cross_entropy_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _TensorPointer,
    _StreamPointer]
_lib.ml_cross_entropy_out_on_stream.restype = ctypes.c_int
_lib.ml_embedding_backward_add_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _StreamPointer]
_lib.ml_embedding_backward_add_on_stream.restype = ctypes.c_int
_lib.ml_softmax_backward_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _StreamPointer]
_lib.ml_softmax_backward_out_on_stream.restype = ctypes.c_int
_lib.ml_rms_norm_backward_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _TensorPointer,
    _TensorPointer, _TensorPointer, ctypes.c_float, _StreamPointer]
_lib.ml_rms_norm_backward_out_on_stream.restype = ctypes.c_int
_lib.ml_swiglu_backward_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _TensorPointer,
    _TensorPointer, _StreamPointer]
_lib.ml_swiglu_backward_out_on_stream.restype = ctypes.c_int
_lib.ml_rope_backward_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_float, _StreamPointer]
_lib.ml_rope_backward_out_on_stream.restype = ctypes.c_int
_lib.ml_cross_entropy_backward_out_on_stream.argtypes = [
    _TensorPointer, _TensorPointer, _TensorPointer, _TensorPointer,
    _TensorPointer, _TensorPointer, _StreamPointer]
_lib.ml_cross_entropy_backward_out_on_stream.restype = ctypes.c_int


def _check(status: int) -> None:
    if status == 0:
        return
    message = _lib.ml_last_error()
    decoded = message.decode("utf-8", errors="replace") if message else "unknown engine error"
    raise MicroLLMError(f"microLLM status {status}: {decoded}")


def _parse_device(device: str | Device | tuple[Device, int]) -> tuple[Device, int]:
    if isinstance(device, tuple):
        return Device(device[0]), int(device[1])
    if isinstance(device, Device):
        return device, 0
    kind, separator, index = device.partition(":")
    parsed = Device.CPU if kind.lower() == "cpu" else Device.HIP if kind.lower() == "hip" else None
    if parsed is None:
        raise ValueError(f"unknown device {device!r}")
    return parsed, int(index) if separator else 0


class Tensor:
    def __init__(self, handle: _TensorPointer, owner=None):
        if not handle:
            raise MicroLLMError("cannot construct Tensor from a null handle")
        self._handle = handle
        self._owner = owner

    @classmethod
    def from_f32(
        cls,
        values: Iterable[float],
        shape: Sequence[int],
        device: str | Device | tuple[Device, int] = "cpu",
    ) -> "Tensor":
        flat = [float(value) for value in values]
        dimensions = [int(value) for value in shape]
        expected = 1
        for dimension in dimensions:
            if dimension < 0:
                raise ValueError("shape dimensions must be non-negative")
            expected *= dimension
        if expected != len(flat):
            raise ValueError(f"shape expects {expected} values, received {len(flat)}")
        values_array = (ctypes.c_float * len(flat))(*flat)
        shape_array = (ctypes.c_int64 * len(dimensions))(*dimensions)
        kind, index = _parse_device(device)
        output = _TensorPointer()
        _check(
            _lib.ml_tensor_from_f32(
                values_array, shape_array, len(dimensions), int(kind), index, ctypes.byref(output)
            )
        )
        return cls(output)

    @classmethod
    def from_i32(
        cls,
        values: Iterable[int],
        shape: Sequence[int],
        device: str | Device | tuple[Device, int] = "cpu",
    ) -> "Tensor":
        flat = [int(value) for value in values]
        dimensions = [int(value) for value in shape]
        expected = 1
        for dimension in dimensions:
            if dimension < 0:
                raise ValueError("shape dimensions must be non-negative")
            expected *= dimension
        if expected != len(flat):
            raise ValueError(f"shape expects {expected} values, received {len(flat)}")
        values_array = (ctypes.c_int32 * len(flat))(*flat)
        shape_array = (ctypes.c_int64 * len(dimensions))(*dimensions)
        kind, index = _parse_device(device)
        output = _TensorPointer()
        _check(
            _lib.ml_tensor_from_i32(
                values_array, shape_array, len(dimensions), int(kind), index, ctypes.byref(output)
            )
        )
        return cls(output)

    @classmethod
    def from_external(
        cls,
        data_ptr: int,
        storage_bytes: int,
        shape: Sequence[int],
        strides: Sequence[int],
        *,
        dtype: DType = DType.FLOAT32,
        device: str | Device | tuple[Device, int] = "cpu",
        owner=None,
    ) -> "Tensor":
        pointer = int(data_ptr)
        bytes_value = int(storage_bytes)
        dimensions = [int(value) for value in shape]
        stride_values = [int(value) for value in strides]
        if pointer <= 0 or bytes_value <= 0:
            raise ValueError("external Tensor pointer and storage bytes must be positive")
        if len(dimensions) != len(stride_values):
            raise ValueError("external Tensor shape/stride rank mismatch")
        shape_array = (ctypes.c_int64 * len(dimensions))(*dimensions)
        stride_array = (ctypes.c_int64 * len(stride_values))(*stride_values)
        kind, index = _parse_device(device)
        output = _TensorPointer()
        _check(_lib.ml_tensor_from_external(
            pointer, bytes_value, shape_array, stride_array, len(dimensions),
            int(DType(dtype)), int(kind), index, ctypes.byref(output)))
        return cls(output, owner=owner)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.ml_tensor_destroy(self._handle)
            self._handle = _TensorPointer()
            self._owner = None

    def __del__(self) -> None:
        self.close()

    @property
    def shape(self) -> tuple[int, ...]:
        rank = ctypes.c_size_t()
        _check(_lib.ml_tensor_rank(self._handle, ctypes.byref(rank)))
        dimensions = []
        for dim in range(rank.value):
            size = ctypes.c_int64()
            _check(_lib.ml_tensor_shape(self._handle, dim, ctypes.byref(size)))
            dimensions.append(size.value)
        return tuple(dimensions)

    @property
    def numel(self) -> int:
        elements = ctypes.c_int64()
        _check(_lib.ml_tensor_numel(self._handle, ctypes.byref(elements)))
        return elements.value

    @property
    def dtype(self) -> DType:
        value = ctypes.c_int()
        _check(_lib.ml_tensor_dtype(self._handle, ctypes.byref(value)))
        return DType(value.value)

    @property
    def device(self) -> tuple[Device, int]:
        kind = ctypes.c_int()
        index = ctypes.c_int()
        _check(_lib.ml_tensor_device(self._handle, ctypes.byref(kind), ctypes.byref(index)))
        return Device(kind.value), index.value

    @property
    def owning(self) -> bool:
        value = ctypes.c_int()
        _check(_lib.ml_tensor_is_owning(self._handle, ctypes.byref(value)))
        return value.value != 0

    @property
    def data_ptr(self) -> int:
        value = ctypes.c_size_t()
        _check(_lib.ml_tensor_data_ptr(self._handle, ctypes.byref(value)))
        return int(value.value)

    @property
    def storage_bytes(self) -> int:
        value = ctypes.c_size_t()
        _check(_lib.ml_tensor_storage_bytes(self._handle, ctypes.byref(value)))
        return int(value.value)

    def tolist(self) -> list[float] | list[int]:
        if self.dtype in (DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16):
            output = (ctypes.c_float * self.numel)()
            _check(_lib.ml_tensor_copy_f32(self._handle, output, self.numel))
            return list(output)
        output = (ctypes.c_int32 * self.numel)()
        _check(_lib.ml_tensor_copy_i32(self._handle, output, self.numel))
        return list(output)

    def to(self, device: str | Device | tuple[Device, int]) -> "Tensor":
        kind, index = _parse_device(device)
        output = _TensorPointer()
        _check(_lib.ml_tensor_to(self._handle, int(kind), index, ctypes.byref(output)))
        return Tensor(output)

    def __add__(self, other: "Tensor") -> "Tensor":
        return add(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return multiply(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)


class Event:
    def __init__(
        self,
        device: str | Device | tuple[Device, int] = "cpu",
        *,
        enable_timing: bool = True,
    ) -> None:
        kind, index = _parse_device(device)
        output = _EventPointer()
        _check(_lib.ml_event_create(
            int(kind), index, int(enable_timing), ctypes.byref(output)))
        self._handle = output
        self._device = (kind, index)
        self._timing = bool(enable_timing)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.ml_event_destroy(self._handle)
            self._handle = _EventPointer()

    def __del__(self) -> None:
        self.close()

    @property
    def device(self) -> tuple[Device, int]:
        return self._device

    @property
    def timing_enabled(self) -> bool:
        return self._timing

    def record_default_stream(self) -> None:
        _check(_lib.ml_event_record_default_stream(self._handle))

    def record(self, stream: "Stream") -> None:
        _check(_lib.ml_event_record(self._handle, stream._handle))

    def wait(self, stream: "Stream") -> None:
        _check(_lib.ml_event_wait(self._handle, stream._handle))

    def ready(self) -> bool:
        result = ctypes.c_int()
        _check(_lib.ml_event_ready(self._handle, ctypes.byref(result)))
        return result.value != 0

    def synchronize(self) -> None:
        _check(_lib.ml_event_synchronize(self._handle))

    def elapsed_ms_since(self, start: "Event") -> float:
        milliseconds = ctypes.c_float()
        _check(_lib.ml_event_elapsed_ms(
            start._handle, self._handle, ctypes.byref(milliseconds)))
        return float(milliseconds.value)


class Stream:
    def __init__(
        self,
        device: str | Device | tuple[Device, int] = "cpu",
        *,
        non_blocking: bool = True,
    ) -> None:
        kind, index = _parse_device(device)
        output = _StreamPointer()
        _check(_lib.ml_stream_create(
            int(kind), index, int(non_blocking), ctypes.byref(output)))
        self._handle = output
        self._device = (kind, index)
        self._non_blocking = bool(non_blocking)

    @classmethod
    def from_external(
        cls,
        native_handle: int,
        device: str | Device | tuple[Device, int] = "hip:0",
    ) -> "Stream":
        if int(native_handle) <= 0:
            raise ValueError("external native Stream handle must be positive")
        kind, index = _parse_device(device)
        output = _StreamPointer()
        _check(_lib.ml_stream_from_external(
            int(kind), index, int(native_handle), ctypes.byref(output)))
        result = cls.__new__(cls)
        result._handle = output
        result._device = (kind, index)
        result._non_blocking = True
        return result

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.ml_stream_destroy(self._handle)
            self._handle = _StreamPointer()

    def __del__(self) -> None:
        self.close()

    @property
    def device(self) -> tuple[Device, int]:
        return self._device

    @property
    def non_blocking(self) -> bool:
        return self._non_blocking

    @property
    def owning(self) -> bool:
        result = ctypes.c_int()
        _check(_lib.ml_stream_is_owning(self._handle, ctypes.byref(result)))
        return result.value != 0

    @property
    def native_handle(self) -> int:
        result = ctypes.c_size_t()
        _check(_lib.ml_stream_native_handle(self._handle, ctypes.byref(result)))
        return int(result.value)

    def synchronize(self) -> None:
        _check(_lib.ml_stream_synchronize(self._handle))


def _binary(function: ctypes._CFuncPtr, left: Tensor, right: Tensor,
            stream: Stream | None = None) -> Tensor:
    output = _TensorPointer()
    if stream is None:
        _check(function(left._handle, right._handle, ctypes.byref(output)))
    else:
        _check(function(left._handle, right._handle, stream._handle,
                        ctypes.byref(output)))
    return Tensor(output)


def add(left: Tensor, right: Tensor, *, stream: Stream | None = None) -> Tensor:
    return _binary(_lib.ml_add if stream is None else _lib.ml_add_on_stream,
                   left, right, stream)


def multiply(left: Tensor, right: Tensor, *,
             stream: Stream | None = None) -> Tensor:
    return _binary(_lib.ml_multiply if stream is None else
                   _lib.ml_multiply_on_stream, left, right, stream)


def matmul(left: Tensor, right: Tensor, *, stream: Stream | None = None) -> Tensor:
    return _binary(_lib.ml_matmul if stream is None else _lib.ml_matmul_on_stream,
                   left, right, stream)


def softmax(input: Tensor, *, stream: Stream | None = None) -> Tensor:
    output = _TensorPointer()
    if stream is None:
        _check(_lib.ml_softmax(input._handle, ctypes.byref(output)))
    else:
        _check(_lib.ml_softmax_on_stream(
            input._handle, stream._handle, ctypes.byref(output)))
    return Tensor(output)


def multiply_out(output: Tensor, left: Tensor, right: Tensor,
                 *, stream: Stream) -> None:
    _check(_lib.ml_multiply_out_on_stream(
        output._handle, left._handle, right._handle, stream._handle))


def matmul_out(output: Tensor, left: Tensor, right: Tensor,
               *, stream: Stream) -> None:
    _check(_lib.ml_matmul_out_on_stream(
        output._handle, left._handle, right._handle, stream._handle))


def add_out(output: Tensor, left: Tensor, right: Tensor,
            *, stream: Stream) -> None:
    _check(_lib.ml_add_out_on_stream(
        output._handle, left._handle, right._handle, stream._handle))


def softmax_out(output: Tensor, input: Tensor, *, stream: Stream) -> None:
    _check(_lib.ml_softmax_out_on_stream(
        output._handle, input._handle, stream._handle))


def rms_norm_out(output: Tensor, input: Tensor, weight: Tensor, *,
                 epsilon: float = 1.0e-5, stream: Stream) -> None:
    _check(_lib.ml_rms_norm_out_on_stream(
        output._handle, input._handle, weight._handle,
        float(epsilon), stream._handle))


def rms_norm_bf16_out(output: Tensor, input: Tensor, weight: Tensor, *,
                      epsilon: float = 1.0e-5, stream: Stream) -> None:
    _check(_lib.ml_rms_norm_bf16_out_on_stream(
        output._handle, input._handle, weight._handle,
        float(epsilon), stream._handle))


def swiglu_out(output: Tensor, gate: Tensor, up: Tensor, *,
               stream: Stream) -> None:
    _check(_lib.ml_swiglu_out_on_stream(
        output._handle, gate._handle, up._handle, stream._handle))


def causal_gqa_attention_out(
        output: Tensor, scaled_query_workspace: Tensor,
        expanded_kv_workspace: Tensor, probabilities_workspace: Tensor,
        query: Tensor, key: Tensor, value: Tensor, *, repeats: int,
        scale: float, stream: Stream) -> None:
    _check(_lib.ml_causal_gqa_attention_out_on_stream(
        output._handle, scaled_query_workspace._handle,
        expanded_kv_workspace._handle, probabilities_workspace._handle,
        query._handle, key._handle, value._handle, int(repeats),
        float(scale), stream._handle))


def embedding_out(output: Tensor, weight: Tensor, indices: Tensor, *,
                  stream: Stream) -> None:
    _check(_lib.ml_embedding_out_on_stream(
        output._handle, weight._handle, indices._handle, stream._handle))


def rope_out(output: Tensor, input: Tensor, *, sequence_dim: int = 1,
             position_offset: int = 0, base: float = 10000.0,
             stream: Stream) -> None:
    _check(_lib.ml_rope_out_on_stream(
        output._handle, input._handle, int(sequence_dim), int(position_offset),
        float(base), stream._handle))


def cross_entropy_out(output: Tensor, row_workspace: Tensor,
                      logits: Tensor, targets: Tensor, *,
                      stream: Stream) -> None:
    _check(_lib.ml_cross_entropy_out_on_stream(
        output._handle, row_workspace._handle, logits._handle,
        targets._handle, stream._handle))


def embedding_backward_add(weight_gradient: Tensor, gradient: Tensor,
                           indices: Tensor, *, stream: Stream) -> None:
    _check(_lib.ml_embedding_backward_add_on_stream(
        weight_gradient._handle, gradient._handle, indices._handle,
        stream._handle))


def softmax_backward_out(input_gradient: Tensor, output: Tensor,
                         gradient: Tensor, *, stream: Stream) -> None:
    _check(_lib.ml_softmax_backward_out_on_stream(
        input_gradient._handle, output._handle, gradient._handle,
        stream._handle))


def rms_norm_backward_out(
        input_gradient: Tensor, weight_gradient: Tensor,
        row_inverse_rms_workspace: Tensor, input: Tensor, weight: Tensor,
        gradient: Tensor, *, epsilon: float = 1.0e-5,
        stream: Stream) -> None:
    _check(_lib.ml_rms_norm_backward_out_on_stream(
        input_gradient._handle, weight_gradient._handle,
        row_inverse_rms_workspace._handle, input._handle, weight._handle,
        gradient._handle, float(epsilon), stream._handle))


def swiglu_backward_out(gate_gradient: Tensor, up_gradient: Tensor,
                        gate: Tensor, up: Tensor, gradient: Tensor, *,
                        stream: Stream) -> None:
    _check(_lib.ml_swiglu_backward_out_on_stream(
        gate_gradient._handle, up_gradient._handle, gate._handle,
        up._handle, gradient._handle, stream._handle))


def rope_backward_out(input_gradient: Tensor, gradient: Tensor, *,
                      sequence_dim: int = 1, position_offset: int = 0,
                      base: float = 10000.0, stream: Stream) -> None:
    _check(_lib.ml_rope_backward_out_on_stream(
        input_gradient._handle, gradient._handle, int(sequence_dim),
        int(position_offset), float(base), stream._handle))


def cross_entropy_backward_out(
        logits_gradient: Tensor, row_stats_workspace: Tensor,
        factor_workspace: Tensor, logits: Tensor, targets: Tensor,
        loss_gradient: Tensor, *, stream: Stream) -> None:
    _check(_lib.ml_cross_entropy_backward_out_on_stream(
        logits_gradient._handle, row_stats_workspace._handle,
        factor_workspace._handle, logits._handle, targets._handle,
        loss_gradient._handle, stream._handle))


def hip_device_count() -> int:
    count = ctypes.c_int()
    _check(_lib.ml_hip_device_count(ctypes.byref(count)))
    return count.value
