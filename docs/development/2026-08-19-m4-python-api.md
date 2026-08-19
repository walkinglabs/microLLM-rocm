# 2026-08-19 — M4 dependency-free Python API

## Contract

Offer an optional lightweight Python layer without making Python, NumPy, pybind11,
or PyTorch dependencies of the core engine. Consume only the versioned C ABI and keep
native Tensor ownership deterministic.

## API

- `Tensor.from_f32` and `Tensor.from_i32` with explicit shape/device;
- shape, numel, dtype, device, flat `tolist`, and `to`;
- Python `+`, `*`, and `@` plus named add/multiply/matmul/Softmax;
- HIP device count;
- Python-side shape/device validation and translated engine errors;
- environment or installed-library discovery for `libmicrollm`.

## Verification

Standard-library `unittest` runs without third-party Python packages. CPU and HIP
builds both pass four cases covering hand-valued add/matmul, stable Softmax, int32
round-trip, Python validation, engine error propagation, and optional MI300X HIP
transfer/add. CPU execution takes about 0.06 seconds; HIP execution about 0.30 seconds.

## Boundary

The ctypes API copies Python values through C buffers. It is appropriate for teaching,
control, and smoke integration, not high-throughput Tensor exchange. PyTorch Custom
Ops require a non-owning external TensorView/Stream ABI and a PyTorch development
installation, which are tracked separately.
