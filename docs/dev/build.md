# Build from source

This document defines both the required baseline and the exact environment used for the
current repository evidence. A version listed as “validated” is measured; a lower
version is not silently claimed to work.

## Toolchain requirements

| Component | Required baseline | Validated in current evidence |
|---|---|---|
| Operating system | Linux x86_64 | Ubuntu 24.04.4 LTS |
| CMake | 3.25 or newer | 3.31.10 |
| C++ language | C++20, extensions disabled | GCC/G++ 13.3.0 |
| C compiler | C11-capable host compiler | GCC 13.3.0 |
| Python | 3.9 or newer for optional API/tests | 3.13.13 |
| HIP/ROCm | optional for CPU; required for AMD GPU | HIP 7.13.99004, AMD Clang 23.0 development build |
| hipBLASLt | optional optimized matmul backend | 1.3.0 |
| RCCL | optional multi-GPU backend | 2.28.3 |
| rocprofv3 | optional profiler | 1.3.0 |
| GPU architecture | a ROCm-supported AMD GPU | 4 × gfx942 MI300X virtual functions |
| GoogleTest | system package or fetched by CMake | v1.14.0 FetchContent fallback |

The project does not currently claim support for GCC versions older than the validated
GCC 13.3 toolchain. Other C++20 compilers may work, but require their own recorded CI
evidence before entering the support matrix.

## Check your environment

```bash
cmake --version
gcc --version
g++ --version
python3 --version
```

For HIP builds:

```bash
hipcc --version
rocminfo | grep -E '^  Name: +gfx'
rocprofv3 --version
```

The repository also provides a non-mutating environment report:

```bash
./scripts/collect_system_info.sh
```

## CPU build

Using CMake presets:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug
```

Equivalent explicit configuration:

```bash
cmake -S . -B build/cpu-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMICROLLM_ENABLE_HIP=OFF \
  -DMICROLLM_BUILD_TORCH_OPS=OFF
cmake --build build/cpu-debug --parallel
ctest --test-dir build/cpu-debug --output-on-failure -L cpu
```

## CPU sanitizers

```bash
cmake --preset cpu-sanitize
cmake --build --preset cpu-sanitize --parallel
ctest --preset cpu-sanitize
```

This enables AddressSanitizer and UndefinedBehaviorSanitizer. Dynamic C/Python binding
tests are kept out of the sanitizer preset because a non-instrumented loader must preload
the ASan runtime; those integrations have separate normal-build tests.

## HIP build

Install a ROCm release supported by the target GPU and ensure its CMake packages are
discoverable. When ROCm is installed outside `/opt/rocm`, set `CMAKE_PREFIX_PATH` or the
package-specific directories.

```bash
cmake --preset hip-release
cmake --build --preset hip-release --parallel
ctest --preset hip-release
```

To compile for an explicit target rather than auto-detection:

```bash
cmake --preset hip-release -DMICROLLM_HIP_ARCHITECTURES=gfx942
```

`MICROLLM_ENABLE_HIP=ON` makes a missing HIP compiler a configuration error. `AUTO`
enables HIP only when the toolchain is found; `OFF` guarantees a CPU-only build.

## RCCL build

```bash
cmake --preset rccl-release
cmake --build --preset rccl-release --parallel
ctest --preset rccl-release
```

RCCL tests require at least two visible GPUs. Four-rank execution is not currently a
release claim because the recorded container exposes only 64 MB of shared memory.

## Build options

| CMake option | Default | Purpose |
|---|---:|---|
| `MICROLLM_ENABLE_HIP` | `AUTO` | `AUTO`, `ON`, or `OFF` HIP backend |
| `MICROLLM_HIP_ARCHITECTURES` | empty | explicit targets such as `gfx942` or `gfx1100` |
| `MICROLLM_ENABLE_HIPBLASLT` | `ON` | optional optimized 2D FP32 matmul |
| `MICROLLM_ENABLE_RCCL` | `OFF` | single-node multi-GPU collectives |
| `MICROLLM_BUILD_TESTS` | `ON` | unit/conformance/integration tests |
| `MICROLLM_BUILD_EXAMPLES` | `ON` | runnable examples |
| `MICROLLM_BUILD_BENCHMARKS` | `ON` | micro and end-to-end benchmarks |
| `MICROLLM_BUILD_CAPI` | `ON` | versioned C shared library |
| `MICROLLM_BUILD_PYTHON` | `ON` | ctypes API integration tests |
| `MICROLLM_BUILD_TORCH_OPS` | `AUTO` | optional PyTorch dispatcher library |
| `MICROLLM_ENABLE_SANITIZERS` | `OFF` | host ASan and UBSan |

## Common failures

### HIP requested but no compiler found

Verify `hipcc --version`, then provide the ROCm prefix:

```bash
cmake --preset hip-release -DCMAKE_PREFIX_PATH=/opt/rocm
```

### hipBLASLt or RCCL not found

Check for `hipblaslt-config.cmake` or `rccl-config.cmake` under the ROCm installation.
hipBLASLt is optional; RCCL is required only when `MICROLLM_ENABLE_RCCL=ON`.

### Wrong GPU target

Use `rocminfo` to find the `gfx` name and pass it through
`MICROLLM_HIP_ARCHITECTURES`. Do not copy an architecture value from another machine.

### FetchContent cannot download GoogleTest

Install a system GoogleTest development package or configure once in a network-enabled
environment. Production/offline builds should pin and provide dependencies explicitly.
