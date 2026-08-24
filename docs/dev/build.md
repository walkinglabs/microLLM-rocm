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
| PyTorch | optional, required only for external oracle/alignment | 2.13.0+cpu in isolated environment |
| safetensors Python package | optional, required for two-way weight format interop | 0.6.2 |

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

## Installable CMake package

Install the complete build into one prefix:

```bash
cmake --install build/cpu-debug --prefix "$PWD/install/microllm"
```

The prefix contains headers, static C++ libraries, the optional versioned C ABI shared
library, command-line programs and:

```text
lib/cmake/microLLM/microLLMConfig.cmake
lib/cmake/microLLM/microLLMConfigVersion.cmake
lib/cmake/microLLM/microLLMTargets.cmake
```

Consumers use `find_package(microLLM CONFIG REQUIRED)` and link namespaced targets such
as `microLLM::core`, `microLLM::model` or `microLLM::inference`. Linking a higher-level
target propagates its public microLLM dependencies. When the C API is built, plain-C
consumers request the `capi` component and link `microLLM::capi`. A package produced by
a HIP build also resolves HIP and hipBLASLt; an RCCL build additionally resolves RCCL.

Think of the Config package as an installed instruction card: the consumer names the
capability it needs, while CMake reads the card and supplies include directories,
libraries, compile requirements, and backend dependencies. A minimal consumer uses:

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference)
target_link_libraries(app PRIVATE microLLM::inference)
```

A minimal C consumer uses:

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS capi)
target_link_libraries(c_app PRIVATE microLLM::capi)
```

The Config file also exposes `microLLM_AVAILABLE_COMPONENTS` and the boolean feature
metadata `microLLM_WITH_HIP`, `microLLM_WITH_HIPBLASLT`, `microLLM_WITH_RCCL`, and
`microLLM_WITH_CAPI`. Prefer testing targets or requested components for linking;
these variables are intended for diagnostics and optional application features.

Point `CMAKE_PREFIX_PATH` at the installation root. As a narrower alternative, set
`microLLM_DIR` to the directory containing `microLLMConfig.cmake`. Neither variable
should point at the microLLM source tree. Pre-1.0 compatibility is limited to the
installed `0.1.x` line.

To use a nonstandard package directory inside the prefix:

```bash
cmake -S . -B build/install \
  -DMICROLLM_INSTALL_CMAKEDIR=share/microLLM/cmake
```

`PackageConfig.InstalledConsumer` is not a source-tree link test. It installs into a
fresh temporary prefix, moves the prefix to prove relocatability, configures a separate
CMake project, checks every expected target, and runs installed-package C++ and C
consumers. A second configure intentionally asks for a nonexistent required component
and must fail. A third configure requests an incompatible pre-1.0 minor version and
must also fail. CPU, HIP, and RCCL presets label and execute the same contract.

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
| `MICROLLM_ENABLE_COVERAGE` | `OFF` | GCC/Clang line and branch instrumentation |
| `MICROLLM_SAFETENSORS_PYTHON` | empty | interpreter with torch/safetensors used by the optional official interop CTest |
| `MICROLLM_INSTALL_CMAKEDIR` | `lib/cmake/microLLM` | package-config destination relative to the install prefix |

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

### `find_package(microLLM)` cannot find the package

First confirm that `microLLMConfig.cmake` exists under the chosen installation prefix.
Then pass either the prefix root:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/prefix
```

or the exact Config directory:

```bash
cmake -S . -B build -DmicroLLM_DIR=/absolute/prefix/lib/cmake/microLLM
```

Do not combine a Config file from one installation with libraries from another.
