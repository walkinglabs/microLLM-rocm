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
| rocWMMA | optional, benchmark-only matrix-fragment headers | 2.2.0 |
| OpenMP C++ | optional transitive requirement of the rocWMMA CMake target | LLVM OpenMP from ROCm SDK |
| RCCL | optional multi-GPU backend | 2.28.3 |
| rocprofv3 | optional profiler | 1.3.0 |
| GPU architecture | a ROCm-supported AMD GPU | 4 × gfx942 MI300X virtual functions |
| GoogleTest | system package or fetched by CMake | v1.14.0 FetchContent fallback |
| PyTorch | optional, required only for external oracle/alignment/Custom Ops | 2.13.0+cpu and 2.11.0+rocm7.13.0rc2 in isolated environments |
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

When both rocWMMA and its OpenMP C++ dependency are discoverable, CMake also builds
`microllm_bench_rocwmma_qk` and `microllm_bench_rocwmma_online_attention`. These are
research benchmarks, not engine or
installed-SDK dependencies. If either optional package is absent, ordinary HIP operators,
applications and the exported `microLLM::*` targets are unchanged.

## Optional PyTorch ROCm Custom Ops

Keep PyTorch optional and isolated. Point CMake at the interpreter and `TorchConfig.cmake`
from the same virtual environment; mixing a CPU Torch CMake package with a ROCm Python
interpreter can compile successfully and still produce the wrong dispatcher library.

```bash
MICROLLM_TORCH_PYTHON=/path/to/rocm-venv/bin/python
MICROLLM_TORCH_PREFIX="$($MICROLLM_TORCH_PYTHON -c \
  'import torch; print(torch.utils.cmake_prefix_path)')"

cmake -S . -B build/torch-rocm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMICROLLM_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_PREFIX_PATH="$MICROLLM_TORCH_PREFIX" \
  -DPython3_EXECUTABLE="$MICROLLM_TORCH_PYTHON" \
  -DMICROLLM_ENABLE_HIP=ON \
  -DMICROLLM_ENABLE_HIPBLASLT=ON \
  -DMICROLLM_BUILD_TORCH_OPS=ON
cmake --build build/torch-rocm --target microllm_torch_ops --parallel

MICROLLM_TORCH_OP_LIBRARY="$PWD/build/torch-rocm/bindings/torch/libmicrollm_torch_ops.so" \
PYTHONPATH="$PWD/python" \
  "$MICROLLM_TORCH_PYTHON" python/tests/test_torch_ops.py
```

Replace `gfx942` with the target architecture. PyTorch ROCm intentionally presents AMD
GPU tensors through the `CUDA` dispatcher/device API, so the adapter registers the CUDA
dispatch key and obtains the current HIP Stream from `c10::hip`. That naming does not mean
the binary contains a CUDA backend. FP32/FP16/BF16, error contracts, Autograd, Meta/fullgraph
compile and current-Stream execution are covered by the test above.
The same test covers the fused `torch.ops.microllm.swiglu` forward, Meta and Autograd
contracts. Its internal backward schema is an implementation detail of the Python
Autograd registration; call the public `microllm.torch_ops.swiglu` wrapper.
Its FP32 sum-backward path recognizes PyTorch's zero-stride expanded scalar and avoids
materializing it. This is a layout contract inside the wrapper, not a user-facing flag.
SwiGLU Autograd is implemented in C++; Python registration remains only for add/multiply.
GCC's false-positive array-bounds warning inside Torch's custom-function header is disabled
only for the optional adapter target.
FP16/BF16 SwiGLU Autograd calls a typed fused engine backward; it does not build an ATen
formula graph. FP32 retains the general/scalar-seed producers.
The same adapter registers last-dimension Softmax for FP32/FP16/BF16 with CPU, ROCm,
Meta/fullgraph and C++ Autograd. Inputs without gradients bypass `Function::apply`; the
functional output still follows PyTorch allocation semantics.
An explicit `softmax_out` Custom Op accepts caller Storage through a `Tensor(a!)`
mutation schema. It is inference-only; use functional `softmax` when gradients are required.

## RCCL build

```bash
cmake --preset rccl-release
cmake --build --preset rccl-release --parallel
ctest --preset rccl-release
```

RCCL tests require at least two visible GPUs. Four-rank execution is not currently a
release claim because the recorded container exposes only 64 MB of shared memory.

## Installable CMake package

Use a package-focused preset and install it into one explicit prefix. These presets
build the public libraries and CMake package without repository tests, applications,
examples, benchmarks, Python tests, or PyTorch adapters.

CPU-only:

```bash
cmake --preset sdk-cpu
cmake --build --preset sdk-cpu --parallel
cmake --install build/sdk-cpu --prefix "$PWD/install/microllm"
```

HIP:

```bash
cmake --preset sdk-hip -DMICROLLM_HIP_ARCHITECTURES=gfx942
cmake --build --preset sdk-hip --parallel
cmake --install build/sdk-hip --prefix "$PWD/install/microllm"
```

HIP with RCCL:

```bash
cmake --preset sdk-rccl -DMICROLLM_HIP_ARCHITECTURES=gfx942
cmake --build --preset sdk-rccl --parallel
cmake --install build/sdk-rccl --prefix "$PWD/install/microllm"
```

The SDK presets retain all C++ component libraries and the versioned C ABI. Replace
`gfx942` with the architecture of the destination GPU. Do not overlay SDKs produced by
different presets in one prefix: installation does not remove stale files from a prior
backend build.

The prefix contains headers, static C++ libraries, the optional versioned C ABI shared
library, command-line programs when `MICROLLM_BUILD_APPS=ON`, and:

```text
lib/cmake/microLLM/microLLMConfig.cmake
lib/cmake/microLLM/microLLMConfigVersion.cmake
lib/cmake/microLLM/microLLMTargets.cmake
```

Consumers use `find_package(microLLM CONFIG REQUIRED)`. The recommended
`microLLM::microLLM` target carries the complete single-device C++ training and
inference SDK. Advanced consumers can instead link narrower targets such as
`microLLM::core`, `microLLM::model` or `microLLM::inference`; linking a higher-level
target propagates its public microLLM dependencies. When the C API is built, plain-C
consumers request the `capi` component and link `microLLM::capi`. A package produced by
a HIP build also resolves HIP and hipBLASLt; an RCCL build additionally resolves RCCL.

Run the focused package contract after a CPU build:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset package-cpu
```

The four tests cover the configured build tree, a relocated install tree, the standalone
public example, and rejection of a Config destination outside the install prefix. Their
independent consumers also cover the umbrella C++ target, the narrow `core` component,
the versioned C ABI from a truly C-only project, required-component errors, and the
pre-1.0 version-compatibility rule.

A copy-paste-ready standalone consumer lives in
[`examples/package-consumer`](../../examples/package-consumer). The
`PackageConfig.PublicExample` test installs the current SDK, configures that independent
project through `CMAKE_PREFIX_PATH`, compiles it, and runs it. This keeps the public
instructions executable instead of treating them as documentation-only text.

Think of the Config package as an installed instruction card: the consumer names the
capability it needs, while CMake reads the card and supplies include directories,
libraries, compile requirements, and backend dependencies. A minimal consumer uses:

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED)
target_link_libraries(app PRIVATE microLLM::microLLM)
```

To keep the dependency request narrow, name the capability explicitly:

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference)
target_link_libraries(app PRIVATE microLLM::inference)
```

Library components and backend feature components answer different questions. A library
component chooses an imported target. A feature component rejects an SDK that lacks a
required backend capability:

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference hip)
target_link_libraries(gpu_app PRIVATE microLLM::inference)

find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS multi_gpu rccl)
target_link_libraries(distributed_app PRIVATE microLLM::multi_gpu)
```

Available feature-component names are `hip`, `hipblaslt`, `rccl`, and `rocwmma`. They
do not create targets with those names. The corresponding libraries remain available
through targets such as `microLLM::ops` and `microLLM::multi_gpu`.

A minimal C consumer uses:

```cmake
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS capi)
target_link_libraries(c_app PRIVATE microLLM::capi)
```

The C API contract is tested from a separate `project(... LANGUAGES C)` consumer. It
does not initialize a C++ compiler; the imported shared-library target carries the C
header, library location, and runtime link information required by the C application.
The same stable header exposes opaque `ml_event*` and owned `ml_stream*` lifecycles,
default/explicit-Stream Event record/query/wait, Event-only synchronization, elapsed
device time, Stream versions of all four C ABI operators, and caller-owned
multiply/matmul outputs. These additive functions retain the v1 ABI. The API accepts
a non-owning native Stream from another framework through
`ml_stream_from_external`; destroying the wrapper never destroys that handle. The caller
must preserve native Stream lifetime. `ml_tensor_from_external` accepts a non-owning
pointer/byte/shape/stride/dtype/device
descriptor. Destroying it never frees caller memory. The Python wrapper should receive
`owner=` to keep the external allocation alive. The first strict zero-copy output path is
FP32 contiguous `ml_add_out_on_stream`; unsupported layouts fail instead of copying.
External descriptors also map FP16/BF16, with caller-owned multiply/matmul paths tested
against PyTorch ROCm. FP32 Softmax/RMSNorm, BF16 RMSNorm output and F32/F16/BF16
SwiGLU have strict caller-owned Stream APIs. HIP low-precision Softmax is rejected until
a typed reduction Kernel exists; no temporary FP32 copy is inserted. FP32 causal MHA/GQA
also exposes caller-owned output plus scaled-Q, expanded-K/V and
probability workspaces. Writable tensors must not alias each other or any input.
RoPE, Embedding and CrossEntropy now have caller-owned outputs; loss additionally requires
a caller-owned `[rows,2]` reduction workspace.
Softmax/RMSNorm/SwiGLU/RoPE/CrossEntropy backward and Embedding gradient accumulation
now have caller-owned APIs with explicit scratch. Full Autograd leaf-pool binding remains
outside the ABI.

During local development, installation is optional. After configuring and building
microLLM, point the consumer directly at that configured build tree:

```bash
cmake -S /path/to/consumer -B /path/to/consumer/build \
  -DmicroLLM_DIR=/absolute/path/to/microLLM-rocm/build/cpu-debug
cmake --build /path/to/consumer/build
```

This mode references artifacts inside that checkout and is not a deployable SDK. Use
`cmake --install` and `CMAKE_PREFIX_PATH` when the result must be movable or shared.
Neither package mode propagates the repository's warning or instrumentation compile
flags. An instrumented static build carries only the runtime link option required by
its object files; ordinary builds do not. Public requirements such as C++20 and backend
dependencies are propagated normally.

The Config file also exposes `microLLM_VERSION`, `microLLM_BACKEND` (`CPU`, `HIP`, or
`HIP_RCCL`), `microLLM_CXX_STANDARD`,
`microLLM_HIP_ARCHITECTURES`, `microLLM_AVAILABLE_COMPONENTS`,
`microLLM_AVAILABLE_FEATURE_COMPONENTS`, and the boolean feature
metadata `microLLM_WITH_HIP`, `microLLM_WITH_HIPBLASLT`,
`microLLM_WITH_ROCWMMA`, `microLLM_WITH_RCCL`, and `microLLM_WITH_CAPI`, plus
`microLLM_WITH_SANITIZERS` and
`microLLM_WITH_COVERAGE`. `microLLM_DEFAULT_TARGET` names the recommended umbrella
target and `microLLM_TARGETS` lists every imported target in the selected SDK. Prefer
testing targets or requested components for linking;
these variables are intended for diagnostics and optional application features. A CPU
package reports an empty `microLLM_HIP_ARCHITECTURES` value.

Point `CMAKE_PREFIX_PATH` at the installation root. As a narrower alternative, set
`microLLM_ROOT` to one installation root, or set `microLLM_DIR` to either the installed
Config directory or a configured microLLM build directory. `microLLM_ROOT` is a prefix;
`microLLM_DIR` directly contains `microLLMConfig.cmake`. Neither variable should point
at the unbuilt source tree. Pre-1.0
compatibility is limited to the installed `0.1.x` line.

To use a nonstandard package directory inside the prefix:

```bash
cmake -S . -B build/install \
  -DMICROLLM_INSTALL_CMAKEDIR=share/microLLM/cmake
```

`PackageConfig.BuildTreeConsumer` configures, compiles, links, and runs independent C++,
core-component-only, mixed-language, and genuinely C-only projects against the generated
build-tree Config.
`PackageConfig.InstalledConsumer`
installs into a fresh temporary prefix, moves the prefix to prove relocatability, then
does the same against the installed SDK. Both check every expected target, reject
repository-only compile options, and require ordinary builds to add no link options.
An instrumented build may carry exactly its required runtime link option. Both gates
also ask for a nonexistent component and an incompatible pre-1.0 minor version; both
requests must fail. CPU, HIP, and RCCL presets label and execute the same contracts.
`PackageConfig.PublicExample` independently compiles and runs the short example linked
above against a fresh installation discovered through `microLLM_ROOT`.
`PackageConfig.RejectsNonRelocatableDestination`
also proves that an absolute package destination is rejected instead of silently
producing an SDK that cannot be moved.

To diagnose discovery without printing every CMake lookup, use:

```bash
cmake -S . -B build --fresh \
  -DmicroLLM_ROOT=/absolute/path/to/install/microllm \
  --debug-find-pkg=microLLM
```

The selected path should end in `microLLMConfig.cmake`. When switching SDK prefixes,
use `--fresh` or remove the consumer build directory so an older `microLLM_DIR` cached in
`CMakeCache.txt` cannot win the search.

## Build options

| CMake option | Default | Purpose |
|---|---:|---|
| `MICROLLM_ENABLE_HIP` | `AUTO` | `AUTO`, `ON`, or `OFF` HIP backend |
| `MICROLLM_HIP_ARCHITECTURES` | empty | explicit targets such as `gfx942` or `gfx1100` |
| `MICROLLM_ENABLE_HIPBLASLT` | `ON` | optional optimized 2D FP32 matmul |
| `MICROLLM_ENABLE_ROCWMMA` | `ON` | optional gfx942 online-Attention operator and research benchmarks |
| `MICROLLM_ENABLE_RCCL` | `OFF` | single-node multi-GPU collectives |
| `MICROLLM_BUILD_TESTS` | `ON` | unit/conformance/integration tests |
| `MICROLLM_BUILD_APPS` | `ON` | training, inspection, inference and profiling command-line applications |
| `MICROLLM_BUILD_EXAMPLES` | `ON` | runnable examples |
| `MICROLLM_BUILD_BENCHMARKS` | `ON` | micro and end-to-end benchmarks |
| `MICROLLM_BUILD_CAPI` | `ON` | versioned C shared library |
| `MICROLLM_BUILD_PYTHON` | `ON` | ctypes API integration tests |
| `MICROLLM_BUILD_TORCH_OPS` | `AUTO` | optional PyTorch dispatcher library |
| `MICROLLM_ENABLE_SANITIZERS` | `OFF` | host ASan and UBSan |
| `MICROLLM_ENABLE_COVERAGE` | `OFF` | GCC/Clang line and branch instrumentation |
| `MICROLLM_SAFETENSORS_PYTHON` | empty | interpreter with torch/safetensors used by the optional official interop CTest |
| `MICROLLM_INSTALL_CMAKEDIR` | `lib/cmake/microLLM` | non-empty package-config destination inside and relative to the install prefix |

## Common failures

### HIP requested but no compiler found

Verify `hipcc --version`, then provide the ROCm prefix:

```bash
cmake --preset hip-release -DCMAKE_PREFIX_PATH=/opt/rocm
```

### hipBLASLt or RCCL not found

Check for `hipblaslt-config.cmake` or `rccl-config.cmake` under the ROCm installation.
hipBLASLt is optional; RCCL is required only when `MICROLLM_ENABLE_RCCL=ON`.

### rocWMMA QK benchmark target not found

The target is intentionally conditional. Confirm that `rocwmma-config.cmake` is visible
through `CMAKE_PREFIX_PATH` and that CMake can find an OpenMP C++ target. Its absence does
not disable the existing Attention implementation or add a dependency to an installed SDK.

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
