# CMake Config and README usability audit

## Why this follow-up exists

Having a `microLLMConfig.cmake` template is not enough. A usable SDK must let a
different project discover the package, inherit the correct headers and libraries,
compile, link, and run without copying microLLM sources or guessing compiler flags.
The repository README must also expose that path before the chronological performance
notes overwhelm a new reader.

## Audited public path

The supported beginner path is:

```text
configure microLLM -> build -> install into one prefix
                  -> find_package(microLLM CONFIG REQUIRED)
                  -> link microLLM::microLLM
                  -> build and run the external application
```

The Config package records the package version, C++20 requirement, available component
targets, enabled HIP architecture, and optional HIP, hipBLASLt, rocWMMA, RCCL, C API,
sanitizer, and coverage features. Backend dependencies are resolved from the build that
produced the SDK; a CPU-only installation does not ask its consumer to find ROCm.

## README change

The project maturity statement remains visible at the top. A short status table now
separates what works from what is still unproven, and the long sequence of optimization
checkpoints is collapsed. Quick Start, the CMake package entry point, the evidence status,
and the optimization journal remain one click away. No experiment or limitation was
deleted.

## Fresh result

Validated with CMake 3.31.10 and GCC/G++ 13.3.0 on the CPU Debug build:

| Gate | Result | What it proves |
|---|---:|---|
| `PackageConfig.InstalledConsumer` | pass | install, move prefix, discover, compile, link, and run |
| `PackageConfig.BuildTreeConsumer` | pass | consume an already-built checkout without installation |
| `PackageConfig.PublicExample` | pass | the copy-paste README/example path remains executable |

The consumer gates additionally exercise the default C++ SDK target, narrower required
components, optional components, a genuinely C-only application, unavailable-component
failure, incompatible-version failure, and non-leakage of repository-only warning flags.

Run the same focused contract with:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --test-dir build/cpu-debug -R '^PackageConfig\\.' --output-on-failure
```
