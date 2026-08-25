# CMake Config package completion record

## Problem

An external project should be able to use microLLM without copying source files or
guessing header and library flags. The supported contract is a relocatable CMake Config
package discovered with `find_package(microLLM CONFIG REQUIRED)`.

## Public contract

- `microLLM::microLLM` is the default single-device C++ SDK target. It carries both
  training and inference and does not silently add RCCL, the C ABI, or PyTorch.
- Narrow component targets remain available for callers that want only one layer.
- `capi` and `multi_gpu` are optional components and fail clearly when absent.
- `microLLM_DEFAULT_TARGET`, feature flags, version fields, enabled GPU architectures,
  and the component list are readable from the Config package.
- Both an already configured build tree and a moved installation tree are supported.

## Evidence gate

The repository does not accept the package merely because Config files were generated.
CTest must install or export it, configure a separate source tree, compile, link, and run
the consumer. The gate also covers a project whose only enabled language is C, rejects a
missing required component, rejects an incompatible pre-1.0 minor version, and moves the
installation before discovery to catch hard-coded paths.

Run the focused gate with:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --test-dir build/cpu-debug -R '^PackageConfig\\.' --output-on-failure
```

The copy-paste public example lives in `examples/package-consumer`; README and build
documentation use the same target names exercised by that gate.
