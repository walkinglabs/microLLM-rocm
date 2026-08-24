# Standalone CMake consumer

This directory is deliberately independent of the microLLM source build. It shows the
smallest supported way for another C++ project to consume an installed microLLM SDK.

From the repository root, first build and install a CPU SDK:

```bash
cmake -S . -B build/sdk \
  -DMICROLLM_ENABLE_HIP=OFF \
  -DMICROLLM_BUILD_TESTS=OFF \
  -DMICROLLM_BUILD_EXAMPLES=OFF \
  -DMICROLLM_BUILD_BENCHMARKS=OFF
cmake --build build/sdk --parallel
cmake --install build/sdk --prefix "$PWD/install/microllm"
```

Then configure this directory as a separate project:

```bash
cmake -S examples/package-consumer -B build/package-example \
  -DCMAKE_PREFIX_PATH="$PWD/install/microllm"
cmake --build build/package-example
./build/package-example/microllm_package_example
```

Expected output starts with `microLLM package example:`. For a local build-tree
dependency, replace `CMAKE_PREFIX_PATH` with
`-DmicroLLM_DIR="$PWD/build/sdk"`. Do not point `microLLM_DIR` at the source tree.
