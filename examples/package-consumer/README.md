# Standalone CMake consumer

This directory is deliberately independent of the microLLM source build. It shows the
smallest supported way for another C++ project to consume an installed microLLM SDK.

From the repository root, first build and install a CPU SDK:

```bash
cmake --preset sdk-cpu
cmake --build --preset sdk-cpu --parallel
cmake --install build/sdk-cpu --prefix "$PWD/install/microllm"
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
`-DmicroLLM_DIR="$PWD/build/sdk-cpu"`. Do not point `microLLM_DIR` at the source tree.
