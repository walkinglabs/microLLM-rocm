# 2026-08-19 — M0/N0 bootstrap

## Observed state

Only the README was tracked in Git. The source tree contained declaration-only
Storage/Tensor headers, INTERFACE CMake targets, and two test-framework smoke tests.
The top-level project required HIP even though the course promises a CPU path.

## Scope of this change

- make HIP, hipBLASLt, and RCCL explicit optional capabilities;
- replace declaration-only core targets with a real static library;
- implement CPU Storage and FP32 Tensor metadata/view behavior;
- add zero-copy transpose/slice, contiguous materialization, and shape validation;
- add an N0 PPM artifact;
- add deterministic, random, lifetime, and negative tests;
- establish architecture, roadmap, evidence, and contributor records.

## Decisions

- Storage owns bytes and device, not dtype or shape.
- Tensor owns dtype/shape/stride/offset and shares Storage.
- TensorView is non-owning and will become the operator/PyTorch integration seam.
- scalar `{}` has one element; zero dimensions are valid; negative strides are not
  supported initially.
- N0 implements CPU float32 only. Other enum values do not imply working kernels.
- cross-device transfer fails explicitly until M1 rather than synchronizing secretly.

## Initial verification

```text
cmake -S . -B build-m0 -DMICROLLM_ENABLE_HIP=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build-m0 --parallel 8
ctest --test-dir build-m0 --output-on-failure
```

Initial result: 17/17 discovered unit tests passed. The PPM test was not initially
registered because CTest was included after the examples directory; the build order
was corrected. Final host verification passed 18/18 tests in both ordinary Debug and
ASan/UBSan builds.

The N0 artifact reported:

```text
checksum_fnv1a=17940541173909021477
```

## Known incomplete areas

- HIP allocation/copy/stream/event;
- operator dispatch and CPU/HIP operator set;
- autograd, model, training, inference, bindings, profiling, and distributed code.

These are subsequent milestones, not capabilities claimed by this record.
