# Hardware and software compatibility

Compatibility is an evidence matrix, not a promise based on a product family name.
ROCm support changes by release; check AMD's current
[compatibility matrix](https://rocm.docs.amd.com/en/develop/compatibility/compatibility-matrix.html)
before reproducing a run.

Compiler, CMake, Python, HIP, hipBLASLt, RCCL, and rocprofv3 requirements plus the
exact validated versions are maintained in [dev/build.md](dev/build.md).

## Current measured matrix

| Device | gfx | Software | Evidence | State |
|---|---|---|---|---|
| AMD Instinct MI300X VF | gfx942 | HIP/ROCm runtime 7.13 development build | operators, Model-S forward, tiny/Model-M training | smoke-tested |
| 2× MI300X VF | gfx942, XGMI | RCCL 2.28.3 | global-batch equivalence, buckets, overlap | smoke-tested |
| 4× MI300X VF | gfx942, XGMI | RCCL 2.28.3 | init fails with 64MB `/dev/shm` | failed in current container |
| Radeon | — | — | no hardware run | unverified |
| PyTorch CPU | host | Torch 2.13.0+cpu | Custom Op add/multiply | smoke-tested |
| PyTorch ROCm | gfx942 candidate | Torch 2.11.0 + ROCm 7.13.0rc2 wheel | `import torch` Bus error before binding build | environment failure |

## Radeon validation procedure

1. Confirm the exact GPU/OS/ROCm combination in AMD's matrix.
2. Collect environment evidence:

   ```bash
   ./scripts/collect_system_info.sh > system-info.txt
   ```

3. Configure the reported gfx target explicitly when auto-detection is inappropriate:

   ```bash
   ./scripts/configure.sh \
     -DMICROLLM_ENABLE_HIP=ON \
     -DMICROLLM_HIP_ARCHITECTURES=gfxXXXX \
     -DMICROLLM_ENABLE_RCCL=OFF
   ```

4. Run HIP conformance before a model:

   ```bash
   ./scripts/build.sh
   MICROLLM_BUILD_DIR=build ./scripts/verify_evidence.sh hip
   ```

5. Run tiny model forward/training before Model-S. Record failures, device name, gfx,
   ROCm/driver, memory, exact command, and whether hipBLASLt was found.

## Expected downgrade path

- RCCL is off for one consumer GPU.
- If hipBLASLt is unavailable, Auto selects readable matmul.
- Start with FP32, batch one, and short context.
- BF16/FP16 storage, native basic kernels and tolerance tests pass on the recorded MI300X.
  A later single-representation BF16 FFN inference policy passes official Qwen/DeepSeek
  exact-token and self-baseline gates. It is not full-model BF16, and 3/4 selected PyTorch
  full-BF16 performance rows remain below parity.
- A successful compile does not establish kernel correctness or performance.

## Publication rule

Add a Radeon row to the measured matrix only after CPU/HIP conformance, tiny training,
Model-S forward where memory permits, peak engine allocation, and raw benchmark
metadata are retained. Do not infer support for neighboring SKUs.
