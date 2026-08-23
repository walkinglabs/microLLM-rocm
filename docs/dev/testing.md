# Testing and evidence

## Test layers

| Layer | Purpose | Command |
|---|---|---|
| CPU unit/integration | reference math, graph, model, weights, checkpoint | `ctest --preset cpu-debug` |
| CPU sanitizers | lifetime, bounds, undefined behavior | `ctest --preset cpu-sanitize` |
| HIP conformance | CPU/HIP operators, graph, model, direct weight load | `ctest --preset hip-release` |
| RCCL | two-rank collectives and update equivalence | `ctest --preset rccl-release` |
| PyTorch oracle | external FP32 forward/backward/model/optimizer comparison | `ctest ... -R '^TorchOps\.'` |
| Coverage audit | public API gates and test-file registration | `python3 scripts/audit_test_coverage.py` |

## Current measured matrix

```text
framework CPU                 252/252 pass
CPU ASan/UBSan                250/250 pass
full CPU/HIP configuration    370/370 pass (2 conditional skips)
MI300X/gfx942 HIP             114/114 pass
PyTorch-enabled CPU           226/226 pass
two-rank RCCL                  11/11 pass
registered test files               54
```

These counts describe the current commit. They do not imply every dtype, shape, GPU, or
external model is supported.

## Numerical gates

The PyTorch oracle includes deterministic FP32 numerical cases for ordinary and
transpose-aware matmul, tied-head gradients, the operator family and complete tiny
Transformer graphs. Latest
measured maximum absolute differences are:

```text
forward operators             1.90734863e-06
autograd graphs               9.53674316e-07
tiny Transformer              1.43051147e-06
SGD/AdamW                     3.72529030e-08
```

The detailed per-operator tolerances and shape contracts are in
`docs/OPERATOR_CONTRACTS.zh-CN.md`. BF16/F16 safetensors conversion, native basic
kernels and BF16 mixed GEMM are tested. Whole-model BF16 is not yet a correctness or
performance claim; the first shape-selected policy was measured and rejected.

## Line and branch coverage

Test-file registration proves that a test was wired into CTest; it does not prove which
source lines ran. The coverage build is a separate gate:

```bash
python -m pip install gcovr
./scripts/run_coverage.sh /tmp/microllm-coverage
```

It runs CMake clean and then explicitly removes stale runtime profile files from the
fixed coverage build tree before rebuilding. This matters because CMake clean does not
remove those files, and a changed binary would otherwise produce checksum-conflict
warnings during test discovery. The script runs the CPU test label and emits
`summary.json`, Cobertura XML, and a detailed HTML report. Only `src/` and `include/`
are counted. Coverage is evidence for finding blind spots, not permission to replace
numeric, shape, failure, HIP, or external-oracle tests.

The current source snapshot measures 82.6% lines (6,624/8,022), 90.3% functions
(710/786), and 63.2% branches (6,389/10,108). The repeatability evidence for the
coverage pipeline remains separately archived; source growth is expected to change totals.

## Adding a test

Every native `*_test.c/cpp` and Python `test_*.py` file must be registered with CTest
and listed in `tests/coverage_manifest.json`. The audit fails on an unregistered file or
a public math/graph/weight API without an explicit gate.

## Performance evidence

Performance changes require:

1. fixed hardware, ROCm, model, shape, dtype, batch, and context;
2. warm-up and repeated measurements;
3. numerical regression before and after;
4. raw JSONL and, for system claims, a profiler trace;
5. both operator and end-to-end results;
6. a documented regression or unsupported boundary where applicable.

The single-GPU model matrix is a HIP CTest gate and writes its raw build artifact to
`build/hip-release/benchmarks/hip-model-matrix.jsonl`. It records performance without
using a noisy throughput threshold as a correctness condition.

Two additional CPU schema gates recompute built-in and official-HF microLLM/PyTorch
ratios from committed raw JSONL. They reject mismatched workload fields and do not need
PyTorch installed in ordinary CPU CI; generation of new PyTorch raw rows is a separate
ROCm-machine workflow.
