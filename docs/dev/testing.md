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
framework CPU                 121/121 pass
CPU ASan/UBSan                119/119 pass
MI300X/gfx942 HIP              25/25 pass
PyTorch CPU oracle/alignment      3/3 pass
two-rank RCCL                  11/11 pass
registered test files               33
```

These counts describe the current commit. They do not imply every dtype, shape, GPU, or
external model is supported.

## Numerical gates

The PyTorch oracle currently contains 70 deterministic FP32 numerical cases. Latest
measured maximum absolute differences are:

```text
forward operators             1.90734863e-06
autograd graphs               9.53674316e-07
tiny Transformer              1.43051147e-06
SGD/AdamW                     3.72529030e-08
```

The detailed per-operator tolerances and shape contracts are in
`docs/OPERATOR_CONTRACTS.zh-CN.md`. BF16/F16 safetensors conversion is tested, but
BF16/F16 compute kernels are not yet a correctness claim.

## Line and branch coverage

Test-file registration proves that a test was wired into CTest; it does not prove which
source lines ran. The coverage build is a separate gate:

```bash
python -m pip install gcovr
./scripts/run_coverage.sh /tmp/microllm-coverage
```

It starts from a clean generated build, runs the same CPU test label, and emits
`summary.json`, Cobertura XML, and a detailed HTML report. Only `src/` and `include/`
are counted. Coverage is evidence for finding blind spots, not permission to replace
numeric, shape, failure, HIP, or external-oracle tests.

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
