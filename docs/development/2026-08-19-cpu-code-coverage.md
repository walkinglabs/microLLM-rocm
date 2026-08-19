# 2026-08-19 — measured CPU code coverage

## Goal

The API/test-file audit answers “did we remember a test file?” It cannot answer “which
source paths actually ran?” This milestone adds an instrumented build and records the
first real line/function/branch result.

## Reproducible gate

```text
compiler       GCC/G++ 13.3.0
coverage tool  gcovr 8.3
scope          src/ and include/
CTest          119/119 pass (dynamic C/Python bindings excluded)
lines          2753 / 3281 = 83.9%
functions      390 / 429 = 90.9%
branches       2672 / 4011 = 66.6%
```

Run it with:

```bash
python -m pip install gcovr==8.3
./scripts/run_coverage.sh /tmp/microllm-coverage
```

The output includes machine-readable JSON, Cobertura XML, and per-line HTML. The compact
tracked result is in `benchmarks/results/2026-08-19-cpu-coverage/coverage-summary.json`.

## What the number does and does not mean

The lowest files are `src/ops/optimized.cpp` (26.5% lines) and
`src/runtime/runtime.cpp` (63.4% lines). That is expected in a CPU-only report because
large parts select hipBLASLt or HIP runtime behavior. It is still a useful warning:
those paths need a separate GPU coverage/evidence job rather than being hidden inside a
single optimistic percentage.

No minimum percentage is enforced yet. First split host-only and HIP-only code in the
report, then choose thresholds that cannot be raised by adding unimportant tests while
critical numeric and failure paths remain uncovered.
