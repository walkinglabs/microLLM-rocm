#!/usr/bin/env bash
set -euo pipefail

MICROLLM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MICROLLM_PROJECT_ROOT="$(cd -- "${MICROLLM_SCRIPT_DIR}/.." && pwd)"
MICROLLM_COVERAGE_OUTPUT="${1:-${MICROLLM_PROJECT_ROOT}/build/cpu-coverage/report}"

if ! command -v gcovr >/dev/null 2>&1; then
    echo "run_coverage: gcovr is required (python -m pip install gcovr)" >&2
    exit 2
fi

cmake --preset cpu-coverage -S "${MICROLLM_PROJECT_ROOT}"
cmake --build --preset cpu-coverage --target clean
cmake --build --preset cpu-coverage --parallel
ctest --preset cpu-coverage
mkdir -p "${MICROLLM_COVERAGE_OUTPUT}"

gcovr \
    --root "${MICROLLM_PROJECT_ROOT}" \
    --object-directory "${MICROLLM_PROJECT_ROOT}/build/cpu-coverage" \
    --filter "${MICROLLM_PROJECT_ROOT}/src/" \
    --filter "${MICROLLM_PROJECT_ROOT}/include/" \
    --exclude-unreachable-branches \
    --exclude-throw-branches \
    --json-summary-pretty \
    --json-summary "${MICROLLM_COVERAGE_OUTPUT}/summary.json" \
    --cobertura-pretty \
    --cobertura "${MICROLLM_COVERAGE_OUTPUT}/cobertura.xml" \
    --html-details "${MICROLLM_COVERAGE_OUTPUT}/index.html" \
    --print-summary

echo "coverage_output=${MICROLLM_COVERAGE_OUTPUT}"
