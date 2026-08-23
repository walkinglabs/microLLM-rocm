#!/usr/bin/env bash
set -euo pipefail

MICROLLM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MICROLLM_PROJECT_ROOT="$(cd -- "${MICROLLM_SCRIPT_DIR}/.." && pwd)"
MICROLLM_COVERAGE_BUILD="${MICROLLM_PROJECT_ROOT}/build/cpu-coverage"
MICROLLM_COVERAGE_OUTPUT="${1:-${MICROLLM_PROJECT_ROOT}/build/cpu-coverage/report}"

if ! command -v gcovr >/dev/null 2>&1; then
    echo "run_coverage: gcovr is required (python -m pip install gcovr)" >&2
    exit 2
fi

# Turn any future runtime profile mismatch into a failing process instead of a
# warning that can scroll past in a successful coverage job.
export GCOV_EXIT_AT_ERROR=1

cmake --preset cpu-coverage -S "${MICROLLM_PROJECT_ROOT}"
cmake --build --preset cpu-coverage --target clean
# CMake's clean target removes objects but intentionally leaves runtime profile
# data. A rebuilt GTest binary can otherwise read an old-checksum .gcda while
# gtest_discover_tests enumerates it, contaminating an otherwise successful run.
if [[ -d "${MICROLLM_COVERAGE_BUILD}" ]]; then
    MICROLLM_STALE_PROFILE_COUNT="$({
        find "${MICROLLM_COVERAGE_BUILD}" -type f -name '*.gcda' -print
    } | wc -l)"
    find "${MICROLLM_COVERAGE_BUILD}" -type f -name '*.gcda' -delete
    echo "coverage_stale_profiles_removed=${MICROLLM_STALE_PROFILE_COUNT//[[:space:]]/}"
fi
cmake --build --preset cpu-coverage --parallel
ctest --preset cpu-coverage
mkdir -p "${MICROLLM_COVERAGE_OUTPUT}"

gcovr \
    --root "${MICROLLM_PROJECT_ROOT}" \
    --object-directory "${MICROLLM_COVERAGE_BUILD}" \
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
