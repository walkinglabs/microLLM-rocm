#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cmake --preset cpu-sanitize -S "$PROJECT_DIR"
cmake --build --preset cpu-sanitize --parallel "${MICROLLM_JOBS:-$(nproc)}"
ctest --preset cpu-sanitize
