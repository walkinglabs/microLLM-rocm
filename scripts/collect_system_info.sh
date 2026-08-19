#!/usr/bin/env bash
set -euo pipefail

echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "kernel=$(uname -srmo)"
echo "cmake=$(cmake --version | head -1)"
if command -v hipcc >/dev/null 2>&1; then
    echo "hipcc=$(hipcc --version 2>/dev/null | head -1)"
else
    echo "hipcc=not-found"
fi
if command -v rocminfo >/dev/null 2>&1; then
    echo "gpu_agents:"
    rocminfo 2>/dev/null | awk '/^  Name: +gfx/{print "  - "$2}'
else
    echo "gpu_agents: []"
fi
if command -v rocm-smi >/dev/null 2>&1; then
    echo "topology:"
    rocm-smi --showtopo 2>/dev/null || true
fi
