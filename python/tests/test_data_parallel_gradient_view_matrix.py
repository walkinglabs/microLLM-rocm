#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/data_parallel_gradient_view_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "--gradient-bucket-views", "gradient_view_count",
        "unpack_storage_removed", "unpack_copies_removed",
        "BUCKET_ONLY_BYTES", "current_bytes_added_vs_transient",
        "keep explicit and continue to direct bucket-gradient accumulation",
    ):
        assert token in text
    print("data-parallel gradient view matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
