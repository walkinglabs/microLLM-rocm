#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/distributed/ranked_training_matrix.py"


def main() -> int:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)
    for token in (
        "ranked_training_summary", "ranked_peer_failure_summary",
        "maximum_rank_difference", "maximum_reference_difference",
        "peer_processes_terminated", "median_rank_group_ms",
        "admit one-process-per-GPU ready-bucket migration",
    ):
        assert token in text
    print("ranked training matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
