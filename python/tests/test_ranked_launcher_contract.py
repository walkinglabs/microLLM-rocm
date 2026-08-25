#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/distributed/run_ranked.py"
WORKER = ROOT / "apps/distributed_rank.cpp"


def main() -> int:
    runner = RUNNER.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    ast.parse(runner)
    for token in (
        "--failure-mode", "peer-failure", "peer_processes_terminated",
        "maximum_rank_difference", "maximum_reference_difference",
        "communicator.id", "timeout-seconds", "parameter_values",
    ):
        assert token in runner
    for token in (
        "create_communicator_id", "RankCommunicator", "timed out waiting",
        "--world-size", "--local-rank", "--id-file", "global_batch",
    ):
        assert token in worker
    print("ranked launcher contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
