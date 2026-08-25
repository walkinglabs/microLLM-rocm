#!/usr/bin/env python3
"""Static contract for ranked checkpoint ownership and resume."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    runner = (ROOT / "tools/distributed/run_ranked_checkpoint.py").read_text(
        encoding="utf-8")
    worker = (ROOT / "apps/distributed_rank.cpp").read_text(encoding="utf-8")
    for token in (
        "first-steps", "resumed-steps", "interrupted.ckpt",
        "resumed-final.ckpt", "uninterrupted-final.ckpt",
        "checkpoint_bytes_equal", "rank0_checkpoint_writes",
        "nonzero_rank_checkpoint_writes", "checkpoint_files_retained",
        "inject-checkpoint-failure", "peer_processes_terminated",
        "admit Model-S ranked checkpoint smoke",
    ):
        assert token in runner
    for token in (
        "save_checkpoint", "restore_checkpoint", "publish_checkpoint_ready",
        "wait_for_checkpoint_ready", "checkpoint_written",
        "optimizer_step", "injected rank0 checkpoint failure",
    ):
        assert token in worker
    print("ranked checkpoint contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
