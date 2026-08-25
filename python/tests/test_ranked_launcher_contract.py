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
        "--reducer", "collectives_per_rank", "buckets_per_rank",
        "--compare-binary", "compare_safetensors", "reference.safetensors",
        "maximum_rank_training_ms", "maximum_rank_reducer_ms",
        "parameter_files_retained", "unlink(missing_ok=True)",
        "maximum_mean_loss_difference", "math.isfinite",
        "maximum_rank_step_reducer_ms",
        "maximum_rank_step_reducer_backend_allocation_calls",
        "persistent-bucket", "plan_reuses_per_rank",
        "maximum_rank_step_plan_reused",
        "maximum_engine_current_bytes", "maximum_engine_peak_bytes",
        "step_reducer_current_bytes_after",
        "bucket-views", "gradient_views_per_rank",
        "maximum_rank_step_gradient_views",
    ):
        assert token in runner
    for token in (
        "create_communicator_id", "RankCommunicator", "timed out waiting",
        "--world-size", "--local-rank", "--id-file", "global_batch",
        "all_reduce_rank_gradients", "--bucket-bytes",
        "save_safetensors", "model-s", "--parameter-file",
        "forward_backward_ms", "reducer_ms", "optimizer_ms",
        "step_reducer_total_allocated_bytes", "allocation_stats",
        "RankGradientBucketPlan", "plan_capacity_bytes",
        "engine_current_bytes", "engine_peak_bytes",
        "gradient_view_count", "bucket-views",
    ):
        assert token in worker
    print("ranked launcher contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
