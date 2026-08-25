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
        "collective_reduction", "bucket_wall_speedup", "per-parameter",
        "--compare-binary", "model-s", "15586176",
        "bucket_training_speedup", "bucket_reducer_speedup",
        "median_maximum_rank_training_ms",
        "maximum_mean_loss_difference",
        "--steady-skip-steps", "bucket_steady_reducer_speedup",
        "bucket_steady_training_speedup",
        "steady_maximum_rank_reducer_cv",
        "median_steady_reducer_backend_allocation_calls",
        "median_steady_pack_copies", "median_steady_unpack_copies",
        "--policies", "persistent-bucket",
        "persistent_steady_reducer_speedup_vs_per_parameter",
        "persistent_steady_training_speedup_vs_transient",
        "persistent_maximum_steady_backend_allocation_calls",
        "persistent_current_bytes_added_vs_transient",
        "persistent_peak_bytes_added_vs_per_parameter",
        "bucket-views", "views_steady_reducer_speedup_vs_persistent_copy",
        "views_steady_training_speedup_vs_per_parameter",
        "views_plan_capacity_bytes_per_rank",
        "views_current_bytes_added_vs_persistent_copy",
        "overlap-views", "overlap_steady_finish_speedup_vs_synchronous_views",
        "overlap_steady_training_speedup_vs_synchronous_views",
        "overlap_peak_bytes_added_vs_synchronous_views",
        "--context", "context exceeds the selected model contract",
        "--world-size", "args.world_size",
        "--rank-batch-rows", "--input-weighting", "token-weighted",
        "maximum_rank_step_weighted_gradient_scales",
        "rank_weighted_gradient_scales",
        "maximum_weighted_gradient_scales_per_rank",
        "bucket-weighted-overlap",
        "maximum_rank_step_weighted_bucket_scales",
        "maximum_weighted_bucket_scales_per_rank",
        "profile ranked Model-S cold and steady reducer",
        "admit measured ranked Model-S bucket baseline",
        "admit one-process-per-GPU ready-bucket migration",
    ):
        assert token in text
    worker = (ROOT / "apps/distributed_rank.cpp").read_text(encoding="utf-8")
    hook_begin = worker.index("set_gradient_ready_hook(")
    hook = worker[hook_begin:worker.index("PhaseTimings timings;", hook_begin)]
    assert hook.index("scale_in_place_") < hook.index("mark_parameter_ready")
    print("ranked training matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
