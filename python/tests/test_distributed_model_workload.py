#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "apps/distributed_train.cpp"
RUNNER = ROOT / "tools/distributed/run.py"


def main() -> int:
    cli = CLI.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    for token in (
        '"--model"', '"model-s"', "ModelConfig::model_s()",
        '"--context"', '"--batch"', "bucket_parameter_count",
        "bucket_total_elements", "maximum_engine_peak_bytes",
        "bucket_temporary_bytes", "average_tensor_count",
        "bucket_plan_capacity_bytes", "bucket_plan_reused",
        "pack_copy_calls", "unpack_copy_calls",
        "communication_allocation_calls",
        "communication_backend_allocation_calls",
        "--inplace-bucket-average", "inplace_bucket_average",
        "--persistent-gradient-buckets", "persistent_gradient_buckets",
        "--gradient-bucket-views", "gradient_bucket_views",
        "gradient_view_count",
        "--direct-bucket-gradients", "direct_bucket_gradients",
        "direct_gradient_target_count",
        "maximum_engine_current_bytes",
        "distributed step failed its loss or rank-parameter gate",
    ):
        assert token in cli
    for token in (
        'choices=("tiny", "model-s")', '"--context"', '"--batch"',
        '"parameter_check_interval"',
    ):
        assert token in runner
    print("distributed model workload contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
