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
        "pack_copy_calls", "unpack_copy_calls",
        "communication_allocation_calls",
        "communication_backend_allocation_calls",
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
