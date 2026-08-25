#!/usr/bin/env python3
"""Static contract for the ranked overlap context-scale runner."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    text = (ROOT / "benchmarks/distributed/ranked_overlap_context_matrix.py").read_text(
        encoding="utf-8")
    for token in (
        "--contexts", "32", "128", "bucket-views", "overlap-views",
        "steady-skip-steps", "median_steady_training_ms",
        "median_steady_finish_ms", "finish_speedup", "training_speedup",
        "forward_backward_added_ms", "maximum_engine_current_bytes",
        "maximum_engine_peak_bytes", "minimum_required_speedup",
        "longer_context_gate_passed", "peer_processes_terminated",
        "retain context-selective ranked overlap",
        "close Model-S ranked overlap scale track",
    ):
        assert token in text
    print("ranked overlap context matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
