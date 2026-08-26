#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-autograd-external-gradient-pool"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/external_gradient_pool_matrix.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_external_gradient_pool.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))

    raw = [json.loads(line) for line in
           (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (RESULTS / "verification.json").read_text(encoding="utf-8"))
    assert len(raw) == 18
    assert {(row["model"], row["context"], row["first"])
            for row in raw} == {
                ("tiny", 8, "baseline"), ("tiny", 8, "external"),
                ("model-s", 8, "baseline"), ("model-s", 8, "external"),
                ("model-s", 32, "baseline"), ("model-s", 32, "external"),
            }
    assert all(row["status"] == "pass" for row in raw)
    assert all(row["maximum_gradient_error"] == 0.0 for row in raw)
    assert all(row["rms_gradient_error"] == 0.0 for row in raw)
    assert all(row["all_external_addresses_stable"] for row in raw)

    assert summary["status"] == "pass"
    assert summary["correctness_pass"] is True
    assert summary["performance_pass"] is False
    assert summary["decision"] == "keep_explicit_interop_only"
    assert len(summary["groups"]) == 3
    assert all(group["processes"] == 6 for group in summary["groups"])
    assert all(group["event_speedup_median"] < 1.0
               for group in summary["groups"])
    assert all(group["wall_speedup_median"] < 1.0
               for group in summary["groups"])
    assert all(group["peak_extra_bytes_delta_median"] > 0
               for group in summary["groups"])

    assert verification == {
        "all_addresses_stable": True,
        "all_gradients_exact": True,
        "all_raw_records_pass": True,
        "contexts": [8, 32],
        "models": ["model-s", "tiny"],
        "processes": 18,
        "rotated_orders": ["baseline", "external"],
        "schema_version": 1,
        "status": "pass",
    }
    ET.parse(ROOT / "docs/optimization-log/assets/external-gradient-pool-discard.svg")
    print("external gradient pool matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

