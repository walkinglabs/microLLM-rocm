#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-fp16-wave-softmax"
CANDIDATE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-fast-exp-softmax-reject"


def workers(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            (path / "raw.jsonl").read_text(encoding="utf-8").splitlines()]


def median(rows: list[dict], field: str) -> float:
    return statistics.median(
        record[field]
        for worker in rows
        for record in worker["records"]
        if record["dtype"] == "fp16" and record["width"] == 4096)


def main() -> int:
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_fast_exp_softmax_reject.py"
    ast.parse(renderer.read_text(encoding="utf-8"))
    baseline = workers(BASELINE)
    candidate = workers(CANDIDATE)
    summary = json.loads((CANDIDATE / "summary.json").read_text(encoding="utf-8"))
    assert len(baseline) == len(candidate) == 6
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_pointers_match"] and row["all_wrappers_non_owning"] and
               row["microllm_peak_extra_bytes_median"] == 0
               for row in summary["groups"])
    event_gain = median(baseline, "microllm_event_ms") / \
                 median(candidate, "microllm_event_ms")
    wall_gain = median(baseline, "microllm_wall_ms") / \
                median(candidate, "microllm_wall_ms")
    assert event_gain < 1.05 and wall_gain < 1.05
    source = (ROOT / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    assert "__expf" not in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-fast-exp-softmax-reject.svg")
    print("PyTorch fast-exp Softmax rejection contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
