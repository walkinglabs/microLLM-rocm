#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-typed-softmax"


def main() -> int:
    runner = ROOT / "benchmarks/single_gpu/pytorch_zero_copy_typed_softmax.py"
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_typed_softmax.py"
    ast.parse(runner.read_text(encoding="utf-8"))
    ast.parse(renderer.read_text(encoding="utf-8"))
    workers = [json.loads(line) for line in
               (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 6
    assert all(worker["status"] == "pass" and len(worker["records"]) == 10
               for worker in workers)
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert summary["worker_processes"] == 6 and summary["case_count"] == 10
    assert {row["dtype"] for row in summary["groups"]} == {"fp16", "bf16"}
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_pointers_match"] and row["all_wrappers_non_owning"] and
               row["microllm_peak_extra_bytes_median"] == 0
               for row in summary["groups"])
    wide = [row for row in summary["groups"] if row["width"] == 4096]
    model = [row for row in summary["groups"] if row["width"] == 1024]
    assert max(row["event_speedup_median"] for row in wide) <= 0.0041
    assert max(row["event_speedup_median"] for row in model) <= 0.012
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-typed-softmax.svg")
    print("PyTorch typed Softmax baseline contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
