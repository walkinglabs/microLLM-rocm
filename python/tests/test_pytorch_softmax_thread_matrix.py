#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = {
    128: ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-softmax-threads128",
    256: ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-fp16-wave-softmax",
    512: ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-softmax-threads512",
    1024: ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-softmax-threads1024",
}


def measurements(path: Path) -> tuple[float, float, float]:
    workers = [json.loads(line) for line in
               (path / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert len(workers) == 6 and summary["correctness_pass"] is True
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_pointers_match"] and row["all_wrappers_non_owning"] and
               row["microllm_peak_extra_bytes_median"] == 0
               for row in summary["groups"])
    selected = [row for worker in workers for row in worker["records"]
                if row["dtype"] == "fp16" and row["width"] == 4096]
    return (
        statistics.median(row["microllm_event_ms"] for row in selected),
        statistics.median(row["microllm_wall_ms"] for row in selected),
        statistics.median(row["torch_event_ms"] / row["microllm_event_ms"]
                          for row in selected),
    )


def main() -> int:
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_softmax_thread_matrix.py"
    ast.parse(renderer.read_text(encoding="utf-8"))
    values = {threads: measurements(path) for threads, path in RESULTS.items()}
    assert values[128][0] > values[256][0]
    assert values[256][0] / values[512][0] >= 1.30
    assert values[256][1] / values[512][1] >= 1.30
    assert values[512][0] / values[1024][0] >= 1.05
    assert values[512][1] / values[1024][1] >= 1.05
    assert values[1024][2] >= 0.85
    source = (ROOT / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    assert "constexpr unsigned cached_wave_threads = 1024" in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-softmax-thread-matrix.svg")
    print("PyTorch Softmax thread matrix contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
