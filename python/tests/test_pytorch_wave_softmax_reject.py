#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-cached-softmax"
CANDIDATE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-wave-softmax-reject"


def workers(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            (path / "raw.jsonl").read_text(encoding="utf-8").splitlines()]


def medians(rows: list[dict], field: str) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for worker in rows:
        for record in worker["records"]:
            grouped.setdefault((record["dtype"], record["width"]), []).append(
                record[field])
    return {key: statistics.median(values) for key, values in grouped.items()}


def main() -> int:
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_wave_softmax_reject.py"
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

    base_event = medians(baseline, "microllm_event_ms")
    wave_event = medians(candidate, "microllm_event_ms")
    base_wall = medians(baseline, "microllm_wall_ms")
    wave_wall = medians(candidate, "microllm_wall_ms")
    bf16 = ("bf16", 4096)
    fp16 = ("fp16", 4096)
    assert base_event[bf16] / wave_event[bf16] >= 1.05
    assert base_wall[bf16] / wave_wall[bf16] < 1.05
    assert base_event[fp16] / wave_event[fp16] >= 1.05
    assert base_wall[fp16] / wave_wall[fp16] >= 1.05
    source = (ROOT / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    assert "block_reduce_sum_wave" not in source
    assert "block_reduce_max_wave" not in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-wave-softmax-reject.svg")
    print("PyTorch wave Softmax rejection contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
