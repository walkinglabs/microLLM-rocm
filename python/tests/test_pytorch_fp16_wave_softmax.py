#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-cached-softmax"
CANDIDATE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-fp16-wave-softmax"


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
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_fp16_wave_softmax.py"
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
    new_event = medians(candidate, "microllm_event_ms")
    base_wall = medians(baseline, "microllm_wall_ms")
    new_wall = medians(candidate, "microllm_wall_ms")
    bf16 = ("bf16", 4096)
    fp16 = ("fp16", 4096)
    assert 0.95 <= base_event[bf16] / new_event[bf16] <= 1.05
    assert 0.95 <= base_wall[bf16] / new_wall[bf16] <= 1.05
    assert base_event[fp16] / new_event[fp16] >= 1.05
    assert base_wall[fp16] / new_wall[fp16] >= 1.05
    current = {(row["dtype"], row["width"]): row["event_speedup_median"]
               for row in summary["groups"]}
    assert current[fp16] >= 0.60

    source = (ROOT / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    assert "softmax_typed_block_cached_kernel<__half, true>" in source
    assert "constexpr unsigned cached_wave_threads = 1024" in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-fp16-wave-softmax.svg")
    print("PyTorch FP16 wave Softmax result contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
