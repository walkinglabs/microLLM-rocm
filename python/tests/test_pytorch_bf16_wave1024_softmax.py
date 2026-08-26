#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE_BEFORE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-cached-softmax"
CORE_AFTER = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-bf16-wave1024-softmax"
OUT_BEFORE = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-softmax-out"
OUT_AFTER = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-custom-op-softmax-out-wave1024"


def rows(path: Path) -> dict[tuple[str, int], dict]:
    data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert data["status"] == "pass" and data["case_count"] == 10
    return {(row["dtype"], row["width"]): row for row in data["groups"]}


def core_median(path: Path, field: str) -> float:
    values = []
    for line in (path / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        for item in json.loads(line)["records"]:
            if item["dtype"] == "bf16" and item["width"] == 4096:
                values.append(item[field])
    return statistics.median(values)


def main() -> int:
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_bf16_wave1024_softmax.py"
    ast.parse(renderer.read_text(encoding="utf-8"))
    core_before, core_after = rows(CORE_BEFORE), rows(CORE_AFTER)
    out_before, out_after = rows(OUT_BEFORE), rows(OUT_AFTER)
    key = ("bf16", 4096)
    assert (core_median(CORE_BEFORE, "microllm_event_ms") /
            core_median(CORE_AFTER, "microllm_event_ms")) >= 1.50
    assert (core_median(CORE_BEFORE, "microllm_wall_ms") /
            core_median(CORE_AFTER, "microllm_wall_ms")) >= 1.50
    assert core_after[key]["event_speedup_median"] >= 0.85
    assert (out_before[key]["custom_event_ms_median"] /
            out_after[key]["custom_event_ms_median"]) >= 1.50
    assert out_after[key]["event_speedup_median"] >= 0.80
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_pointers_match"] and row["microllm_peak_extra_bytes_median"] == 0
               for row in core_after.values())
    assert all(row["all_returned_pointers_match"] and row["all_peak_extra_zero"]
               for row in out_after.values())
    source = (ROOT / "src/ops/hip/basic_kernels.hip").read_text(encoding="utf-8")
    assert "softmax_typed_block_cached_kernel<hip_bfloat16, true>" in source
    assert "constexpr unsigned cached_wave_threads = 1024" in source
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-bf16-wave1024-softmax.svg")
    print("PyTorch BF16 wave1024 Softmax contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
