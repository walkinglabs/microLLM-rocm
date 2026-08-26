#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BLOCK = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-block-softmax"
CACHED = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-cached-softmax"


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
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_cached_softmax.py"
    ast.parse(renderer.read_text(encoding="utf-8"))
    block_workers = workers(BLOCK)
    cached_workers = workers(CACHED)
    summary = json.loads((CACHED / "summary.json").read_text(encoding="utf-8"))
    assert len(block_workers) == len(cached_workers) == 6
    assert all(row["status"] == "pass" and len(row["records"]) == 10
               for row in cached_workers)
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert summary["worker_processes"] == 6 and summary["case_count"] == 10
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_pointers_match"] and row["all_wrappers_non_owning"] and
               row["microllm_peak_extra_bytes_median"] == 0
               for row in summary["groups"])

    block_event = medians(block_workers, "microllm_event_ms")
    cache_event = medians(cached_workers, "microllm_event_ms")
    block_wall = medians(block_workers, "microllm_wall_ms")
    cache_wall = medians(cached_workers, "microllm_wall_ms")
    wide_keys = [key for key in cache_event if key[1] == 4096]
    assert len(wide_keys) == 2
    assert all(block_event[key] / cache_event[key] >= 1.20 for key in wide_keys)
    assert all(block_wall[key] / cache_wall[key] >= 1.18 for key in wide_keys)

    speedups = {(row["dtype"], row["width"]): row["event_speedup_median"]
                for row in summary["groups"]}
    assert all(speedups[key] >= 0.53 for key in wide_keys)
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-cached-softmax.svg")
    print("PyTorch cached Softmax result contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
