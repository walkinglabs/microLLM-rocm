#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERIAL = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-typed-softmax"
BLOCK = ROOT / "benchmarks/results/2026-08-26-pytorch-rocm-block-softmax"


def workers(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            (path / "raw.jsonl").read_text(encoding="utf-8").splitlines()]


def event_medians(rows: list[dict]) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for worker in rows:
        for record in worker["records"]:
            grouped.setdefault((record["dtype"], record["width"]), []).append(
                record["microllm_event_ms"])
    return {key: statistics.median(values) for key, values in grouped.items()}


def main() -> int:
    renderer = ROOT / "docs/optimization-log/scripts/render_pytorch_block_softmax.py"
    ast.parse(renderer.read_text(encoding="utf-8"))
    serial_workers = workers(SERIAL)
    block_workers = workers(BLOCK)
    summary = json.loads((BLOCK / "summary.json").read_text(encoding="utf-8"))
    assert len(serial_workers) == len(block_workers) == 6
    assert all(row["status"] == "pass" and len(row["records"]) == 10
               for row in block_workers)
    assert summary["status"] == "pass" and summary["correctness_pass"] is True
    assert summary["worker_processes"] == 6 and summary["case_count"] == 10
    assert all(row["maximum_error"] <= row["tolerance"] and
               row["all_pointers_match"] and row["all_wrappers_non_owning"] and
               row["microllm_peak_extra_bytes_median"] == 0
               for row in summary["groups"])

    serial_times = event_medians(serial_workers)
    block_times = event_medians(block_workers)
    gains = {key: serial_times[key] / block_times[key] for key in block_times}
    assert all(gains[key] >= 0.95 for key in gains if key[1] in (1, 17))
    assert all(gains[key] >= 10.0 for key in gains if key[1] == 128)
    assert all(gains[key] >= 80.0 for key in gains if key[1] == 1024)
    assert all(gains[key] >= 120.0 for key in gains if key[1] == 4096)

    speedups = {(row["dtype"], row["width"]): row["event_speedup_median"]
                for row in summary["groups"]}
    assert all(speedups[key] >= 1.0 for key in speedups if key[1] in (128, 1024))
    assert all(speedups[key] >= 0.40 for key in speedups if key[1] == 4096)
    ET.parse(ROOT / "docs/optimization-log/assets/pytorch-rocm-block-softmax.svg")
    print("PyTorch block Softmax result contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
