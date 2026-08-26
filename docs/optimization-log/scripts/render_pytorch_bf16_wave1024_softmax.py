#!/usr/bin/env python3
"""Render BF16 tree256 versus wave1024 evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def row(path: Path, dtype: str, width: int) -> dict:
    data = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return next(item for item in data["groups"]
                if item["dtype"] == dtype and item["width"] == width)


def core_median(path: Path, field: str) -> float:
    values = []
    for line in (path / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        for item in json.loads(line)["records"]:
            if item["dtype"] == "bf16" and item["width"] == 4096:
                values.append(item[field])
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-baseline", type=Path, required=True)
    parser.add_argument("--core-candidate", type=Path, required=True)
    parser.add_argument("--out-baseline", type=Path, required=True)
    parser.add_argument("--out-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core_before = row(args.core_baseline, "bf16", 4096)
    core_after = row(args.core_candidate, "bf16", 4096)
    out_before = row(args.out_baseline, "bf16", 4096)
    out_after = row(args.out_candidate, "bf16", 4096)
    core_before_event = core_median(args.core_baseline, "microllm_event_ms")
    core_after_event = core_median(args.core_candidate, "microllm_event_ms")
    core_before_wall = core_median(args.core_baseline, "microllm_wall_ms")
    core_after_wall = core_median(args.core_candidate, "microllm_wall_ms")
    values = (
        ("core tree256", core_before_event * 1000.0, "#aab3c2"),
        ("core wave1024", core_after_event * 1000.0, "#198754"),
        ("Custom out before", out_before["custom_event_ms_median"] * 1000.0, "#d39b45"),
        ("Custom out after", out_after["custom_event_ms_median"] * 1000.0, "#2b8a66"),
    )
    width, height = 1240, 650
    left, right, top, bottom = 100, 800, 125, 525
    maximum = 10.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="620" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">BF16 Needed More Waves, Not the Old Broad Policy</text>',
        '<text x="620" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">BF16 rows8 × width4096 Event time · lower is better · MI300X</text>',
    ]
    for tick in (0, 2, 4, 6, 8, 10):
        y = bottom - tick / maximum * (bottom - top)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="88" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick} μs</text>')
    for index, (label, value, color) in enumerate(values):
        center = 180 + index * 165
        y = bottom - value / maximum * (bottom - top)
        lines.append(
            f'<rect x="{center - 42}" y="{y:.1f}" width="84" height="{bottom - y:.1f}" rx="8" fill="{color}"/>')
        lines.append(
            f'<text x="{center}" y="{y - 10:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#172033">{value:.3f}</text>')
        lines.append(
            f'<text x="{center}" y="558" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#172033">{label}</text>')
    core_event_gain = core_before_event / core_after_event
    core_wall_gain = core_before_wall / core_after_wall
    out_gain = (out_before["custom_event_ms_median"] /
                out_after["custom_event_ms_median"])
    lines.extend([
        '<rect x="845" y="125" width="345" height="400" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="875" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Scoped keep</text>',
        '<text x="875" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 10 / 10 precision + resources</text>',
        f'<text x="875" y="266" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">core Event / wall {core_event_gain:.3f}× / {core_wall_gain:.3f}×</text>',
        f'<text x="875" y="299" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">Custom out Event {out_gain:.3f}×</text>',
        '<text x="875" y="349" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">ctypes / PyTorch = 0.888×</text>',
        '<text x="875" y="382" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">Custom out / native = 0.804×</text>',
        '<text x="875" y="432" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">predicate = BF16 + cached range</text>',
        '<text x="875" y="463" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">+ wave + 1024 threads</text>',
        '<text x="620" y="615" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Thread count is part of the algorithm contract</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
