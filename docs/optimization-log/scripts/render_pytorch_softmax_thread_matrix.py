#!/usr/bin/env python3
"""Render the FP16 cached/wave Softmax block-size matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def measurements(path: Path) -> dict[str, float]:
    rows = []
    for line in (path / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        rows.extend(json.loads(line)["records"])
    selected = [row for row in rows
                if row["dtype"] == "fp16" and row["width"] == 4096]
    return {
        "event_us": statistics.median(
            row["microllm_event_ms"] for row in selected) * 1000.0,
        "wall_us": statistics.median(
            row["microllm_wall_ms"] for row in selected) * 1000.0,
        "torch_us": statistics.median(
            row["torch_event_ms"] for row in selected) * 1000.0,
        "torch_ratio": statistics.median(
            row["torch_event_ms"] / row["microllm_event_ms"]
            for row in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads128", type=Path, required=True)
    parser.add_argument("--threads256", type=Path, required=True)
    parser.add_argument("--threads512", type=Path, required=True)
    parser.add_argument("--threads1024", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = {
        128: measurements(args.threads128),
        256: measurements(args.threads256),
        512: measurements(args.threads512),
        1024: measurements(args.threads1024),
    }

    width, height = 1380, 720
    left, right, top, bottom = 105, 930, 125, 585
    maximum = 14.0
    colors = {128: "#b42335", 256: "#aab3c2", 512: "#52a36d", 1024: "#198754"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f7f9fc"/>',
        '<text x="690" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Wide FP16 Softmax Wants the Full Workgroup</text>',
        '<text x="690" y="79" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">width4096 · lower is better · six fresh MI300X processes per thread count</text>',
    ]
    for tick in (0, 2, 4, 6, 8, 10, 12, 14):
        y = bottom - tick / maximum * (bottom - top)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(
            f'<text x="92" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick} μs</text>')
    torch_us = values[1024]["torch_us"]
    torch_y = bottom - torch_us / maximum * (bottom - top)
    lines.append(
        f'<line x1="{left}" y1="{torch_y:.1f}" x2="{right}" y2="{torch_y:.1f}" stroke="#4776c5" stroke-width="2" stroke-dasharray="8 7"/>')
    lines.append(
        f'<text x="{right - 5}" y="{torch_y - 8:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" font-weight="700" fill="#4776c5">PyTorch {torch_us:.2f} μs</text>')

    for index, threads in enumerate((128, 256, 512, 1024)):
        center = 210 + index * 190
        for offset, field, label in ((-34, "event_us", "Event"),
                                     (34, "wall_us", "wall")):
            value = values[threads][field]
            x = center + offset
            y = bottom - value / maximum * (bottom - top)
            lines.append(
                f'<rect x="{x - 25}" y="{y:.1f}" width="50" height="{bottom - y:.1f}" rx="6" fill="{colors[threads]}" opacity="{1.0 if field == "event_us" else 0.62}"/>')
            lines.append(
                f'<text x="{x}" y="{y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{value:.2f}</text>')
            lines.append(
                f'<text x="{x}" y="610" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#5b6474">{label}</text>')
        lines.append(
            f'<text x="{center}" y="648" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">{threads} threads</text>')

    gain_512 = values[256]["event_us"] / values[512]["event_us"]
    gain_1024_event = values[512]["event_us"] / values[1024]["event_us"]
    gain_1024_wall = values[512]["wall_us"] / values[1024]["wall_us"]
    lines.extend([
        '<rect x="970" y="125" width="360" height="460" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1000" y="168" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Matrix decision</text>',
        '<text x="1000" y="216" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ all precision / pointer / peak gates</text>',
        '<text x="1000" y="261" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">128: slower than 256</text>',
        f'<text x="1000" y="294" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">512 / 256 Event = {gain_512:.3f}×</text>',
        f'<text x="1000" y="327" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">1024 / 512 Event = {gain_1024_event:.3f}×</text>',
        f'<text x="1000" y="360" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">1024 / 512 wall = {gain_1024_wall:.3f}×</text>',
        '<text x="1000" y="410" font-family="Inter,Arial,sans-serif" font-size="16" font-weight="700" fill="#198754">retain 1024 threads</text>',
        '<text x="1000" y="457" font-family="Inter,Arial,sans-serif" font-size="15" fill="#e28b22">current = 0.880× PyTorch</text>',
        '<text x="1000" y="490" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: attribute ≈0.6 μs gap</text>',
        '<text x="1000" y="523" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">before another Kernel edit</text>',
        '<text x="690" y="690" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">1024 is measured winner, not a generic thread-count rule</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
