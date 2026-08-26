#!/usr/bin/env python3
"""Render typed Softmax correctness/memory baseline and performance failure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = sorted(summary["groups"], key=lambda row: (row["dtype"], row["width"]))
    width, height = 1320, 720
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1320" height="720" fill="#f7f9fc"/>',
        '<text x="660" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Typed Softmax Is Correct and Zero-Temporary, but Serial</text>',
        '<text x="660" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">Torch / microLLM Event ratio on log scale · six fresh MI300X processes</text>',
        '<line x1="100" y1="570" x2="970" y2="570" stroke="#aab3c2" stroke-width="2"/>',
    ]
    log_min, log_max = -3.0, math.log10(2.0)
    for tick in (0.001, 0.01, 0.1, 1.0):
        y = 570 - (math.log10(tick) - log_min) / (log_max - log_min) * 430
        lines.append(f'<line x1="100" y1="{y:.1f}" x2="970" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(f'<text x="88" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:g}×</text>')
    for index, row in enumerate(groups):
        center = 145 + index * 80
        value = row["event_speedup_median"]
        normalized = (math.log10(max(value, 0.001)) - log_min) / (log_max - log_min)
        y = 570 - normalized * 430
        color = "#198754" if value >= 1.0 else "#b42335"
        lines.append(f'<rect x="{center - 23}" y="{y:.1f}" width="46" height="{570 - y:.1f}" rx="6" fill="{color}"/>')
        lines.append(f'<text x="{center}" y="{y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="11" font-weight="700" fill="#172033">{value:.4f}</text>')
        lines.append(f'<text x="{center}" y="597" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#172033" transform="rotate(40 {center} 597)">{row["dtype"]} w{row["width"]}</text>')
    lines.extend([
        '<rect x="1010" y="125" width="260" height="445" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1040" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Evidence</text>',
        '<text x="1040" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 10 / 10 precision rows</text>',
        '<text x="1040" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ all pointers identical</text>',
        '<text x="1040" y="277" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ wrapper non-owning</text>',
        '<text x="1040" y="310" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ micro peak extra = 0</text>',
        '<text x="1040" y="358" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">w1024 ≈ 0.011×</text>',
        '<text x="1040" y="391" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">w4096 ≈ 0.004×</text>',
        '<text x="1040" y="439" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">one thread scans one row</text>',
        '<text x="1040" y="472" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: block reduction</text>',
        '<text x="660" y="680" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep correctness baseline · reject performance readiness</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
