#!/usr/bin/env python3
"""Render scalar/broad/selective vector16 Custom Op comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    selected = [row for row in report["groups"] if row["kind"] == "forward" and
                row["shape"] == "bandwidth"]
    order = [("fp32", "add"), ("fp32", "multiply"),
             ("fp16", "add"), ("fp16", "multiply"),
             ("bf16", "add"), ("bf16", "multiply")]
    rows = [next(row for row in selected
                 if (row["dtype"], row["operation"]) == item) for item in order]
    labels = [f"{dtype.upper()} {operation}" for dtype, operation in order]
    width, height = 1320, 720
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1320" height="720" fill="#f7f9fc"/>',
        '<text x="660" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Vector16 Needs a Shape and Dtype Gate</text>',
        '<text x="660" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">16M elements · candidate/scalar Event speed · three matrices × six fresh processes</text>',
        '<line x1="80" y1="510" x2="940" y2="510" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.0, 0.5, 1.0, 1.5):
        y = 510 - tick / 1.5 * 360
        lines.append(f'<line x1="80" y1="{y:.1f}" x2="940" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(f'<text x="68" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.1f}×</text>')
    for index, (label, row) in enumerate(zip(labels, rows)):
        center = 155 + index * 130
        broad = row["broad_vs_scalar_event"]
        selective = row["selective_vs_scalar_event"]
        for offset, value, color in ((-24, broad, "#d97706"),
                                     (24, selective, "#3769d6")):
            bar_height = value / 1.5 * 360
            y = 510 - bar_height
            lines.append(f'<rect x="{center + offset - 19}" y="{y:.1f}" width="38" height="{bar_height:.1f}" rx="5" fill="{color}"/>')
            lines.append(f'<text x="{center + offset}" y="{y - 7:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{value:.3f}</text>')
        lines.append(f'<text x="{center}" y="540" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="#172033">{label}</text>')
    lines.extend([
        '<rect x="980" y="125" width="285" height="385" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1010" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Decision</text>',
        '<text x="1010" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">Broad vector16 rejected</text>',
        '<text x="1010" y="244" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">FP32 falls to 0.845×–0.879×</text>',
        '<text x="1010" y="292" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">Selective route kept</text>',
        '<text x="1010" y="325" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">FP16/BF16 ≥4M + aligned</text>',
        '<text x="1010" y="358" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">1.277×–1.411× vs scalar</text>',
        '<text x="1010" y="391" font-family="Inter,Arial,sans-serif" font-size="15" fill="#198754">all outputs/gradients exact</text>',
        '<text x="1010" y="424" font-family="Inter,Arial,sans-serif" font-size="15" fill="#198754">allocator peaks unchanged</text>',
        '<rect x="250" y="620" width="18" height="18" rx="3" fill="#d97706"/><text x="278" y="635" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">broad vector16</text>',
        '<rect x="440" y="620" width="18" height="18" rx="3" fill="#3769d6"/><text x="468" y="635" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">selective vector16</text>',
        '<text x="660" y="685" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep only low precision, aligned, bandwidth-scale dispatch</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

