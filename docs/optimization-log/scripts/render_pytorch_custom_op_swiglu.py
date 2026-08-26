#!/usr/bin/env python3
"""Render fused SwiGLU Custom Op forward/backward evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = summary["groups"]

    def find(kind: str, dtype: str, shape: str) -> dict:
        return next(row for row in groups if row["kind"] == kind and
                    row["dtype"] == dtype and row["shape"] == shape)

    rows = [(f"{dtype.upper()} forward 16M", find("forward", dtype, "bandwidth"))
            for dtype in ("fp32", "fp16", "bf16")]
    rows += [(f"{dtype.upper()} F+B 1M", find("forward_backward", dtype, "large"))
             for dtype in ("fp32", "fp16", "bf16")]
    width, height = 1320, 720
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1320" height="720" fill="#f7f9fc"/>',
        '<text x="660" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">Fused SwiGLU Wins Forward, Not Training Yet</text>',
        '<text x="660" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">Torch / microLLM Event ratio · six fresh MI300X processes · 15 correctness/performance cases</text>',
        '<line x1="80" y1="520" x2="935" y2="520" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
        y = 520 - tick / 2.0 * 360
        lines.append(f'<line x1="80" y1="{y:.1f}" x2="935" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(f'<text x="68" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.1f}×</text>')
    for index, (label, row) in enumerate(rows):
        center = 150 + index * 135
        value = row["event_speedup_median"]
        color = "#198754" if row["kind"] == "forward" else "#b42335"
        height_bar = value / 2.0 * 360
        y = 520 - height_bar
        lines.append(f'<rect x="{center - 32}" y="{y:.1f}" width="64" height="{height_bar:.1f}" rx="7" fill="{color}"/>')
        lines.append(f'<text x="{center}" y="{y - 9:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="#172033">{value:.3f}×</text>')
        lines.append(f'<text x="{center}" y="550" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#172033" transform="rotate(25 {center} 550)">{label}</text>')
    lines.extend([
        '<rect x="980" y="125" width="285" height="390" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="1010" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Evidence boundary</text>',
        '<text x="1010" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ 15 / 15 precision gates</text>',
        '<text x="1010" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ FP32 / FP16 / BF16</text>',
        '<text x="1010" y="277" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ forward peak halves at 16M</text>',
        '<text x="1010" y="325" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">Forward 1.142×–1.570×</text>',
        '<text x="1010" y="373" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">F+B only 0.597×–0.761×</text>',
        '<text x="1010" y="406" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">no training speed claim</text>',
        '<text x="1010" y="439" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">backward is next isolated target</text>',
        '<text x="660" y="685" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Keep explicit fused API · admit large forward evidence · reject training promotion</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

