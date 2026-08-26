#!/usr/bin/env python3
"""Render rejected vector4 and retained scalar SwiGLU backward evidence."""

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
    width, height = 1220, 690
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1220" height="690" fill="#f7f9fc"/>',
        '<text x="610" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">SwiGLU Backward · Vector4 Was the Wrong Explanation</text>',
        '<text x="610" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">six fresh MI300X processes · complete two-gradient comparison · Event ratios</text>',
        '<line x1="80" y1="490" x2="790" y2="490" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.0, 1.0, 2.0, 3.0):
        y = 490 - tick / 3.0 * 330
        lines.append(f'<line x1="80" y1="{y:.1f}" x2="790" y2="{y:.1f}" stroke="#dbe1ea"/>')
        lines.append(f'<text x="68" y="{y + 5:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">{tick:.1f}×</text>')
    for index, row in enumerate(groups):
        center = 160 + index * 165
        vector = row["vector_vs_scalar_event_median"]
        native = row["vector_vs_native_event_median"]
        for offset, value, color in ((-30, vector, "#b42335"),
                                     (30, native, "#198754")):
            height_bar = value / 3.0 * 330
            y = 490 - height_bar
            lines.append(f'<rect x="{center + offset - 24}" y="{y:.1f}" width="48" height="{height_bar:.1f}" rx="6" fill="{color}"/>')
            lines.append(f'<text x="{center + offset}" y="{y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="#172033">{value:.3f}</text>')
        lines.append(f'<text x="{center}" y="520" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="14" fill="#172033">{row["shape"]} · {row["elements"]}</text>')
    lines.extend([
        '<rect x="835" y="125" width="325" height="365" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="865" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">What changed</text>',
        '<text x="865" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ Max vs native 1.19e-7</text>',
        '<text x="865" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ scalar peak 1/3 below native</text>',
        '<text x="865" y="291" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">✗ vector/scalar 0.946×–1.039×</text>',
        '<text x="865" y="324" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">✗ 0 / 2 large gates pass 1.05</text>',
        '<text x="865" y="371" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">scalar already 2.07×–2.82×</text>',
        '<text x="865" y="404" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">vs readable native formula</text>',
        '<text x="865" y="447" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: remove zero-stride materialization</text>',
        '<rect x="250" y="590" width="18" height="18" rx="3" fill="#b42335"/><text x="278" y="605" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">vector / scalar</text>',
        '<rect x="440" y="590" width="18" height="18" rx="3" fill="#198754"/><text x="468" y="605" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">vector / native formula</text>',
        '<text x="610" y="655" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Delete vector candidate · keep scalar producer · move to gradient layout</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

