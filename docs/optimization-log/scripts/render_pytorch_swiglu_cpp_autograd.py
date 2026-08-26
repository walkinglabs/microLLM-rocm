#!/usr/bin/env python3
"""Render Python-to-C++ SwiGLU Autograd admission evidence."""

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
    groups = report["groups"]
    width, height = 1320, 720
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1320" height="720" fill="#f7f9fc"/>',
        '<text x="660" y="48" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="30" font-weight="700" fill="#172033">C++ Autograd Closes the Python Callback Gap</text>',
        '<text x="660" y="78" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="16" fill="#5b6474">six fresh MI300X processes · complete loss/gradient/peak gates</text>',
        '<line x1="75" y1="500" x2="920" y2="500" stroke="#aab3c2" stroke-width="2"/>',
    ]
    for tick in (0.0, 0.5, 1.0, 1.5):
        y = 500 - tick / 1.5 * 340
        lines.append(f'<line x1="75" y1="{y:.1f}" x2="920" y2="{y:.1f}" stroke="#dbe1ea"/>')
    for index, row in enumerate(groups):
        center = 135 + index * 135
        py = row["cpp_vs_python_event"]
        native = row["cpp_vs_native_event"]
        for offset, value, color in ((-24, py, "#198754"), (24, native, "#3769d6")):
            height_bar = value / 1.5 * 340
            y = 500 - height_bar
            lines.append(f'<rect x="{center + offset - 18}" y="{y:.1f}" width="36" height="{height_bar:.1f}" rx="5" fill="{color}"/>')
        lines.append(f'<text x="{center - 24}" y="{490 - py / 1.5 * 340:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{py:.3f}</text>')
        lines.append(f'<text x="{center + 24}" y="{490 - native / 1.5 * 340:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#172033">{native:.3f}</text>')
        lines.append(f'<text x="{center}" y="530" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" fill="#172033">{row["dtype"]} {row["shape"]}</text>')
    lines.extend([
        '<rect x="965" y="125" width="300" height="375" rx="18" fill="#ffffff" stroke="#dbe1ea"/>',
        '<text x="995" y="167" font-family="Inter,Arial,sans-serif" font-size="20" font-weight="700" fill="#172033">Admission</text>',
        '<text x="995" y="211" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ C++/Python 1.286×–1.475×</text>',
        '<text x="995" y="244" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ FP32/native 1.136×–1.144×</text>',
        '<text x="995" y="277" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ FP32 peak only 1536 B</text>',
        '<text x="995" y="310" font-family="Inter,Arial,sans-serif" font-size="16" fill="#198754">✓ low-precision peak = native</text>',
        '<text x="995" y="358" font-family="Inter,Arial,sans-serif" font-size="16" fill="#b42335">low F+B still 0.799×–0.812×</text>',
        '<text x="995" y="401" font-family="Inter,Arial,sans-serif" font-size="15" fill="#5b6474">next: typed fused backward</text>',
        '<rect x="300" y="610" width="18" height="18" rx="3" fill="#198754"/><text x="328" y="625" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">C++ / Python Autograd</text>',
        '<rect x="550" y="610" width="18" height="18" rx="3" fill="#3769d6"/><text x="578" y="625" font-family="Inter,Arial,sans-serif" font-size="14" fill="#5b6474">C++ / native Torch</text>',
        '<text x="660" y="678" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#172033">Recommend C++ Autograd as the optional adapter default</text>',
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

